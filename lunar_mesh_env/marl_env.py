# marl_env.py

import numpy as np
import functools
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pygame
from collections import defaultdict

from .marl_entities import MarlMeshAgent as MeshAgent, BaseStation
from .radio_model_nn import RadioMapModelNN
from .pathfinding import a_star_search

class LunarRoverMeshEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "name": "lunar_mesh_v1",
        "render_fps": 4
    }

    def __init__(self, 
                 hm_path='../radio_data_2/radio_data_2/hm/hm_18.npy',
                 radio_model: RadioMapModelNN = None,
                 num_agents=3,
                 render_mode=None):
        
        self.render_mode = render_mode
        self.hm_path = hm_path
        self.radio_model = radio_model
        
        # Physics constants
        self.width = 256.0 
        self.height = 256.0
        self.ROVER_SPEED = 0.2
        self.STEP_LENGTH = 20
        self.METERS_PER_PIXEL = 1.0
        self.MAX_DIST_PER_STEP = (self.ROVER_SPEED * self.STEP_LENGTH) / self.METERS_PER_PIXEL
        
        # restricts movement on steep inclines
        self.MAX_INCLINE_PER_STEP = self.MAX_DIST_PER_STEP*0.2
        
        
        # Energy & Rewards
        self.START_ENERGY = 50000.0
        self.COST_MOVE_PER_STEP = 5.0
        self.COST_TX_5G_PER_STEP = 1.0
        self.COST_TX_415_PER_STEP = 2.0
        self.COST_IDLE_PER_STEP = 0.1
        self.EP_MAX_TIME = 300
        
        # Reward Config
        self.REWARD_PEER_LINK = 1.0
        self.REWARD_BS_LINK = 5.0 
        
        # The rovers know how to reach the goal (preset path)
        # but since we allow them to leave the path for comms, 
        # we need to reward them for arriving.
        self.REWARD_GOAL_ARRIVAL = 100.0 
        self.REWARD_DIST_SCALE = 2.0      
        self.PENALTY_FAIL = -0.1
        self.PENALTY_INVALID_MOVE = -1.0 

        # Agent set up
        # petting zoo style 
        self.possible_agents = [f"rover_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # bridges petting zoo agent_id to actual agent object
        self.agent_map = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_map[agent_id] = MeshAgent(ue_id=i+1)

        self.base_station = BaseStation(x=0.0, y=0.0)

        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        # Load Heightmap
        self.heightmap = None
        try:
            self.heightmap = np.load(self.hm_path)
        except Exception as e:
            print(f"Warning: Could not load heightmap: {e}")
            self.heightmap = np.zeros((int(self.height), int(self.width)))

        # Cache & History
        self.radio_cache = {} 
        self.connections = defaultdict(set)
        self.custom_links = {} 
        self.all_visible_links = []
        self.sim_time = 0
        self.history = {"datarate": [], "energy": []}
        self.total_energy_consumed_step = 0.0

        # Pygame
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.sim_time = 0
        self.connections = defaultdict(set)
        self.custom_links = {}
        self.all_visible_links = []
        self.history = {"datarate": [], "energy": []}
        
        # randomly place bs
        new_bs_x, new_bs_y = np.random.uniform(0, self.width, size=(2,))
        self.base_station.x = new_bs_x
        self.base_station.y = new_bs_y
        
        safety_factor = 0.5
        per_pixel_threshold = (self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)) * safety_factor
        # reset agent states
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            agent.energy = self.START_ENERGY
            agent.total_distance = 0.0
            agent.active_route = None
            agent.current_datarate = 0.0
            
            agent.x = np.random.uniform(0, self.width)
            agent.y = np.random.uniform(0, self.height)
            
            # generate task, right now it's set to be far
            # away from initial pos
            max_retries = 100
            for _ in range(max_retries):

                # agent start
                agent.x = np.random.uniform(0, self.width)
                agent.y = np.random.uniform(0, self.height)
                
                # goal
                gx = np.random.uniform(0, self.width)
                gy = np.random.uniform(0, self.height)
                
                dist = np.sqrt((gx - agent.x)**2 + (gy - agent.y)**2)
                
                if dist > 50.0:
                    # make sure there is valid path to goal
                    start_node = (int(agent.x), int(agent.y))
                    end_node = (int(gx), int(gy))
                    path = a_star_search(self.heightmap, start_node, end_node, per_pixel_threshold)
                    
                    if path:
                        agent.goal_x = gx
                        agent.goal_y = gy
                        agent.nav_path = path 
                        break
            
        self._update_radio_cache_batch()

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def step(self, actions):
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        self.total_energy_consumed_step = 0.0
        
        per_pixel_threshold = self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)
        
        # movement update
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            
            # follow path to objective
            if agent.nav_path and len(agent.nav_path) > 1:
                # only look at the next 15 nodes to find closest
                search_window = agent.nav_path[:15] 

                dists = [np.sqrt((n[0]-agent.x)**2 + (n[1]-agent.y)**2) for n in search_window]
                
                closest_idx = np.argmin(dists)
                
                # remove passed nodes
                if closest_idx > 0:
                    agent.nav_path = agent.nav_path[closest_idx:]

            prev_dist = np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2)
            
            move_cmd = action[0] 
            agent.current_datarate = 0.0 
            
            dx, dy = 0, 0
            
            # cardinal directions
            if move_cmd == 1: dy = 1        # North
            elif move_cmd == 2: dy = -1     # South
            elif move_cmd == 3: dx = -1     # West
            elif move_cmd == 4: dx = 1      # East
            
            # diagonals
            elif move_cmd == 5: dx, dy = 0.707, 0.707   # NE
            elif move_cmd == 6: dx, dy = -0.707, 0.707  # NW
            elif move_cmd == 7: dx, dy = 0.707, -0.707  # SE
            elif move_cmd == 8: dx, dy = -0.707, -0.707 # SW
            
            step_energy = 0.0
            
            if dx != 0 or dy != 0:
                dist = np.sqrt(dx**2 + dy**2)
                scale = self.MAX_DIST_PER_STEP / dist
                new_x = np.clip(agent.x + dx*scale, 0, self.width - 1)
                new_y = np.clip(agent.y + dy*scale, 0, self.height - 1)
                
                # slope check
                current_z = self.heightmap[int(agent.y), int(agent.x)]
                target_z = self.heightmap[int(new_y), int(new_x)]
                height_diff = target_z - current_z
                
                if height_diff > self.MAX_INCLINE_PER_STEP:
                    # blocked
                    # print("blocked action")
                    rewards[agent_id] += self.PENALTY_INVALID_MOVE
                    step_energy = self.COST_IDLE_PER_STEP 
                else:
                    #  allowed
                    agent.x, agent.y = new_x, new_y
                    agent.total_distance += self.MAX_DIST_PER_STEP
                    incline_cost = max(0, height_diff * 0.5) 
                    step_energy = self.COST_MOVE_PER_STEP + incline_cost
            else:
                step_energy = self.COST_IDLE_PER_STEP
            
            agent.energy -= step_energy
            self.total_energy_consumed_step += step_energy

            curr_dist = np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2)
            
            # dense reward for getting closer
            # reward function has not been tuned extensively
            # so this could be changed later
            dist_delta = prev_dist - curr_dist 
            rewards[agent_id] += dist_delta * self.REWARD_DIST_SCALE

            # arrival logic
            if curr_dist < (self.MAX_DIST_PER_STEP*2): 
                rewards[agent_id] += self.REWARD_GOAL_ARRIVAL
                infos[agent_id]['task_complete'] = True
                
                # respawn
                max_retries = 20
                for _ in range(max_retries):
                    gx = np.random.uniform(0, self.width)
                    gy = np.random.uniform(0, self.height)
                    dist = np.sqrt((gx - agent.x)**2 + (gy - agent.y)**2)
                    
                    if dist > 50.0:
                        start_node = (int(agent.x), int(agent.y))
                        end_node = (int(gx), int(gy))
                        path = a_star_search(self.heightmap, start_node, end_node, per_pixel_threshold)
                        
                        if path:
                            agent.goal_x = gx
                            agent.goal_y = gy
                            agent.nav_path = path
                            break
            else:
                infos[agent_id]['task_complete'] = False

        self._update_radio_cache_batch()

        # comms
        self.connections = defaultdict(set) 
        self.custom_links = {} 
        self.all_visible_links = []
        
        num_rovers = len(self.agents)
        # bs_target_idx = num_rovers + 1 

        for agent_id, action in actions.items():
            if terminations[agent_id]: continue
            
            agent = self.agent_map[agent_id]
            move_cmd = action[0]
            comm_flags = action[1:]
            # print(comm_flags)
            radio_map = self._get_cached_radio_map(agent)
            agent.active_route = ([], '5.8')
            for i in range(num_rovers):
                if comm_flags[i] == 1:
                    target_id = self.possible_agents[i]
                    if target_id != agent_id:
                        target_entity = self.agent_map[target_id]
                        rssi = self._get_signal_strength(radio_map, target_entity)
                        
                        if rssi > -90.0: 
                            self.connections[agent].add(target_entity)
                            rewards[agent_id] += self.REWARD_PEER_LINK
                            self.custom_links[(agent, target_entity)] = 'green'
                            tx_cost = self.COST_TX_5G_PER_STEP
                            agent.energy -= tx_cost
                            self.total_energy_consumed_step += tx_cost
                            old_active = agent.active_route[0]
                            new_active = old_active + [(agent, target_entity)]
                            agent.active_route = (new_active, '5.8')
                        else:
                            rewards[agent_id] += self.PENALTY_FAIL
            agent.current_datarate = 0.0
            if comm_flags[-1] == 1:
                rssi_bs = self._get_signal_strength(radio_map, self.base_station)
                if rssi_bs > -90.0:
                    self.connections[agent].add(self.base_station)
                    rewards[agent_id] += self.REWARD_BS_LINK
                    self.custom_links[(agent, self.base_station)] = 'cyan' 
                    tx_cost = self.COST_TX_5G_PER_STEP
                    agent.energy -= tx_cost
                    self.total_energy_consumed_step += tx_cost
                    agent.current_datarate = 500.0
                    old_active = agent.active_route[0]
                    new_active = old_active + [(agent, self.base_station)]
                    agent.active_route = (new_active, '5.8')
                else:
                    rewards[agent_id] += self.PENALTY_FAIL

            num_active_comms = len(self.connections[agent])
            agent.current_datarate += num_active_comms * 100.0
            

        # color possible links
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            if agent.energy <= 0: continue
            
            rm = self._get_cached_radio_map(agent)
            if rm is None: continue

            # bs
            rssi_bs = self._get_signal_strength(rm, self.base_station)
            if rssi_bs > -90.0:
                self.all_visible_links.append((agent, self.base_station))
            
            #  agents
            for other_id in self.agents:
                if agent_id == other_id: continue
                other = self.agent_map[other_id]
                if other.energy <= 0: continue

                rssi_other = self._get_signal_strength(rm, other)
                if rssi_other > -90.0:
                     self.all_visible_links.append((agent, other))

        # metrics & termination checks
        self.sim_time += 1
        global_truncate = self.sim_time >= self.EP_MAX_TIME

        active_datarates = [self.agent_map[a].current_datarate for a in self.agents if not terminations[a]]
        avg_datarate = np.mean(active_datarates) if active_datarates else 0.0
        self.history['datarate'].append(avg_datarate)
        self.history['energy'].append(self.total_energy_consumed_step)
        
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                terminations[agent_id] = True
            truncations[agent_id] = global_truncate
            infos[agent_id] = {"energy": agent.energy, "pos": (agent.x, agent.y)}

        active_this_step = list(actions.keys())
        
        global_truncate = self.sim_time >= self.EP_MAX_TIME
        for agent_id in active_this_step:
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                terminations[agent_id] = True
            truncations[agent_id] = global_truncate
            infos[agent_id] = {"energy": agent.energy, "pos": (agent.x, agent.y)}

        observations = {a: self._get_obs(a) for a in active_this_step}
        
        self.agents = [a for a in self.agents if not terminations.get(a, False) and not truncations.get(a, False)]
        
        return observations, rewards, terminations, truncations, infos

    def heuristic_move_action(self, agent_id):
        """
        picks movement action based on pure pursuit to follow the precomputed path
        """
        agent = self.agent_map[agent_id]
        
        if not agent.nav_path:
            return 0 
        
        # path is in pixel coords, but our actions are in directions
        # so we need to find the next target node along the path
        step_capacity = int(self.MAX_DIST_PER_STEP)
        safe_lookahead = max(1, step_capacity - 2)
        lookahead = min(len(agent.nav_path) - 1, safe_lookahead) 
        
        # Scan forward if needed (Pure Pursuit logic from previous step)
        lookahead_radius = self.MAX_DIST_PER_STEP * 1.5
        target_node = agent.nav_path[-1]
        for node in agent.nav_path:
            d_sq = (node[0] - agent.x)**2 + (node[1] - agent.y)**2
            if d_sq > lookahead_radius**2:
                target_node = node
                break

        if target_node == agent.nav_path[-1]:
             tx, ty = agent.goal_x, agent.goal_y
        else:
             tx, ty = target_node


        dx = tx - agent.x
        dy = ty - agent.y

        if abs(dx) < 0.5 and abs(dy) < 0.5:
             return 0
        
        # determine angle
        angle = np.arctan2(dy, dx)
        
        sector_idx = int(np.round(angle / (np.pi / 4))) % 8
        
        # map sector to action
        map_sector_to_action = {
            0: 4, # East
            1: 5, # NE
            2: 1, # North
            3: 6, # NW
            4: 3, # West
            5: 8, # SW
            6: 2, # South
            7: 7  # SE
        }

        check_order = [0, 1, -1, 2, -2] 
        
        for offset in check_order:
            if check_order == 1:
                print('failed initial move')
            check_idx = (sector_idx + offset) % 8
            proposed_action = map_sector_to_action[check_idx]
            
            if self._is_move_valid(agent, proposed_action):
                return proposed_action
                
        # all moves are blocked
        return 0
    

    def _is_move_valid(self, agent, move_cmd):
        """Helper to check if a move is valid (in bounds and valid slope)"""
        if move_cmd == 0: return True # Idle is always valid
        
        dx, dy = 0, 0
        # cardinal
        if move_cmd == 1: dy = 1        # N
        elif move_cmd == 2: dy = -1     # S
        elif move_cmd == 3: dx = -1     # W
        elif move_cmd == 4: dx = 1      # E
        # diagonal
        elif move_cmd == 5: dx, dy = 0.707, 0.707   # NE
        elif move_cmd == 6: dx, dy = -0.707, 0.707  # NW
        elif move_cmd == 7: dx, dy = 0.707, -0.707  # SE
        elif move_cmd == 8: dx, dy = -0.707, -0.707 # SW

        dist = np.sqrt(dx**2 + dy**2)
        scale = self.MAX_DIST_PER_STEP / dist
        
        new_x = np.clip(agent.x + dx*scale, 0, self.width - 1)
        new_y = np.clip(agent.y + dy*scale, 0, self.height - 1)

        current_z = self.heightmap[int(agent.y), int(agent.x)]
        target_z = self.heightmap[int(new_y), int(new_x)]
        height_diff = target_z - current_z
        
        if height_diff > self.MAX_INCLINE_PER_STEP:
            return False # blocked by slope
            
        return True
    
    def _update_radio_cache_batch(self):
        """
        Updates the radio maps for all active agents in a single batch.
        Checks cache first to avoid re-generating maps for stationary agents.
        """
        if not self.radio_model:
            return

        needed_indices = []
        needed_positions = []
        needed_ids = []

        # find which agents need new maps
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            
            key = (int(agent.x), int(agent.y), '5.8') 
            
            if key not in self.radio_cache:
                needed_ids.append(agent_id)
                needed_positions.append((agent.x, agent.y))

        # batch inference
        if len(needed_positions) > 0:
            new_maps = self.radio_model.generate_map_batch(needed_positions, '5.8')
            
            # update cache
            for i, agent_id in enumerate(needed_ids):
                agent = self.agent_map[agent_id]
                key = (int(agent.x), int(agent.y), '5.8')
                self.radio_cache[key] = new_maps[i]

    def _get_cached_radio_map(self, agent):
        """Retrieves map from cache. Assumes _update_radio_cache_batch was called."""
        grid_pos = (int(agent.x), int(agent.y), '5.8')
        return self.radio_cache.get(grid_pos, None)


    def _get_radio_map(self, agent):
        """Cached Neural Network Inference"""
        grid_pos = (int(agent.x), int(agent.y), '5.8') 
        
        if grid_pos not in self.radio_cache:
            if self.radio_model:
                self.radio_cache[grid_pos] = self.radio_model.generate_map((agent.x, agent.y), '5.8')
            else:
                return None
        return self.radio_cache[grid_pos]

    def _get_signal_strength(self, radio_map, target_agent):
        """Gets the signal strength (dBm) from a given radio_map at target's location."""
        if radio_map is None: return -150.0
        r_idx = int((target_agent.y / self.height) * (radio_map.shape[0]-1))
        c_idx = int((target_agent.x / self.width) * (radio_map.shape[1]-1))
        r_idx = np.clip(r_idx, 0, radio_map.shape[0]-1)
        c_idx = np.clip(c_idx, 0, radio_map.shape[1]-1)
        return radio_map[r_idx, c_idx]

    def _get_obs(self, agent_id):
        agent = self.agent_map[agent_id]
        all_agents = list(self.agent_map.values())
        bs_loc = self.base_station.get_position()
        current_rm = self._get_cached_radio_map(agent)
        
        safe_rm = np.clip(current_rm, -190.0, 0.0).astype(np.float32)

        obs_dict = agent.get_local_observation(self.heightmap.astype(np.float32), all_agents, bs_loc, safe_rm)

        num_rovers = len(self.possible_agents)
        move_mask = np.zeros(9, dtype=np.int8)
        move_mask[0] = 1
        comm_mask = np.zeros(num_rovers + 1, dtype=np.int8)

        for move_cmd in range(9):
            if self._is_move_valid(agent, move_cmd):
                move_mask[move_cmd] = 1

        for i, target_id in enumerate(self.possible_agents):
            if target_id == agent_id:
                comm_mask[i] = 0
                continue
            target_entity = self.agent_map[target_id]
            rssi = self._get_signal_strength(current_rm, target_entity)
            comm_mask[i] = 1 if (rssi > -90.0 and target_entity.energy > 0) else 0

        rssi_bs = self._get_signal_strength(current_rm, self.base_station)
        comm_mask[-1] = 1 if rssi_bs > -90.0 else 0

        obs_dict["action_mask"] = np.concatenate([move_mask, comm_mask]).astype(np.int8)
        
        if len(obs_dict['terrain'].shape) == 2:
            obs_dict['terrain'] = np.expand_dims(obs_dict['terrain'], axis=0)
          
        # have to cast everything to correct dtypes for rllib  
        final_obs = {
            "self_state": obs_dict["self_state"].astype(np.float32),
            "terrain": obs_dict["terrain"].astype(np.float32),
            "neighbors": obs_dict["neighbors"].astype(np.float32),
            "base_station": obs_dict["base_station"].astype(np.float32),
            "goal_vector": obs_dict["goal_vector"].astype(np.float32),
            "radio_map": obs_dict["radio_map"].astype(np.float32),
            "action_mask": obs_dict["action_mask"].astype(np.int8)
        }

        # safety check for NaNs/Infs
        for key, val in final_obs.items():
            if not np.all(np.isfinite(val)):
                final_obs[key] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        
        return final_obs
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num_rovers = len(self.possible_agents)
        # 9 movement + (num_rovers + bs) comm targets
        mask_dim = 9 + (num_rovers + 1) 
        
        return spaces.Dict({
            "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "terrain": spaces.Box(low=0, high=496, shape=(1, 256, 256), dtype=np.float32),
            "neighbors": spaces.Box(low=-np.inf, high=np.inf, shape=(num_rovers - 1, 2), dtype=np.float32),
            "base_station": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "goal_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "radio_map": spaces.Box(low=-200, high=0, shape=(1, 256, 256), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(mask_dim,), dtype=np.int8)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # action [0]movement
        # 0: Idle
        # 1-4: N, S, W, E
        # 5-8: NE, NW, SE, SW
        num_rovers = len(self.possible_agents)
        # action[1-N]: comm target
        # action[N+1]: base station
        shape = [9] + [2]*(num_rovers+1)
        # print(shape)
        return spaces.MultiDiscrete(shape)

    def render(self):
        if self.render_mode is None: return

        n_agents = len(self.possible_agents)
        width_ratios = [4] + ([4] * n_agents) + [3.5]
        num_cols = len(width_ratios)
        fx = 5.0 * num_cols
        fy = max(1.25 * self.height / 100.0 * 4.0, 10.0)
        
        plt.close()
        fig = plt.figure(figsize=(fx, fy), dpi=65)

        gs = fig.add_gridspec(
            ncols=num_cols, nrows=3,
            width_ratios=width_ratios, height_ratios=(3, 3, 3),
            hspace=0.45, wspace=0.6,
            top=0.95, bottom=0.15, left=0.025, right=0.955,
        )

        sim_ax = fig.add_subplot(gs[:, 0])
        dash_ax = fig.add_subplot(gs[0, -1]) 
        qoe_ax = fig.add_subplot(gs[1, -1])  
        conn_ax = fig.add_subplot(gs[2, -1]) 

        self.render_simulation(sim_ax)

        # render radio maps
        for i, agent_id in enumerate(self.possible_agents):
            col_idx = i + 1
            map_ax = fig.add_subplot(gs[:, col_idx])
            if agent_id in self.agent_map:
                agent = self.agent_map[agent_id]
                if agent.energy > 0:
                    radio_map = self._get_cached_radio_map(agent)
                    label_char = chr(ord('A') + i) 
                    self.render_radio_map(map_ax, radio_map, f"Radio Map (Agent {label_char})")
                else:
                    map_ax.text(0.5, 0.5, "SIGNAL LOST\n(Dead)", ha='center', va='center')
                    map_ax.axis('off')
            else:
                map_ax.axis('off')

        self.render_dashboard(dash_ax)
        self.render_avg_datarate(qoe_ax)
        self.render_avg_energy(conn_ax)

        fig.align_ylabels((qoe_ax, conn_ax))
        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.close()

        if self.render_mode == "rgb_array":
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            return data.reshape(canvas.get_width_height()[::-1] + (3,))
        elif self.render_mode == "human":
            self._render_human(canvas, fig)

    def render_simulation(self, ax):
        if self.heightmap is not None:
            ax.imshow(self.heightmap, cmap='terrain', extent=[0, self.width, 0, self.height], origin='lower', zorder=0)

        #draw bs
        ax.scatter(self.base_station.x, self.base_station.y, s=300, c='cyan', marker='^', edgecolors='black', zorder=3, label="Base Station")
        ax.annotate("BS", xy=(self.base_station.x, self.base_station.y), xytext=(0, 10), textcoords='offset points', ha='center', color='cyan', fontweight='bold')

        # draw agents and goals
        for agent_id, agent in self.agent_map.items():
            if agent.energy <= 0: continue
            
            color = 'gray'
            if agent_id == "rover_0": color = 'blue'
            elif agent_id == "rover_1": color = 'purple'
            elif agent_id == "rover_2": color = 'green'

            # agent
            ax.scatter(agent.x, agent.y, s=200, zorder=2, color=color, marker="o")
            
            # goal
            ax.scatter(agent.goal_x, agent.goal_y, s=200, zorder=1, color=color, marker="x", linewidth=3)

            if agent.nav_path and len(agent.nav_path) > 1:
                path_x = [p[0] for p in agent.nav_path]
                path_y = [p[1] for p in agent.nav_path]
                ax.plot(path_x, path_y, color=color, linestyle='-', linewidth=1, alpha=0.6, zorder=1)
                
                
            label_map = {"rover_0": "A", "rover_1": "B", "rover_2": "C"}
            label = label_map.get(agent_id, agent_id[-1])
            ax.annotate(label, xy=(agent.x, agent.y), ha="center", va="center", color='white', fontweight='bold')

        # possible links
        for (u, v) in self.all_visible_links:
            ax.plot([u.x, v.x], [u.y, v.y], color='red', 
                    linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)

        # active links
        for (u, v), color in self.custom_links.items():
            vx, vy = v.x, v.y 
            ax.plot([u.x, vx], [u.y, vy], color=color, 
                    path_effects=[pe.SimpleLineShadow(shadow_color="black"), pe.Normal()],
                    linewidth=3, zorder=1.5) 

        ax.axis('off')
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])


    def render_dashboard(self, ax):
        ax.axis('off')
        ax.set_title(f"Sim Time: {self.sim_time * self.STEP_LENGTH:.1f} s", fontweight='bold')

        # dynamically generate rows for all possible agents
        rows = []
        cell_text = []
        
        cols = ["Rate (Mbps)", "Energy", "Dist (m)"]

        for i, agent_id in enumerate(self.possible_agents):
            label_char = chr(ord('A') + i)
            rows.append(f"Agent {label_char}")

            if agent_id in self.agent_map:
                agent = self.agent_map[agent_id]
                cell_text.append([
                    f"{agent.current_datarate:.1f}",
                    f"{agent.energy:.1f}",
                    f"{agent.total_distance:.1f}"
                ])
            else:
                cell_text.append(["N/A", "Dead", "N/A"])

        table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, 
                         cellLoc="center", loc="upper center", bbox=[0.0, -0.25, 1.0, 1.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        
    def _render_human(self, canvas, fig):

        data = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        
        if self.window is None:
            pygame.init()
            self.clock = pygame.time.Clock()

            self.window = pygame.display.set_mode((width+(width/10.0), height))
            pygame.display.set_caption("Lunar Mesh Env (MARL)")

        # specify RGBA for conversion from matplotlib
        plot = pygame.image.frombuffer(data, (width, height), "RGBA")
        
        # draw
        self.window.fill("white")
        self.window.blit(plot, (0, 0))
        pygame.display.flip()
        
        self.clock.tick(self.metadata["render_fps"])

        pygame.event.pump()

        
    def render_radio_map(self, ax, radio_map, title):
        ax.set_title(title)
        if radio_map is not None:
            im = ax.imshow(radio_map, cmap='viridis', origin='lower', 
                           extent=[0, self.width, 0, self.height], vmin=-130, vmax=0)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.figure.colorbar(im, cax=cax, orientation="vertical", label="Signal (dBm)")
        
        # overlay Agents
        for agent_id, agent in self.agent_map.items():
            if agent.energy <= 0: continue
            color = 'blue' if agent_id == "rover_0" else ('purple' if agent_id == "rover_1" else 'gray')
            ax.scatter(agent.x, agent.y, s=100, color=color, edgecolors='white')

        ax.axis('off')
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])


    def render_avg_datarate(self, ax):
        if len(self.history['datarate']) > 0:
            ax.plot(self.history['datarate'], color="black")
        ax.set_ylabel("Avg Rate (Mbps)")
        ax.set_xlim([0, self.EP_MAX_TIME])
        ax.set_ylim([0, 550])

    def render_avg_energy(self, ax):
        if len(self.history['energy']) > 0:
            ax.plot(self.history['energy'], color="black")
        ax.set_ylabel("Step Energy")
        ax.set_xlabel("Step")
        ax.set_xlim([0, self.EP_MAX_TIME])
        ax.set_ylim([0, 100])