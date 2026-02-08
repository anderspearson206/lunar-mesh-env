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
import random

from .marl_entities import MarlMeshDTNAgent as MeshAgent, BaseStation
from .radio_model_nn import RadioMapModelNN
from .pathfinding import a_star_search_rm as a_star_search


PACKET_SIZE_BITS = 100 * 1024 # 100 Kb per packet
BURST_SIZE_MBITS = 10.0       # A "science event" generates 50 Mb of data
TELEMETRY_RATE_MBPS = 0.5     # Constant 0.5 Mbps background stream
BURST_PROBABILITY = 0.30

def set_seed(seed=42):
    """Sets all possible seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

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
                 radio_bias=0.0,
                 render_mode=None,
                 num_goals = 5,
                 packet_mode = 'boolean',
                 seed=206):
        
        self.render_mode = render_mode
        self.hm_path = hm_path
        self.radio_model = radio_model
        self.radio_bias = radio_bias
        self.num_goals = num_goals
        self.packet_mode = packet_mode
        self.seed = seed
        set_seed(self.seed)
        # Physics constants
        self.width = 256.0 
        self.height = 256.0
        
        # this was pulled from the viper 
        self.ROVER_SPEED = 0.2
        # 
        self.STEP_LENGTH = 20
        self.METERS_PER_PIXEL = 1.0
        self.MAX_DIST_PER_STEP = (self.ROVER_SPEED * self.STEP_LENGTH) / self.METERS_PER_PIXEL
        
        # restricts movement on steep inclines
        self.MAX_INCLINE_PER_STEP = self.MAX_DIST_PER_STEP*0.25
        
        
        # Energy & Rewards
        self.START_ENERGY = 50000.0
        self.COST_MOVE_PER_STEP = 5.0
        self.COST_TX_5G_PER_STEP = 1.0
        self.COST_TX_415_PER_STEP = 2.0
        self.COST_IDLE_PER_STEP = 0.1
        self.EP_MAX_TIME = 300
        self.MIN_DBM_THRESHOLD = -82.0 # look at get_throughput_rss() in radio_model_nn.py
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


        # DTN config
        self.PACKET_GEN_PROB = 0.5
        self.REWARD_PACKET_DELIVERY = 20.0 
        self.PENALTY_BUFFER_OVERFLOW = -5.0
        

        self.base_station = BaseStation(x=0.0, y=0.0)

        

        
        # Load Heightmap
        self.heightmap = None
        try:
            self.heightmap = np.load(self.hm_path).astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not load heightmap: {e}")
            self.heightmap = np.zeros((int(self.height), int(self.width)), dtype=np.float32)
            
            
        # Agent set up
        # petting zoo style 
        self.possible_agents = [f"rover_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.buffer_capacity_bits = 1000 * 1024 * 1024
        # bridges petting zoo agent_id to actual agent object
        self.agent_map = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_map[agent_id] = MeshAgent(ue_id=i+1, 
                                                 heightmap=self.heightmap,
                                                 radio_model=self.radio_model, 
                                                 bs=self.base_station, 
                                                 buffer_size=self.buffer_capacity_bits)


        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }
        
        # Cache & History
  
        self.connections = defaultdict(set)
        self.custom_links = {} 
        self.all_visible_links = []
        self.sim_time = 0
        self.agent_goals_completed = {aid: 0 for aid in self.possible_agents}
        self.mission_done = {aid: False for aid in self.possible_agents}
        self.history = {"datarate": [], "energy": [], "bs_link_ratio": [], "avg_rss": []}
        self.agent_rss_history = {aid: [] for aid in self.possible_agents}
        self.total_energy_consumed_step = 0.0
        self.bs_radio_map = self.radio_model.generate_map((self.base_station.x, self.base_station.y), '5.8') if self.radio_model else None
        # Pygame
        self.window = None
        self.clock = None
    
    @property
    def packets_generated(self):
        """Calculates total packets generated across all agents."""
        return sum(
            self.agent_map[aid].payload_manager.num_packets_generated
            for aid in self.possible_agents if aid in self.agent_map
        )

    @property
    def unique_packets_rcvd(self):
        """Total unique packets received by the base station."""
        return len(self.base_station.packets_received)

    @property
    def total_packets_rcvd(self):
        """Total raw packets (including duplicates) received by base station."""
        return self.base_station.num_packets_received
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.sim_time = 0
        self.connections = defaultdict(set)
        self.custom_links = {}
        self.all_visible_links = []
        self.history = {"datarate": [], "energy": [], "bs_link_ratio": [], "avg_rss": []}
        self.agent_rss_history = {aid: [] for aid in self.possible_agents}
        self.agent_goals_completed = {aid: 0 for aid in self.possible_agents}
        self.mission_done = {aid: False for aid in self.possible_agents}
        
        # randomly place bs
        rng_env = np.random.RandomState(self.seed)
        new_bs_x, new_bs_y = np.random.uniform(0, self.width, size=(2,))
        self.base_station.x = new_bs_x
        self.base_station.y = new_bs_y

        self.bs_radio_map = self.radio_model.generate_map((self.base_station.x, self.base_station.y), '5.8')
            
        safety_factor = 0.9
        per_pixel_threshold = (self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)) * safety_factor
        
        
        # reset agent states
        for i, agent_id in enumerate(self.agents):
            agent = self.agent_map[agent_id]
            agent.energy = self.START_ENERGY
            agent.total_distance = 0.0
            agent.active_route = None
            agent.current_datarate = 0.0
            agent_rng = rng = np.random.RandomState(self.seed+i)
            max_retries = 100
            for i in range(max_retries):
                candidate_x = agent_rng.uniform(0, self.width)
                candidate_y = agent_rng.uniform(0, self.height)
                
                gx = agent_rng.uniform(0, self.width)
                gy = agent_rng.uniform(0, self.height)
                dist = np.sqrt((gx - agent.x)**2 + (gy - agent.y)**2)
                
                if dist > 50.0:
                    start_node = (int(candidate_x), int(candidate_y))
                    end_node = (int(gx), int(gy))
                    path = a_star_search(self.heightmap, self.bs_radio_map, start_node, end_node, per_pixel_threshold, self.radio_bias)
                    if path:
                        agent.x, agent.y = candidate_x, candidate_y
                        agent.goal_x, agent.goal_y = gx, gy
                        agent.nav_path = path 
                        # print(i)
                        break
                    else:
                        print(f"could not find valid path for radio_bias={self.radio_bias}, try: {i}")
                    
        # initial radio cache update
        self.radio_model.generate_map_batch([(self.agent_map[a].x, self.agent_map[a].y) for a in self.agents], '5.8')

        # get connection list
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            agent.update_neighbors(list(self.agent_map.values()), self.width, self.height)
            agent.update_bs_connection(self.radio_model, self.width, self.height)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def step(self, actions):
        
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        self.total_energy_consumed_step = 0.0
        self.sim_time += 1
        
        # Reset viz and metrics for this step
        self.all_visible_links = [] 
        self.custom_links = {}     
        for agent in self.agent_map.values():
            agent.current_datarate = 0.0  

        # we first need to process the input actions
        # after these are processed (move, send packets),
        # then we update all the observation information
        # including radio maps, connected neighbors
        self._handle_communication_step(actions, rewards, infos)
        
        self._handle_movement_step(actions, rewards, infos)

        # after movement and comms actions have been done, update
        # comms info
        
        # global rm update to leverage batch processing
        agent_positions = [(a.x, a.y) for a in self.agent_map.values()]
        self.radio_model.generate_map_batch(agent_positions, '5.8')

        for agent_id, agent in self.agent_map.items():
            if agent.energy <= 0: continue
            
            # Update connectivity for the new positions
            agent.update_neighbors(list(self.agent_map.values()), self.width, self.height)
            agent.update_bs_connection(self.radio_model, self.width, self.height)

            # populate visible links for rendering
            if agent.bs_connected:
                self.all_visible_links.append((agent, self.base_station))
            
            for neighbor in agent.neighbors:
                if agent.ue_id < neighbor.ue_id:
                    self.all_visible_links.append((agent, neighbor))

        
        # metrics and cleanup
        global_truncate = self.sim_time >= self.EP_MAX_TIME

        active_datarates = [self.agent_map[a].current_datarate for a in self.agents if not terminations[a]]
        avg_datarate = np.mean(active_datarates) if active_datarates else 0.0
        self.history['datarate'].append(avg_datarate)
        self.history['energy'].append(self.total_energy_consumed_step)
        
        # calculate BS link ratio
        total_possible_links = len(self.agents)
        if total_possible_links > 0:
            bs_links = sum(1 for a_id in self.agents if self.base_station in self.connections[self.agent_map[a_id]])
            link_ratio = bs_links / total_possible_links
        else:
            link_ratio = 0.0
        self.history['bs_link_ratio'].append(link_ratio)
        
        # calculate RSS stats
        current_step_rss = []
        for aid in self.possible_agents:
            agent = self.agent_map[aid]
            if agent.energy > 0:
                rssi_from_bs = self.radio_model.get_signal_strength(*self.base_station.get_position(), agent.x, agent.y)
                self.agent_rss_history[aid].append(rssi_from_bs)
                current_step_rss.append(rssi_from_bs)
            
        if current_step_rss:
            self.history["avg_rss"].append(np.mean(current_step_rss))
        else:
            self.history["avg_rss"].append(-150.0)
            
        # final termination checks
        active_this_step = list(actions.keys())
        for agent_id in active_this_step:
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                terminations[agent_id] = True
            truncations[agent_id] = global_truncate
            infos[agent_id].update({"energy": agent.energy, "pos": (agent.x, agent.y)})

        infos[agent_id].update({
            "mission_completed": self.mission_done[agent_id],
            "goals_reached": self.agent_goals_completed[agent_id]
        })
        
        observations = {a: self._get_obs(a) for a in active_this_step}
        
        # Filter dead agents from the loop
        self.agents = [a for a in self.agents if not terminations.get(a, False) and not truncations.get(a, False)]
        
        return observations, rewards, terminations, truncations, infos



    def _handle_communication_step(self, actions, rewards, infos):
        
        active_agents = [self.agent_map[aid] for aid in actions.keys() if self.agent_map[aid].energy > 0]
        
        # if the bs is connected, get the network state info
        for agent in active_agents:
            if agent.bs_connected:
                # agent learns what bs has
                agent.network_state["BS_0"].update(self.base_station.packets_received)

                
        # If agents are neighbors, they also share their net state info
        for agent in active_agents:
            for neighbor in agent.neighbors:
                # we share what we know about the whole network
                agent.merge_network_state(neighbor.network_state)

        # clean up expired and delivered (to BS) packets
        for agent in active_agents:
            agent.cleanup_buffer()
            agent.drop_expired_packets()
            
        # comm step
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            if agent.energy <= 0: continue

            comm_flags = action[1:]
            targets_to_send = []

            # Determine targets using current neighbors
            for i, flag in enumerate(comm_flags[:-1]):
                if flag == 1:
                    target = self.agent_map[self.possible_agents[i]]
                    if target in agent.neighbors:
                        targets_to_send.append(target)
                        self.custom_links[(agent, target)] = 'green'

            # BS target
            if comm_flags[-1] == 1 and agent.bs_connected:
                targets_to_send.append(self.base_station)
                self.custom_links[(agent, self.base_station)] = 'cyan'

            # send packets
            if targets_to_send:
                pre_delivery = self.base_station.num_packets_received
                if self.packet_mode == 'boolean':
                    agent.send_packet(targets_to_send)
                else:
                    for target in targets_to_send:
                        agent.send_packets_to_target(target, self.STEP_LENGTH)
                delivered_now = self.base_station.num_packets_received - pre_delivery
                rewards[agent_id] += delivered_now * self.REWARD_PACKET_DELIVERY
                
                tx_cost = self.COST_TX_5G_PER_STEP * len(targets_to_send)
                agent.energy -= tx_cost
                self.total_energy_consumed_step += tx_cost
               
    
            # random packet generation
            if self.packet_mode == 'boolean':
                if np.random.rand() < self.PACKET_GEN_PROB and not self.mission_done.get(agent_id, False):
                    agent.generate_packet(size=10, time_to_live=50, destination="BS_0")
            else:
                # constant amount
                bits_needed = TELEMETRY_RATE_MBPS * 1e6 
                num_telemetry = int(bits_needed / PACKET_SIZE_BITS)
                
                for _ in range(num_telemetry):
                    agent.generate_packet(size=PACKET_SIZE_BITS, time_to_live=60, destination="BS_0")

                # science burst (simulating finding area of interest and generating large amounts of data)
                if np.random.rand() < BURST_PROBABILITY:
                    bits_burst = BURST_SIZE_MBITS * 1e6
                    num_burst = int(bits_burst / PACKET_SIZE_BITS)
                    
                    dtn_state = agent.payload_manager.get_state()
                    space_left = dtn_state['buffer_size'] - dtn_state['payload_size']
                    
                    packets_to_gen = min(num_burst, int(space_left / PACKET_SIZE_BITS))
                    
                    if packets_to_gen > 0:
                        for _ in range(packets_to_gen):
                            agent.generate_packet(size=PACKET_SIZE_BITS, time_to_live=300, destination="BS_0")
                            
        return actions, rewards, infos



    def _handle_movement_step(self, actions, rewards, infos):
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            
            
            if agent.nav_path and len(agent.nav_path) > 1:
                search_window = agent.nav_path[:15] 
                dists = [np.sqrt((n[0]-agent.x)**2 + (n[1]-agent.y)**2) for n in search_window]
                closest_idx = np.argmin(dists)
                if closest_idx > 0:
                    agent.nav_path = agent.nav_path[closest_idx:]

            prev_dist = np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2)
            
            if self.mission_done.get(agent_id, False):
                move_cmd = 0 
            else:
                move_cmd = action[0]
                
            
            dx, dy = 0, 0
            
            if move_cmd == 1: dy = 1        # North
            elif move_cmd == 2: dy = -1     # South
            elif move_cmd == 3: dx = -1     # West
            elif move_cmd == 4: dx = 1      # East
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
                    rewards[agent_id] += self.PENALTY_INVALID_MOVE
                    step_energy = self.COST_IDLE_PER_STEP 
                else:
                    agent.x, agent.y = new_x, new_y
                    agent.total_distance += self.MAX_DIST_PER_STEP
                    incline_cost = max(0, height_diff * 0.5) 
                    step_energy = self.COST_MOVE_PER_STEP + incline_cost
            else:
                step_energy = self.COST_IDLE_PER_STEP
            
            agent.energy -= step_energy
            self.total_energy_consumed_step += step_energy

            # rewards and arrival logic
            curr_dist = np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2)
            dist_delta = prev_dist - curr_dist 
            rewards[agent_id] += dist_delta * self.REWARD_DIST_SCALE
            
            safety_factor = 0.9
            per_pixel_threshold = (self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)) * safety_factor


            if curr_dist < (self.MAX_DIST_PER_STEP*1.5) and not self.mission_done[agent_id]: 
                rewards[agent_id] += self.REWARD_GOAL_ARRIVAL
                infos[agent_id]['task_complete'] = True
                self.agent_goals_completed[agent_id] += 1
                
                if self.agent_goals_completed[agent_id] >= self.num_goals:
                    self.mission_done[agent_id] = True
                    agent.nav_path = [] 
                    agent.goal_x, agent.goal_y = agent.x, agent.y 
                else:
                    max_retries = 100
                    rng = np.random.RandomState(self.seed + self.agent_goals_completed[agent_id]+int(agent_id[-1])*self.num_goals*2)
                    for i in range(max_retries):
                        gx = rng.uniform(0, self.width)
                        gy = rng.uniform(0, self.height)
                        dist = np.sqrt((gx - agent.x)**2 + (gy - agent.y)**2)
                        
                        if dist > 25.0:
                            start_node = (int(agent.x), int(agent.y))
                            end_node = (int(gx), int(gy))
                            path = a_star_search(self.heightmap, self.bs_radio_map, start_node, end_node, per_pixel_threshold, self.radio_bias)
                            if path:
                                agent.goal_x = gx
                                agent.goal_y = gy
                                agent.nav_path = path
                                # print(i)
                                break
                            else:
                                print(f"could not find valid path for radio_bias={self.radio_bias}, try: {i}")
            else:
                infos[agent_id]['task_complete'] = False
                
        return actions, rewards, infos
            
                
            
    def heuristic_move_action(self, agent_id):
        """
        picks movement action based on pure pursuit to follow the precomputed path
        """
        agent = self.agent_map[agent_id]
        
        if self.mission_done.get(agent_id, False):
            return 0
        
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
    
    

    def _get_obs(self, agent_id):
        agent = self.agent_map[agent_id]
        all_agents = list(self.agent_map.values())
        
        # get raw obs from agent
        obs_dict = agent.get_local_observation(all_agents=all_agents)

        # build action mask
        num_rovers = len(self.possible_agents)
        # 9 movement actions + (num_rovers peer targets + 1 BS target)
        move_mask = np.zeros(9, dtype=np.int8)
        move_mask[0] = 1 #idle is always allowed


        comm_masks = []
        #comm masks for peers
        for target_id in self.possible_agents:
            if target_id == agent_id:
                comm_masks.append([1, 1]) # can always send to self
                continue
            
            target_rover = self.agent_map[target_id]
            rssi = self.radio_model.get_signal_strength(agent.x, agent.y, target_rover.x, target_rover.y)
            # 1st bit is don't send, 2nd bit is send
            # rllib expects [valid_no_send, valid_send]
            mask = [1, 1] if rssi > self.MIN_DBM_THRESHOLD else [1, 0]
            comm_masks.append(mask)
        
        # base station mask
        rssi_bs = self.radio_model.get_signal_strength(agent.x, agent.y, self.base_station.x, self.base_station.y)
        comm_masks.append([1, 1] if rssi_bs > self.MIN_DBM_THRESHOLD else [1, 0])

        # flatten the mask for rllib
        action_mask = np.concatenate([move_mask, np.array(comm_masks).flatten()]).astype(np.int8)

        # final dictionary assemby
        final_obs = {
            "energy": np.array([obs_dict["energy"]], dtype=np.float32),
            "position": obs_dict["position"].astype(np.float32),
            "buffer_usage": obs_dict["buffer_usage"].astype(np.float32),
            "num_packets": obs_dict["num_packets"].astype(np.float32),
            "other_agent_vectors": obs_dict["other_agent_vectors"].astype(np.float32),
            "other_agent_connectivity": obs_dict["other_agent_connectivity"].astype(np.float32),
            "goal_vector": obs_dict["goal_vector"].astype(np.float32),
            "terrain": obs_dict["terrain"].astype(np.float32),
            "radio_map": obs_dict["radio_map"].astype(np.float32),
            "action_mask": action_mask
        }

        # Safety check for non-finite values
        for key, val in final_obs.items():
            if not np.all(np.isfinite(val)):
                final_obs[key] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        
        return final_obs
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num_rovers = len(self.possible_agents)
        # 9 movement + (num_rovers + bs) comm targets
        mask_dim = 9 + (num_rovers + 1) * 2
        
        f32_min = np.finfo(np.float32).min
        f32_max = np.finfo(np.float32).max

        return spaces.Dict({
            "energy": spaces.Box(low=0, high=self.START_ENERGY, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=0, high=256, shape=(2,), dtype=np.float32),
            "buffer_usage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "num_packets": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            "other_agent_vectors": spaces.Box(low=-256, high=256, shape=(num_rovers + 1, 2), dtype=np.float32),
            "other_agent_connectivity": spaces.Box(low=0, high=1, shape=(num_rovers + 1,), dtype=np.float32),
            "goal_vector": spaces.Box(low=-256, high=256, shape=(2,), dtype=np.float32),
            "terrain": spaces.Box(low=0, high=500, shape=(1, 256, 256), dtype=np.float32),
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







    """
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    |||||||||||||||||   RENDERING  METHODS   ||||||||||||||||||||||||||||||||||||
    |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """



    def render(self):
        if self.render_mode is None: return

        n_agents = len(self.possible_agents)
        # We keep the column ratios but add a row for the dashboard
        width_ratios = [4, 4] + ([4] * n_agents) + [3.5]
        num_cols = len(width_ratios)
        fx = 5.0 * num_cols
        fy = 15.0

        plt.close()
        fig = plt.figure(figsize=(fx, fy), dpi=65)

        fig.suptitle(f"Radio Bias: {self.radio_bias} | Seed: {self.seed}", 
                 fontsize=20, fontweight='bold', y=0.98)
        

        gs = fig.add_gridspec(
            ncols=num_cols, nrows=5,
            width_ratios=width_ratios, height_ratios=(2, 2, 2, 2, 1.5),
            hspace=0.6, wspace=0.6,
            top=0.95, bottom=0.05, left=0.025, right=0.955,
        )

        sim_ax = fig.add_subplot(gs[0:4, 0]) # Simulation spans top 4 rows
        bs_map_ax = fig.add_subplot(gs[0:4, 1]) # BS Map spans top 4 rows

        self.render_simulation(sim_ax)
        
        all_agent_paths = {
            aid: self.agent_map[aid].nav_path 
            for aid in self.possible_agents if aid in self.agent_map
        }

        self.render_radio_map(bs_map_ax, self.bs_radio_map, "BS Coverage", agent_paths=all_agent_paths)

        # Agent Radio Maps span top 4 rows
        for i, agent_id in enumerate(self.possible_agents):
            col_idx = i + 2 
            map_ax = fig.add_subplot(gs[0:4, col_idx])
            if agent_id in self.agent_map:
                agent = self.agent_map[agent_id]
                if agent.energy > 0:
                    radio_map = self.radio_model.generate_map((agent.x, agent.y), '5.8')
                    self.render_radio_map(map_ax, radio_map, f"Radio Map (Agent {chr(ord('A') + i)})")
                else:
                    map_ax.text(0.5, 0.5, "SIGNAL LOST", ha='center', va='center', color='red')
                    map_ax.axis('off')
            else:
                map_ax.axis('off')

        # Metrics remain in the last column, top 4 rows
        conn_ax = fig.add_subplot(gs[0, -1]) 
        rss_ax = fig.add_subplot(gs[1, -1])    
        energy_ax = fig.add_subplot(gs[2, -1]) 
        # (gs[3, -1] is currently empty, or you can add another metric here)

        # Dashboard now gets the ENTIRE bottom row
        dash_ax = fig.add_subplot(gs[4, :]) 

        self.render_dashboard(dash_ax)
        self.render_bs_connectivity(conn_ax) 
        self.render_avg_rss(rss_ax)             
        self.render_avg_energy(energy_ax)

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
            ax.plot([u.x, v.x], [u.y, v.y], color=color, 
                    linewidth=4, alpha=0.6, zorder=2,
                    path_effects=[pe.withStroke(linewidth=6, foreground='white')])
            
            # draw a small arrowhead or marker to show direction
            ax.annotate("", xy=(v.x, v.y), xytext=(u.x, u.y),
            arrowprops=dict(
                arrowstyle="->, head_width=0.8, head_length=1.0", 
                color=color, 
                lw=2
            ))

        ax.axis('off')
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])


    def render_dashboard(self, ax):
        ax.axis('off')
        
        bs_total = self.base_station.num_packets_received
        bs_unique = len(self.base_station.packets_received)
        
        packets_generated = sum(
            self.agent_map[aid].payload_manager.get_state().get('num_packets_generated', 0)
            for aid in self.possible_agents if aid in self.agent_map
        )
        ax.set_title(f"DTN Network State | Time: {self.sim_time:.0f} | Total BS Recv: {bs_unique} ({bs_total} raw), Packets Generated: {packets_generated}", 
                     fontweight='bold', fontsize=12, pad=20)

        rows = [f"{chr(ord('A') + i)}" for i in range(len(self.possible_agents))]
        cols = ["Buf%", "Stored", "Gen", "Goals", "Energy", "Dist (m)"]
        cell_text = []

        for agent_id in self.possible_agents:
            if agent_id in self.agent_map:
                agent = self.agent_map[agent_id]
                dtn = agent.payload_manager.get_state()
                buf_per = (dtn["payload_size"] / dtn["buffer_size"]) * 100
                goals_done = self.agent_goals_completed.get(agent_id, 0)
                cell_text.append([
                    f"{buf_per:.1f}%",
                    f"{dtn.get('num_packets', 0)}",
                    f"{dtn.get('num_packets_generated', 0)}",
                    f"{goals_done}/{self.num_goals}",
                    f"{agent.energy:.0f}",
                    f"{agent.total_distance:.1f}"
                ])
            else:
                cell_text.append(["N/A", "N/A", "N/A", "Dead", "N/A"])

        table = ax.table(
            cellText=cell_text, 
            rowLabels=rows, 
            colLabels=cols, 
            loc="center", 
            bbox=[0.15, 0.2, 0.8, 0.7] 
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        # summary
        bs_state = self.base_station.get_state()
        bs_text = (f"BASE STATION SUMMARY: {bs_state['num_packets_received']} Total Received | "
                   f"{bs_state['num_duplicates_received']} Duplicates")
     
        ax.text(0.5, -0.1, bs_text, transform=ax.transAxes, 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='cyan', alpha=0.2, edgecolor='black', boxstyle='round,pad=0.5'),
                clip_on=False)

        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:
                cell.get_text().set_weight('bold')
        
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

        
    def render_radio_map(self, ax, radio_map, title, agent_paths=None):
        ax.set_title(title)
        if radio_map is not None:
            im = ax.imshow(radio_map, cmap='viridis', origin='lower', 
                           extent=[0, self.width, 0, self.height], vmin=-130, vmax=0)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.figure.colorbar(im, cax=cax, orientation="vertical", label="Signal (dBm)")
        
        if agent_paths:
            for agent_id, path in agent_paths.items():
                if path and len(path) > 1:
                    color = 'blue' if agent_id == "rover_0" else ('purple' if agent_id == "rover_1" else 'green')
                    
                    px = [p[0] for p in path]
                    py = [p[1] for p in path]
                    ax.plot(px, py, color=color, linestyle='--', linewidth=1.2, alpha=0.7)

        for agent_id, agent in self.agent_map.items():
            if agent.energy <= 0: continue
            
            is_owner = (title.find(agent_id[-1]) != -1) or (agent_id == "rover_0" and title.find("A") != -1)
            
            color = 'blue' if agent_id == "rover_0" else ('purple' if agent_id == "rover_1" else 'green')
            size = 150 if is_owner else 80
            edge = 'white' if is_owner else 'black'
            
            ax.scatter(agent.x, agent.y, s=size, color=color, edgecolors=edge, linewidth=2, zorder=5)

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
        
    def render_avg_rss(self, ax):
        """Plots RSS of each agent and the global average over time."""
        colors = ['blue', 'purple', 'green'] 
        
        # plot rss per agent
        for i, aid in enumerate(self.possible_agents):
            if aid in self.agent_rss_history and len(self.agent_rss_history[aid]) > 0:
                ax.plot(self.agent_rss_history[aid], color=colors[i % len(colors)], 
                        alpha=0.3, linewidth=1)
        
        # plot global avg
        if len(self.history['avg_rss']) > 0:
            ax.plot(self.history['avg_rss'], color="red", linewidth=2, label="Avg")
            
        ax.set_ylabel("RSS to BS (dBm)")
        ax.set_xlim([0, self.EP_MAX_TIME])
        ax.set_ylim([-130, -30])
        ax.grid(True, alpha=0.3)
        
    def render_bs_connectivity(self, ax):
        """Tracks what % of agents have a valid link to BS over time."""
        if len(self.history['bs_link_ratio']) > 0:
            ax.plot(self.history['bs_link_ratio'], color="cyan", linewidth=2)
            # Calculate moving average for clarity
            if len(self.history['bs_link_ratio']) > 10:
                avg = np.convolve(self.history['bs_link_ratio'], np.ones(10)/10, mode='valid')
                ax.plot(range(9, len(self.history['bs_link_ratio'])), avg, color="blue", alpha=0.5)
                
        ax.set_ylabel("BS Connectivity (%)")
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([0, self.EP_MAX_TIME])
        ax.grid(True, alpha=0.3)

        
