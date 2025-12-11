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
        self.STEP_LENGTH = 25.0
        self.METERS_PER_PIXEL = 1.0
        self.MAX_DIST_PER_STEP = (self.ROVER_SPEED * self.STEP_LENGTH) / self.METERS_PER_PIXEL
        
        # Energy & Rewards
        self.START_ENERGY = 1000.0
        self.COST_MOVE_PER_STEP = 5.0
        self.COST_TX_5G_PER_STEP = 1.0
        self.COST_TX_415_PER_STEP = 2.0
        self.COST_IDLE_PER_STEP = 0.1
        self.EP_MAX_TIME = 300
        
        # Reward Config
        self.REWARD_PEER_LINK = 1.0
        self.REWARD_BS_LINK = 5.0  # Higher reward for connecting to sink
        self.PENALTY_FAIL = -0.1

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
        self.history = {"datarate": [], "energy": []}
        
        # randomly place bs
        new_bs_x, new_bs_y = np.random.uniform(0, self.width, size=(2,))
        self.base_station.x = new_bs_x
        self.base_station.y = new_bs_y
        
        # reset agent states
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            agent.energy = self.START_ENERGY
            agent.total_distance = 0.0
            agent.active_route = None
            agent.current_datarate = 0.0
            
            agent.x = np.random.uniform(0, self.width)
            agent.y = np.random.uniform(0, self.height)
            
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

        # movement logic
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            move_cmd = action[0] 
            
            agent.current_datarate = 0.0 
            
            dx, dy = 0, 0
            if move_cmd == 1: dy = 1 
            elif move_cmd == 2: dy = -1
            elif move_cmd == 3: dx = -1
            elif move_cmd == 4: dx = 1
            
            step_energy = 0.0
            
            if dx != 0 or dy != 0:
                dist = np.sqrt(dx**2 + dy**2)
                scale = self.MAX_DIST_PER_STEP / dist
                new_x = np.clip(agent.x + dx*scale, 0, self.width)
                new_y = np.clip(agent.y + dy*scale, 0, self.height)
                
                agent.x, agent.y = new_x, new_y
                agent.total_distance += self.MAX_DIST_PER_STEP
                step_energy = self.COST_MOVE_PER_STEP
            else:
                step_energy = self.COST_IDLE_PER_STEP
            
            agent.energy -= step_energy
            self.total_energy_consumed_step += step_energy

        self._update_radio_cache_batch()

        # comm logic
        self.connections = defaultdict(set) 
        self.custom_links = {} 
        
        num_rovers = len(self.agents)
        bs_target_idx = num_rovers + 1  # The index assigned to BS

        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            target_idx = action[1] 
            
            target_entity = None
            is_bs = False

            # check if target is a Rover
            if 0 < target_idx <= num_rovers:
                target_str = self.possible_agents[target_idx - 1]
                if target_str != agent_id:
                    target_entity = self.agent_map[target_str]
            
            # check if target is BS
            elif target_idx == bs_target_idx:
                target_entity = self.base_station 
                is_bs = True

            # link check
            if target_entity:
                radio_map = self._get_cached_radio_map(agent)
                rssi = self._get_signal_strength(radio_map, target_entity)
                
                # threshold check
                if rssi > -90.0: 
                    self.connections[agent].add(target_entity)
                    
                    # Reward differentiation between bs and peer links
                    if is_bs:
                        rewards[agent_id] += self.REWARD_BS_LINK
                        self.custom_links[(agent, target_entity)] = 'cyan' 
                        agent.current_datarate = 500.0 # higher rate at BS?
                    else:
                        rewards[agent_id] += self.REWARD_PEER_LINK
                        self.custom_links[(agent, target_entity)] = 'green' # Peer link color
                        agent.current_datarate = 100.0

                    # Energy cost
                    tx_cost = self.COST_TX_5G_PER_STEP
                    agent.energy -= tx_cost
                    self.total_energy_consumed_step += tx_cost
                    agent.active_route = ([(agent, target_entity)], '5.8')
                else:
                    rewards[agent_id] += self.PENALTY_FAIL 

        # metrics and clean up
        self.sim_time += 1
        global_truncate = self.sim_time >= self.EP_MAX_TIME

        avg_datarate = np.mean([self.agent_map[a].current_datarate for a in self.agents])
        self.history['datarate'].append(avg_datarate)
        self.history['energy'].append(self.total_energy_consumed_step)
        
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                terminations[agent_id] = True
            truncations[agent_id] = global_truncate
            infos[agent_id] = {"energy": agent.energy, "pos": (agent.x, agent.y)}

        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]
        observations = {a: self._get_obs(a) for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos


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
        obs_dict = agent.get_local_observation(self.heightmap, all_agents, bs_loc)
        
        if len(obs_dict['terrain'].shape) == 2:
            obs_dict['terrain'] = np.expand_dims(obs_dict['terrain'], axis=0)
        return obs_dict
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num_neighbors = len(self.possible_agents) - 1
        
        # mask dim: movement(5) + comm Targets (Num_Agents + BS(1) + None(1))
        mask_dim = 5 + (len(self.possible_agents) + 1 + 1)
        
        return spaces.Dict({
            "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "terrain": spaces.Box(low=0, high=1, shape=(1, int(self.height), int(self.width)), dtype=np.float32),
            "neighbors": spaces.Box(low=-np.inf, high=np.inf, shape=(num_neighbors, 2), dtype=np.float32),
            "base_station": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "radio_map": spaces.Box(low=-200, high=0, shape=(1, int(self.height), int(self.width)), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(mask_dim,), dtype=np.int8)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # [Movement (0-4), Communication Target (0 = None, 1..N = Rovers, N+1 = BS)]
        # Total comm options: 1 + N + 1 = N + 2
        return spaces.MultiDiscrete([5, len(self.possible_agents) + 2])

    # render logic
    def render(self):
        if self.render_mode is None:
            return

        # determine layout based on agent number
        n_agents = len(self.possible_agents)
        
        # Grid Ratios and figure size
        width_ratios = [4] + ([4] * n_agents) + [3.5]
        num_cols = len(width_ratios)

        base_width_per_col = 5.0 
        fx = base_width_per_col * num_cols
        fy = max(1.25 * self.height / 100.0 * 4.0, 10.0) 
        
        plt.close()
        fig = plt.figure(figsize=(fx, fy), dpi=50)

        gs = fig.add_gridspec(
            ncols=num_cols, nrows=3,
            width_ratios=width_ratios,
            height_ratios=(3, 3, 3),
            hspace=0.45, wspace=0.4,
            top=0.95, bottom=0.15,
            left=0.025, right=0.955,
        )

        sim_ax = fig.add_subplot(gs[:, 0])
        
        dash_ax = fig.add_subplot(gs[0, -1]) 
        qoe_ax = fig.add_subplot(gs[1, -1])  
        conn_ax = fig.add_subplot(gs[2, -1]) 

        # render sim 
        self.render_simulation(sim_ax)

        # render radio maps
        for i, agent_id in enumerate(self.possible_agents):
            col_idx = i + 1
            map_ax = fig.add_subplot(gs[:, col_idx])
            
            # only render if agent is active
            if agent_id in self.agent_map:
                agent = self.agent_map[agent_id]
                
                if agent.energy > 0:
                    radio_map = self._get_cached_radio_map(agent)
                    
                    # generate labels
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
        
        #Cap framerate
        self.clock.tick(self.metadata["render_fps"])
        
        # We process events to keep the window responsive, but we 
        # generally let the Runner script handle the QUIT logic 
        # to avoid closing the environment mid-step.
        pygame.event.pump()

    def render_simulation(self, ax):
        # terrain
        if self.heightmap is not None:
            ax.imshow(self.heightmap, cmap='terrain', extent=[0, self.width, 0, self.height], origin='lower', zorder=0)

        # draw bs
        
        ax.scatter(self.base_station.x, self.base_station.y, s=300, c='cyan', marker='^', edgecolors='black', zorder=3, label="Base Station")
        ax.annotate("BS", xy=(self.base_station.x, self.base_station.y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', color='cyan', fontweight='bold')

        # agents
        for agent_id, agent in self.agent_map.items():
            if agent.energy <= 0: continue
            
            color = 'gray'
            if agent_id == "rover_0": color = 'blue'
            elif agent_id == "rover_1": color = 'purple'
            elif agent_id == "rover_2": color = 'green'

            ax.scatter(agent.x, agent.y, s=200, zorder=2, color=color, marker="o")
            
            label_map = {"rover_0": "A", "rover_1": "B", "rover_2": "C"}
            label = label_map.get(agent_id, agent_id[-1])
            ax.annotate(label, xy=(agent.x, agent.y), ha="center", va="center", color='white', fontweight='bold')

        # draw links
        for (u, v), color in self.custom_links.items():
            # check if v is an agent or the BS object
            vx, vy = v.x, v.y 
            
            ax.plot([u.x, vx], [u.y, vy], color=color, 
                    path_effects=[pe.SimpleLineShadow(shadow_color="black"), pe.Normal()],
                    linewidth=3, zorder=1)

        ax.axis('off')
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        
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
        ax.set_ylim([0, 110])

    def render_avg_energy(self, ax):
        if len(self.history['energy']) > 0:
            ax.plot(self.history['energy'], color="black")
        ax.set_ylabel("Step Energy")
        ax.set_xlabel("Step")
        ax.set_xlim([0, self.EP_MAX_TIME])
        ax.set_ylim([0, 100])