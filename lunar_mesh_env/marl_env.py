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
from collections import defaultdict, Counter

from .marl_entities import MarlMeshAgent as MeshAgent
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
        
        # TODO: realistic energy
        self.START_ENERGY = 1000.0
        self.COST_MOVE_PER_STEP = 5.0
        self.COST_TX_5G_PER_STEP = 1.0
        self.COST_TX_415_PER_STEP = 2.0
        self.COST_IDLE_PER_STEP = 0.1
        self.EP_MAX_TIME = 100

        # agent set up
        self.possible_agents = [f"rover_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        self.agent_map = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_map[agent_id] = MeshAgent(
                ue_id=i+1, 
                velocity=10, 
                snr_tr=20, 
                noise=10, 
                height=1.5, 
                frequency=5.8, 
                tx_power=23, 
                bw=20
            )

        self.action_spaces = {
            # [Movement (0-4), Communication Target (0-N)]
            agent: spaces.MultiDiscrete([5, num_agents + 1]) 
            for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Dict({
                "self_state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "terrain": spaces.Box(low=0, high=1, shape=(1, int(self.height), int(self.width)), dtype=np.float32),
                "neighbors": spaces.Box(low=-np.inf, high=np.inf, shape=(num_agents-1, 2), dtype=np.float32)
            })
            for agent in self.agents
        }


        self.heightmap = None
        try:
            self.heightmap = np.load(self.hm_path)
        except Exception as e:
            print(f"Warning: Could not load heightmap: {e}")
            self.heightmap = np.zeros((int(self.height), int(self.width)))

        # cache for nn rm inference
        self.radio_cache = {} 
        
        self.connections = defaultdict(set)
        self.custom_links = {} 
        self.sim_time = 0
        
        # metric history for viz
        self.history = {
            "datarate": [],
            "energy": []
        }
        self.total_energy_consumed_step = 0.0

        # pygame setup
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.sim_time = 0
        self.connections = defaultdict(set)
        self.custom_links = {}
        self.history = {"datarate": [], "energy": []}
        
        # reset
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            agent.energy = self.START_ENERGY
            agent.total_distance = 0.0
            agent.active_route = None
            agent.current_datarate = 0.0
            
            # start pos
            agent.x = np.random.uniform(0, self.width)
            agent.y = np.random.uniform(0, self.height)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def step(self, actions):
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        self.total_energy_consumed_step = 0.0

        # movement phyics
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            move_cmd = action[0] # 0 is movement
            
            agent.current_datarate = 0.0 
            
            dx, dy = 0, 0
            if move_cmd == 1: dy = 1 
            elif move_cmd == 2: dy = -1
            elif move_cmd == 3: dx = -1
            elif move_cmd == 4: dx = 1
            
            step_energy = 0.0
            
            # scale by speed
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

        # communication 
        self.connections = defaultdict(set) 
        self.custom_links = {} 
        
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            target_idx = action[1] # 1 is comm target
            
            if target_idx > 0 and target_idx <= len(self.agents):
                target_str = self.possible_agents[target_idx - 1]
                
                if target_str != agent_id:
                    target_agent = self.agent_map[target_str]
                    
                    # get map and chack signal
                    radio_map = self._get_radio_map(agent)
                
                    rssi = self._get_signal_strength(radio_map, target_agent)
                    
                    # establish link
                    if rssi > -90.0: 
                        self.connections[agent].add(target_agent)
                        rewards[agent_id] += 1.0 
                        
                        # energy cost
                        tx_cost = self.COST_TX_5G_PER_STEP
                        agent.energy -= tx_cost
                        self.total_energy_consumed_step += tx_cost
                        
                        # viz stuff
                        agent.current_datarate = 100.0 # Placeholder value
                        agent.active_route = ([(agent, target_agent)], '5.8')
                        self.custom_links[(agent, target_agent)] = 'green'
                    else:
                        rewards[agent_id] -= 0.1 

        # metric stuff
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
            
            infos[agent_id] = {
                "energy": agent.energy,
                "pos": (agent.x, agent.y)
            }

        self.agents = [a for a in self.agents if not terminations[a] and not truncations[a]]
        observations = {a: self._get_obs(a) for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos

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
        if radio_map is None: return -150.0
        r_idx = int((target_agent.y / self.height) * (radio_map.shape[0]-1))
        c_idx = int((target_agent.x / self.width) * (radio_map.shape[1]-1))
        r_idx = np.clip(r_idx, 0, radio_map.shape[0]-1)
        c_idx = np.clip(c_idx, 0, radio_map.shape[1]-1)
        return radio_map[r_idx, c_idx]

    def _get_obs(self, agent_id):
        agent = self.agent_map[agent_id]
        all_agents = list(self.agent_map.values())
        obs_dict = agent.get_local_observation(self.heightmap, all_agents)
        
        if len(obs_dict['terrain'].shape) == 2:
            obs_dict['terrain'] = np.expand_dims(obs_dict['terrain'], axis=0)
        return obs_dict

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # render logic
    def render(self):
        if self.render_mode is None:
            return

        # setup fig
        fig = plt.figure()
        fx = max(3.0 / 2.0 * 1.25 * (self.width * 2) / fig.dpi, 24.0)
        fy = max(1.25 * self.height / fig.dpi, 10.0)
        plt.close()
        fig = plt.figure(figsize=(fx, fy))

        gs = fig.add_gridspec(
            ncols=4, nrows=3,
            width_ratios=(4, 4, 4, 3.5),
            height_ratios=(3, 3, 3),
            hspace=0.45, wspace=0.4,
            top=0.95, bottom=0.15,
            left=0.025, right=0.955,
        )

        sim_ax = fig.add_subplot(gs[:, 0])
        rm_a_ax = fig.add_subplot(gs[:, 1])  
        rm_b_ax = fig.add_subplot(gs[:, 2])  
        dash_ax = fig.add_subplot(gs[0, 3]) 
        qoe_ax = fig.add_subplot(gs[1, 3])  
        conn_ax = fig.add_subplot(gs[2, 3]) 

        # render sim
        self.render_simulation(sim_ax)

        # radio maps
        if "rover_0" in self.agent_map:
            agent_a = self.agent_map["rover_0"]
            map_a = self._get_radio_map(agent_a)
            self.render_radio_map(rm_a_ax, map_a, "Radio Map (Agent A)")
        
        if "rover_1" in self.agent_map:
            agent_b = self.agent_map["rover_1"]
            map_b = self._get_radio_map(agent_b)
            self.render_radio_map(rm_b_ax, map_b, "Radio Map (Agent B)")

        #  dashboard
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

    def _render_human(self, canvas, fig):
        data = canvas.buffer_rgba()
        size = canvas.get_width_height()

        if self.window is None:
            pygame.init()
            self.clock = pygame.time.Clock()
            window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
            self.window = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Lunar Mesh Env (MARL)")

        self.window.fill("white")
        screen = pygame.display.get_surface()
        plot = pygame.image.frombuffer(data, size, "RGBA")
        screen.blit(plot, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def render_simulation(self, ax):
        # terrain
        if self.heightmap is not None:
            ax.imshow(self.heightmap, cmap='terrain', extent=[0, self.width, 0, self.height], origin='lower', zorder=0)

        # agents
        colormap = cm.get_cmap("RdYlGn")
        
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

        # Links
        for (u, v), color in self.custom_links.items():
            ax.plot([u.x, v.x], [u.y, v.y], color=color, 
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

    def render_dashboard(self, ax):
        ax.axis('off')
        ax.set_title(f"Sim Time: {self.sim_time * self.STEP_LENGTH:.1f} s", fontweight='bold')

        rows = ["Agent A", "Agent B", "Agent C"]
        cols = ["Rate (Mbps)", "Energy", "Dist (m)"]
        cell_text = []

        target_agents = ["rover_0", "rover_1", "rover_2"]
        for a_id in target_agents:
            if a_id in self.agent_map:
                agent = self.agent_map[a_id]
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