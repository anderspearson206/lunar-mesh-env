# env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import string
from matplotlib.colors import Normalize
import gymnasium
from gymnasium import spaces
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pygame import Surface
from mpl_toolkits.axes_grid1 import make_axes_locatable 


from mobile_env.core.base import MComCore

from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.logging import Monitor
from mobile_env.core import metrics
from mobile_env.core.util import deep_dict_merge, BS_SYMBOL
from mobile_env.handlers.handler import Handler

from .entities import MeshAgent
from .handlers import MeshCentralHandler
from .radio_model import RadioMapModel



class CustomEnv(MComCore):

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "EP_MAX_TIME": 100,
            "seed": 1234,
            'reset_rng_episode': True,
            "handler": MeshCentralHandler,
        })

        config["ue"].update({
            "velocity": 10,
        })
        # Remove scheduler as it's for BS not mesh network
        if "scheduler" in config:
            del config["scheduler"]
        if "scheduler_params" in config:
            del config["scheduler_params"]
            
        return config

    # configure users and cells in the constructor
    def __init__(self, config={}, render_mode=None,
                 hm_path='../radio_data_2/radio_data_2/hm/hm_18.npy',
                 radio_model: RadioMapModel = None, 
                 agent_a_pos: Tuple[float, float] = None, 
                 agent_b_pos: Tuple[float, float] = None  
                ):
       
        super(MComCore, self).__init__()

        self.render_mode = render_mode
        assert render_mode in self.metadata["render_modes"] + [None]

        env_config = self.default_config()
        env_config.update(config)

        config = env_config 
        config = self.seeding(config)

        self.width, self.height = config["width"], config["height"]
        self.seed = config["seed"]
        self.reset_rng_episode = config["reset_rng_episode"]

        self.arrival = config["arrival"](**config["arrival_params"])
        self.channel = config["channel"](**config["channel_params"])

        self.movement = config["movement"](**config["movement_params"])
        self.utility = config["utility"](**config["utility_params"])

        self.EP_MAX_TIME = config["EP_MAX_TIME"]
        self.time = None
        self.closed = False
        
        self.heightmap_path = hm_path
        
        self.radio_model = radio_model
        self.agent_a_pos = agent_a_pos
        self.agent_b_pos = agent_b_pos

        self.heightmap = None
        try:
            self.heightmap = np.load(self.heightmap_path)
        except Exception as e:
            print(f"Warning: Could not load heightmap from {self.heightmap_path}: {e}")
        
        
        # MOVEMENT PARAMS
        #  viper rover is .2 when not using tools, .1 when using tools
        self.ROVER_SPEED = 0.2 # m/s
        self.STEP_LENGTH = 25.0
        self.METERS_PER_PIXEL = 1.0

        self.MAX_DIST_PER_STEP = (self.ROVER_SPEED * self.STEP_LENGTH)/self.METERS_PER_PIXEL
        
        self.sim_time_seconds= 0.0

        # ENERGY CONSTRAINTS:
        # TODO implement real values
        self.START_ENERGY = 1000.0
        self.COST_MOVE_PER_STEP = 5.0
        self.COST_TX_5G_PER_STEP = 1.0
        self.COST_TX_415_PER_STEP = 2.0
        self.COST_IDLE_PER_STEP = 0.1


        tx_params = {
            "frequency": config["bs"]["freq"],
            "tx_power": config["bs"]["tx"],
            "bw": config["bs"]["bw"],
        }
        

        # stationary_agent_config = config["ue"].copy()
        # stationary_agent_config['velocity'] = 0

        # # just 3 agents for now
        # agents = [
        #     # Agent "A": ue_id=1, stationary
        #     MeshAgent(ue_id=1, **stationary_agent_config, **tx_params),
        #     # Agent "B": ue_id=2, stationary
        #     MeshAgent(ue_id=2, **stationary_agent_config, **tx_params),
        #     # Agent "C": ue_id=3, mobile
        #     MeshAgent(ue_id=3, **config["ue"], **tx_params),
        # ]
        
        
        # setattr(agents[0], 'is_stationary', True) # Agent A
        # setattr(agents[1], 'is_stationary', True) # Agent B
        # setattr(agents[2], 'is_stationary', False) # Agent C
        
        # just 3 agents for now, all mobile
        agents = [
            # Agent "A": ue_id=1, mobile
            MeshAgent(ue_id=1, **config["ue"], **tx_params),
            # Agent "B": ue_id=2, mobile
            MeshAgent(ue_id=2, **config["ue"], **tx_params),
            # Agent "C": ue_id=3, mobile
            MeshAgent(ue_id=3, **config["ue"], **tx_params),
        ]

        

        self.agents = {agent.ue_id: agent for agent in agents}
        self.NUM_AGENTS = len(self.agents)

       
        self.feature_sizes = {
            "connections": self.NUM_AGENTS,
            "snrs": self.NUM_AGENTS,
            "utility": 1,
            "peer_utilities": self.NUM_AGENTS,
            "peer_connections": self.NUM_AGENTS,
        }

        # handler and action/obs space
        self.handler: Handler = config["handler"]
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

        # core data structures
        self.active: List[UserEquipment] = None 
        self.connections: Dict[UserEquipment, Set[UserEquipment]] = None
        self.datarates: Dict[Tuple[UserEquipment, UserEquipment], float] = None
        self.utilities: Dict[UserEquipment, float] = None
        self.rng = None
        
        self.custom_links: Dict[Tuple[UserEquipment, UserEquipment], str] = {}

        # pygame stuff
        self.window = None
        self.clock = None
        self.conn_isolines = None
        self.mb_isolines = None

        # metric monitering
        try:

            config["metrics"]["scalar_metrics"].update(
                {
                    "number connections": metrics.number_connections,
                    "number connected": metrics.number_connected,
                    "mean utility": metrics.mean_utility,
                    "mean datarate": metrics.mean_datarate,
                    "datarate": lambda env: env.agents[1].current_datarate if 1 in env.agents else 0,
                    "total_energy_consumed_step": lambda env: env.total_energy_consumed_step
                }
            )
            self.monitor = Monitor(**config["metrics"])
        except KeyError:
            print("Warning: 'metrics' not found in config. Skipping monitor setup.")
            self.monitor = Monitor(scalar_metrics={}, ue_metrics={}, bs_metrics={}) 
            
        self.total_energy_consumed_step = 0.0 
        

    def reset(self, *, seed=None, options=None):
        """
        Overrides the parent reset() method to use `self.agents`
        and skip `self.scheduler.reset()`.
        """
        super(MComCore, self).reset(seed=seed)

 
        self.time = 0.0 # step counter
        self.sim_time_seconds = 0.0 #continous physics time

        if seed is not None:
            self.seeding({"seed": seed})
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        if options is not None and options != {}:
            raise NotImplementedError(
                "Passing extra options on env.reset() is not supported."
            )

        self.arrival.reset()
        self.channel.reset()
        # this was used in original ,mobile-env but crashes here
        # self.scheduler.reset() 
        self.movement.reset()
        self.utility.reset()

        for agent in self.agents.values(): 
            agent.stime = self.arrival.arrival(agent)
            agent.extime = self.arrival.departure(agent)
            
   
            agent.energy = self.START_ENERGY
            agent.active_route = None
            agent.current_datarate = 0.0
            agent.is_moving = False


        # generate initial positions, (A and B stationary for now)
        for agent in self.agents.values(): 
  
            if agent.ue_id == 1 and self.agent_a_pos is not None:
                agent.x, agent.y = self.agent_a_pos
  
            elif agent.ue_id == 2 and self.agent_b_pos is not None:
                agent.x, agent.y = self.agent_b_pos
            else:
       
                agent.x, agent.y = self.movement.initial_position(agent)
      
        self.active = [agent for agent in self.agents.values() if agent.stime <= 0] 
        self.active = sorted(self.active, key=lambda agent: agent.ue_id)

        self.connections = defaultdict(set) 
        self.datarates = defaultdict(float)
        self.utilities = {}

        self.custom_links.clear()

        self.max_departure = max(agent.extime for agent in self.agents.values()) 

        self.monitor.reset()

        self.handler.check(self)

        info = self.handler.info(self)
        info = {**info, **self.monitor.info()}

        return self.handler.observation(self), info
        

    def update_custom_links(self):
        """
        Asks Agent A to determine the routing path.
        The environment then stores this path for visualization.
        """
        self.custom_links.clear()

        if 1 not in self.agents or 3 not in self.agents:
            return 
            
        agent_a = self.agents[1]
        agent_c = self.agents[3]
        
     
        if agent_c not in self.active:

            agent_a.active_route = None
            agent_a.current_datarate = 0.0
            return


        route_links = agent_a.policy_decide_route(
            agent_c, 
            list(self.agents.values()),
            self.radio_model, 
            self.width, 
            self.height
        )
        

        self.custom_links = route_links
    
    # Type hint is UserEquipment for polymorphism, but objects will be MeshAgent
    def check_connectivity(self, agent1: UserEquipment, agent2: UserEquipment) -> bool:
        """
        Overrides parent. Connection is viable if pathloss is above threshold
        for *both* directions.
        """
        if agent1 == agent2:
            return False
            

        snr1 = self.channel.snr(agent2, agent1)
        if snr1 <= agent1.snr_threshold:
            return False
            

        snr2 = self.channel.snr(agent1, agent2)
        if snr2 <= agent2.snr_threshold:
            return False
            
        return True

    def update_connections(self) -> None:
        """
        Overrides parent. Release P2P connections where agents moved out-of-range.
        This implementation assumes an undirected graph (links are symmetric).
        """
        disconnected_links = set()
        for agent1, peers in self.connections.items():
            for agent2 in peers:

                if agent1.ue_id < agent2.ue_id:
                    if not self.check_connectivity(agent1, agent2):
                        disconnected_links.add((agent1, agent2))
        

        for agent1, agent2 in disconnected_links:
            if agent2 in self.connections[agent1]:
                self.connections[agent1].remove(agent2)
            if agent1 in self.connections[agent2]:
                self.connections[agent2].remove(agent1)

    # type hint is UserEquipment for polymorphism
    def apply_action(self, action: int, agent: UserEquipment) -> None:
        """
        Overrides parent. Connect or disconnect `agent` to/from `peer`.
        Assumes an undirected graph (links are symmetric).
        """
        # do not apply update to connections if NOOP_ACTION is selected
        if action == self.NOOP_ACTION or agent not in self.active:
            return
            
        peer_id = action
        
        # check for invalid actions
        if peer_id not in self.agents or peer_id == agent.ue_id:
            return
            
        peer = self.agents[peer_id]

        # disconnect if already connected
        if peer in self.connections[agent]:
            self.connections[agent].remove(peer)
            if agent in self.connections[peer]:
                self.connections[peer].remove(agent)

        # connnect
        elif self.check_connectivity(agent, peer):
            self.connections[agent].add(peer)
            self.connections[peer].add(agent)
            
    def calculate_mesh_datarates(self):
        """
        Placeholder for mesh datarate calculation.
        """
        self.datarates.clear()
        for agent1, peers in self.connections.items():
            for agent2 in peers:
                # TODO: Implement real SINR/datarate calculation
                if (agent2, agent1) not in self.datarates:
                    
                    snr = self.channel.snr(agent1, agent2)
                    dr = self.channel.datarate(agent1, agent2, snr)
                    self.datarates[(agent1, agent2)] = dr 
                    
                    snr = self.channel.snr(agent2, agent1)
                    dr = self.channel.datarate(agent2, agent1, snr)
                    self.datarates[(agent2, agent1)] = dr 

    def macro_datarates(self, datarates):
        """
        Overrides parent. Compute aggregated agent data rates given all its P2P links.
        We will sum the datarate for all links where the agent is a *receiver*.
        """
        agent_datarates = Counter()
        for (sender, receiver), datarate in self.datarates.items():
            agent_datarates.update({receiver: datarate})
        return agent_datarates

    def step(self, actions: Dict[int, int]):
        """
        Overrides the parent `step` method to implement mesh networking logic.
        Now enforces physical movement limits based on ROVER_SPEED and STEP_LENGTH.
        """
        assert not self.time_is_up, "step() called on terminated episode"

        actions = self.handler.action(self, actions)

        self.update_connections()

        # apply actions 
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                self.apply_action(action, self.agents[agent_id])

        # update datarates 
        self.calculate_mesh_datarates()

        self.macro = self.macro_datarates(self.datarates)

        self.utilities = {
            agent: self.utility.utility(self.macro[agent]) for agent in self.active
        }

        self.utilities = {
            agent: self.utility.scale(util) for agent, util in self.utilities.items()
        }

        rewards = self.handler.reward(self)

        self.monitor.update(self)

        #  movement logic
        for agent in self.active:
            agent.is_moving = False 
            
            old_x, old_y = agent.x, agent.y
            proposed_x, proposed_y = self.movement.move(agent)
            
            dx = proposed_x - old_x
            dy = proposed_y - old_y
            dist = np.sqrt(dx**2 + dy**2)
            
            actual_dist_moved = 0.0

            # enforce speed limit and calc distance
            if dist > self.MAX_DIST_PER_STEP:
                scale_factor = self.MAX_DIST_PER_STEP / dist
                new_x = old_x + (dx * scale_factor)
                new_y = old_y + (dy * scale_factor)
                actual_dist_moved = self.MAX_DIST_PER_STEP 
                agent.is_moving = True 
            elif dist > 0:
                new_x = proposed_x
                new_y = proposed_y
                actual_dist_moved = dist
                agent.is_moving = True
            else:
                new_x = old_x
                new_y = old_y
                actual_dist_moved = 0.0
                agent.is_moving = False

            agent.x, agent.y = new_x, new_y
            agent.total_distance += actual_dist_moved

        self.update_custom_links()
        
        # energy deduction logic
        self.total_energy_consumed_step = 0.0
        
        for agent in self.agents.values():
            if agent not in self.active:
                continue 

            agent_energy_cost = 0.0
            is_transmitting = False

            # check if moving
            if agent.is_moving:
                agent_energy_cost += self.COST_MOVE_PER_STEP
            
            # check if transmitting (as source or relay)
            if agent.ue_id == 1 and agent.active_route:
                freq = agent.active_route[1]
                if freq == '5.8':
                    agent_energy_cost += self.COST_TX_5G_PER_STEP
                elif freq == '415':
                    agent_energy_cost += self.COST_TX_415_PER_STEP
                is_transmitting = True
            
            elif agent.ue_id == 2:
                is_relay = any(sender == agent for sender, _ in self.custom_links.keys())
                if is_relay and 1 in self.agents and self.agents[1].active_route:
                    freq = self.agents[1].active_route[1]
                    if freq == '5.8':
                        agent_energy_cost += self.COST_TX_5G_PER_STEP
                    elif freq == '415':
                        agent_energy_cost += self.COST_TX_415_PER_STEP
                    is_transmitting = True

            if not agent.is_moving and not is_transmitting:
                agent_energy_cost += self.COST_IDLE_PER_STEP
            
            # deduct energy
            agent.energy -= agent_energy_cost
            self.total_energy_consumed_step += agent_energy_cost


        # terminate existing connections for exiting agents
        leaving = set([agent for agent in self.active if agent.extime <= self.time])
        new_connections = defaultdict(set)
        for agent, peers in self.connections.items():
            if agent not in leaving:
                new_connections[agent] = peers - leaving
        self.connections = new_connections

        self.active = sorted(
            [
                agent
                for agent in self.agents.values()
                if agent.extime > self.time and agent.stime <= self.time
            ],
            key=lambda agent: agent.ue_id,
        )
        
        self.calculate_mesh_datarates()

        self.time += 1
        self.sim_time_seconds += self.STEP_LENGTH 

        # check whether episode is done & close the environment
        if self.time_is_up and self.window:
            self.close()

        # do not invoke next step on policies before at least one UE is active
        if not self.active and not self.time_is_up:
            # still dummy
            observation = self.handler.observation(self)
            info = self.handler.info(self)
            info = {**info, **self.monitor.info()}
            
            info['sim_time_s'] = self.sim_time_seconds
            
            terminated = False
            truncated = self.time_is_up
            rewards = self.handler.reward(self)
            return observation, rewards, terminated, truncated, info

        # compute observations for next step and information
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        info = {**info, **self.monitor.info()}
        
        # metric tracking
        if 1 in self.agents:
            agent_a = self.agents[1]
            info['datarate'] = agent_a.current_datarate
            info['is_connected'] = (agent_a.active_route is not None)
            if agent_a.active_route:
                info['route_freq'] = 1.0 if agent_a.active_route[1] == '5.8' else 0.0 # 1.0 for 5.8, 0.0 for 415
            else:
                info['route_freq'] = np.nan # No route
            info['agent_1_energy'] = agent_a.energy
            info['agent_1_dist'] = self.agents[1].total_distance
            
        if 2 in self.agents:
            info['agent_2_energy'] = self.agents[2].energy
            info['agent_2_dist'] = self.agents[2].total_distance
        if 3 in self.agents:
            info['agent_3_energy'] = self.agents[3].energy
            info['agent_3_dist'] = self.agents[3].total_distance
            
        info['total_energy_consumed_step'] = self.total_energy_consumed_step

        # pass back the actual physics time
        info['sim_time_s'] = self.sim_time_seconds

        terminated = False
        truncated = self.time_is_up

        return observation, rewards, terminated, truncated, info


    def render(self) -> None:
        """
        Overrides parent `render`.
        Removes all logic related to `bs_isolines` which used `self.stations`.
        Creates a new 4-column layout to show both Radio Maps.
        """
        mode = self.render_mode

        if self.closed:
            return

        # setup 
        fig = plt.figure()
        fx = max(3.0 / 2.0 * 1.25 * (self.width * 2) / fig.dpi, 24.0) 
        fy = max(1.25 * self.height / fig.dpi, 10.0) 
        plt.close()
        fig = plt.figure(figsize=(fx, fy))
    
        gs = fig.add_gridspec(
            ncols=4, 
            nrows=3,
            width_ratios=(4, 4, 4, 3.5), 
            height_ratios=(3, 3, 3),
            hspace=0.45,
            wspace=0.4, 
            top=0.95,
            bottom=0.15,
            left=0.025,
            right=0.955,
        )

        sim_ax = fig.add_subplot(gs[:, 0])
        rm_a_ax = fig.add_subplot(gs[:, 1])  
        rm_b_ax = fig.add_subplot(gs[:, 2])  
        dash_ax = fig.add_subplot(gs[0, 3]) 
        qoe_ax = fig.add_subplot(gs[1, 3])  
        conn_ax = fig.add_subplot(gs[2, 3]) 
        

        # render simulation, metrics and score if step() was called
        if self.time > 0:
            self.render_simulation(sim_ax)
            
            # get correct freq
            active_freq = '5.8' # 
            title_suffix = "(5.8 GHz)"
            if 1 in self.agents and self.agents[1].active_route:
                active_freq = self.agents[1].active_route[1] 
                if active_freq == '415':
                    title_suffix = "(415 MHz)"
            
            map_a = self.radio_model.generate_map((self.agents[1].x, self.agents[1].y), active_freq)
            map_b = self.radio_model.generate_map((self.agents[2].x, self.agents[2].y), active_freq)
    
            self.render_radio_map(rm_a_ax, map_a, f"Radio Map (Agent A) {title_suffix}")
            self.render_radio_map(rm_b_ax, map_b, f"Radio Map (Agent B) {title_suffix}")
            
            self.render_dashboard(dash_ax)

            self.render_avg_datarate(qoe_ax)
            self.render_avg_energy(conn_ax)

        fig.align_ylabels((qoe_ax, conn_ax))
        canvas = FigureCanvas(fig)
        canvas.draw()

        plt.close()

        if mode == "rgb_array":
            # render RGB image video
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            return data.reshape(canvas.get_width_height()[::-1] + (3,))

        
        # pygame, included in mobile-env but have not worked
        # on this much 
        elif mode == "human":
            data = canvas.buffer_rgba()
            size = canvas.get_width_height()

            if self.window is None:
                pygame.init()
                self.clock = pygame.time.Clock()

                window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
                self.window = pygame.display.set_mode(window_size)

                pygame.display.set_icon(Surface((0, 0)))

                pygame.display.set_caption("MComEnv")

            self.window.fill("white")

            screen = pygame.display.get_surface()
            plot = pygame.image.frombuffer(data, size, "RGBA")
            screen.blit(plot, (0, 0))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        else:
            raise ValueError("Invalid rendering mode.")

    def render_simulation(self, ax) -> None:
        """
        Overrides the MComCore.render_simulation method to first
        draw a background image on the simulation axis (ax)
        and then draw the new mesh network topology.
        """
        # draw heightmap
        if self.heightmap is not None:
            ax.imshow(self.heightmap, 
                        cmap='terrain', 
                        extent=[0, self.width, 0, self.height], 
                        origin='lower',
                        zorder=0)

        # draw agents
        colormap = cm.get_cmap("RdYlGn")
        unorm = plt.Normalize(self.utility.lower, self.utility.upper)

        if self.utilities is None:
            self.utilities = {}

        for agent in self.agents.values():
            utility_val = self.utilities.get(agent, self.utility.scale(self.utility.lower))
            utility_val = self.utility.unscale(utility_val)
            color = colormap(unorm(utility_val))
    
            if agent.ue_id == 1:
                color = 'blue'
            elif agent.ue_id == 2:
                color = 'purple'

            ax.scatter(
                agent.point.x,
                agent.point.y,
                s=200,
                zorder=2,
                color=color, 
                marker="o",
            )
            id_to_let = {1:'A', 2: 'B', 3:'C'}
            ax.annotate(id_to_let[agent.ue_id], xy=(agent.point.x, agent.point.y), ha="center", va="center")
            
        if not hasattr(self, 'custom_links'):
                 self.custom_links = {}

        # color links
        for (sender, receiver), color in self.custom_links.items():
            ax.plot(
                [sender.point.x, receiver.point.x],
                [sender.point.y, receiver.point.y],
                color=color, 
                path_effects=[
                    pe.SimpleLineShadow(shadow_color="black"),
                    pe.Normal(),
                ],
                linewidth=3,
                zorder=1, 
            )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])


    def render_radio_map(self, ax, radio_map, title) -> None:
        """
        Renders the loaded radio map on the given axis.
        """
        ax.set_title(title) 
        # radio map
        if radio_map is not None: 
            im = ax.imshow(radio_map, 
                            cmap='viridis', 
                            origin='lower',
                            extent=[0, self.width, 0, self.height],
                            vmin=-130, vmax=0) 
            
            fig = ax.get_figure()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical", label="Signal Strength (dBm)")
            
            ax.set_xlim([0, self.width])
            ax.set_ylim([0, self.height])
            
        # draw agents
        colormap = cm.get_cmap("RdYlGn")
        unorm = plt.Normalize(self.utility.lower, self.utility.upper)
        if self.utilities is None:
            self.utilities = {}

        for agent in self.agents.values():
            utility_val = self.utilities.get(agent, self.utility.scale(self.utility.lower))
            utility_val = self.utility.unscale(utility_val)
            color = colormap(unorm(utility_val))
            
            if agent.ue_id == 1:
                color = 'blue'
            elif agent.ue_id == 2:
                color = 'purple'

            ax.scatter(
                agent.point.x,
                agent.point.y,
                s=200,
                zorder=2,
                color=color,
                marker="o",
                edgecolors='white', 
                linewidth=1.5
            )
            id_to_let = {1:'A', 2: 'B', 3:'C'}
            ax.annotate(id_to_let[agent.ue_id], 
                        xy=(agent.point.x, agent.point.y), 
                        ha="center", 
                        va="center",
                        color='white', 
                        fontweight='bold')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    def render_dashboard(self, ax) -> None:
        """
        Renders a per-agent status dashboard ---
        """
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_title(f"Sim Time: {self.sim_time_seconds:.1f} s", fontweight='bold', fontsize=12)

        rows = ["Agent A", "Agent B", "Agent C"]
        cols = ["Rate (Mbps)", "Util", "Energy", "Dist (m)"] 

        # data for agent A
        dr_a, ut_a, en_a, dist_a = "N/A", "N/A", "N/A", "0.0"
        if 1 in self.agents:
            agent_a = self.agents[1]
            dr_a = f"{agent_a.current_datarate:.1f}"
            ut_a = f"{self.utilities.get(agent_a, 0.0):.2f}"
            en_a = f"{agent_a.energy:.1f}"
            dist_a = f"{agent_a.total_distance:.1f}" 

        # data for agent B
        dr_b, ut_b, en_b, dist_b = "N/A", "N/A", "N/A", "0.0"
        if 2 in self.agents:
            agent_b = self.agents[2]
            is_relay = any(sender == agent_b for sender, _ in self.custom_links.keys())
            dr_b = "Relay" if is_relay else "Idle"
            ut_b = f"{self.utilities.get(agent_b, 0.0):.2f}"
            en_b = f"{agent_b.energy:.1f}"
            dist_b = f"{agent_b.total_distance:.1f}" 

        # data for agent C
        dr_c, ut_c, en_c, dist_c = "N/A", "N/A", "N/A", "0.0"
        if 3 in self.agents:
            agent_c = self.agents[3]
            dr_c = "Receiving" if (1 in self.agents and self.agents[1].active_route) else "Idle"
            ut_c = f"{self.utilities.get(agent_c, 0.0):.2f}"
            en_c = f"{agent_c.energy:.1f}"
            dist_c = f"{agent_c.total_distance:.1f}" 

        text = [
            [dr_a, ut_a, en_a, dist_a],
            [dr_b, ut_b, en_b, dist_b],
            [dr_c, ut_c, en_c, dist_c],
        ]

        table = ax.table(
            cellText=text,
            rowLabels=rows,
            colLabels=cols,
            cellLoc="center",
            edges="B",
            loc="upper center",
            bbox=[0.0, -0.25, 1.0, 1.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10) 

    def render_avg_datarate(self, ax) -> None:
        """
        Renders running average of Agent A's datarate ---
        """
        if self.time > 0 and "datarate" in self.monitor.scalar_results:
            time = np.arange(self.time)
            datarates = self.monitor.scalar_results["datarate"]
            # calculate running average
            running_avg = np.cumsum(datarates) / (time + 1)
            ax.plot(time, running_avg, linewidth=1, color="black")

        ax.set_ylabel("Avg. Datarate (Mbps)")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, 110.0]) 

    def render_avg_energy(self, ax) -> None:
        """
        Renders running average of system energy usage per step ---
        """
        if self.time > 0 and "total_energy_consumed_step" in self.monitor.scalar_results:
            time = np.arange(self.time)
            energy_usage = self.monitor.scalar_results["total_energy_consumed_step"]

            running_avg = np.cumsum(energy_usage) / (time + 1)
            ax.plot(time, running_avg, linewidth=1, color="black")

        ax.set_xlabel("Time")
        ax.set_ylabel("Avg. Energy / Step") 
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, 100.0]) 