import numpy as np
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from .payload_manager import Packet, PayloadManager

# forward-declare to avoid a circular import
if False:
    from .radio_model_nn import RadioMapModelNN


class BaseStation:
    """
    A static entity representing the Lander/Base Station.
    """
    def __init__(self, x: float, y: float, height: float = 1.0):
        self.x = x
        self.y = y
        self.height = height # bs could be higher than rovers, 
        # but we'd have to retrain the model for that.
        # Can we use the 3m height model for BS? used in RLD paper?
        self.id = "BS_0"  
        self.num_packets_received = 0
        self.num_duplicates_received = 0
        self.packets_received = set()

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def receive_packet(self, packet:Packet):
        self.num_packets_received += 1
        if packet.packet_id in self.packets_received:
            self.num_duplicates_received += 1
        else:
            self.packets_received.add(packet.packet_id)
    
    def get_state(self):
        state = {
            "id": self.id,
            "pos": (self.x, self.y),
            "num_packets_received": self.num_packets_received,
            "num_duplicates_received": self.num_duplicates_received
        }
        return state


class MarlAgent(ABC):
    """
    Abstract Base Class for Agents in the Lunar Mesh Environment.
    """
    @abstractmethod
    def receive_packet(self, packet:Packet):
        pass
    
    
    
class MarlMeshDTNAgent(MarlAgent):
    """
    A rover agent with full communication capabilities using DTN logic and movement logic.
    """
    def __init__(self,
                 ue_id: int,
                 x: float = 0.0,
                 y: float = 0.0,
                 buffer_size: int = 1000, 
                 bs: BaseStation = None):        
        
        self.ue_id = ue_id 
        self.id = f"MarlAgent_{ue_id}" 
        
        # position and movement state
        self.x = x
        self.y = y
        self.goal_x: float = 0.0
        self.goal_y: float = 0.0
        self.is_moving = False 
        self.total_distance = 0.0
        
        # energy
        self.energy = 2000.0 
        
        # dtn logic
        self.neighbors: set['MarlAgent'] = set()
        
        bs = bs if bs is not None else BaseStation(0.0, 0.0)
        self.base_station = bs
        
        self.bs_connected: bool = False
        
        self.payload_manager = PayloadManager(self.id, buffer_size=buffer_size)
        

    def __str__(self):
        return self.id

    def __lt__(self, other):
        """Allows auto-sorting by ue_id."""
        if not isinstance(other, MarlMeshDTNAgent):
            return NotImplemented
        return self.ue_id < other.ue_id
    
    def receive_packet(self, packet: Packet):
        self.payload_manager.receive_packet(packet)

    def send_packet(self, targets: List[MarlAgent|BaseStation]):
        self.payload_manager.send_packet(targets)

    def generate_packet(self, size: int, time_to_live: float, destination: str):
        self.payload_manager.generate_packet(size, time_to_live, destination)

    def update_neighbors(self, 
                        all_agents: List['MarlMeshAgent'], 
                        radio_model, 
                        width: float, 
                        height: float, 
                        dbm_thresh: float = -100.0, 
                        frequency: str = '5.8'
                       ) -> set['MarlMeshAgent']:
        """
        Updates the set of neighboring agents based on radio signal strength.
        """
        # get map centered at self
        map_a = radio_model.generate_map((self.x, self.y), frequency)
        self.neighbors = set()
        
        # check signal strength to all other agents and add to neighbors if above threshold
        for agent in all_agents:
            if agent.ue_id == self.ue_id:
                continue
                
            dbm = self.get_signal_strength(map_a, agent, width, height)
            if dbm > dbm_thresh:
                self.neighbors.add(agent)
        
        return self.neighbors

    def update_bs_connection(self, 
                             radio_model,
                             width: float,
                             height: float,
                             dbm_thresh: float = -100.0,
                             frequency: str = '5.8'
                            ) -> bool:
        """
        Updates the connection status to the base station.
        """
        bs_rm = radio_model.generate_map((self.x, self.y), frequency)
        dbm = self.get_signal_strength(bs_rm, self, width, height)
        self.bs_connected = dbm > dbm_thresh
        return self.bs_connected
        
    def get_local_observation(self, 
                              env_heightmap: np.ndarray, 
                              all_agents: List['MarlMeshAgent'],
                              bs_location: Tuple[float, float],
                              radio_map: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        TODO
        """
        dtn_state = self.payload_manager.get_state()
        
        # Self State (enegy, position, neighbors, DTN buffer state)
        # I think the normalized buffer usage is better than both the buffer size (doesnt change)
        # and the total payload size seperately. I also don't think we need to keep track of the 
        # number of packets generated, but we can keep it in the payload manager state for logging.
        my_state = np.array([
            self.energy, 
            self.x, 
            self.y, 
            dtn_state["payload_size"] / dtn_state["buffer_size"], 
            float(dtn_state["num_packets"])
        ], dtype=np.float32)
        
        # terrain processing
        # we expand the dimensions to add a channel dim if needed (for input to NN)
        terrain_obs = env_heightmap.astype(np.float32)
        if len(terrain_obs.shape) == 2:
            terrain_obs = np.expand_dims(terrain_obs, axis=0)

        # radio map
        if radio_map is None:
            rm_obs = np.zeros_like(terrain_obs)
        else:
            rm_obs = radio_map.astype(np.float32)
            if len(rm_obs.shape) == 2:
                rm_obs = np.expand_dims(rm_obs, axis=0)

        # neighbor relative positions
        # the neighbor positions are different than the set of connected neighbors above
        # these are also stored as relative positions (works better for RL)
        num_others = len(all_agents)
        connectivity = np.zeros(num_others + 1, dtype=np.float32)
        rel_positions = np.zeros((num_others + 1, 2), dtype=np.float32)

        # for all rovers
        all_agents = sorted(all_agents, key=lambda a: a.ue_id)
        for i, other in enumerate(all_agents):
            if other.ue_id == self.ue_id:
                # always 0 for self (already initialized to zero)
                continue
            
            # check connectivity
            if other in self.neighbors:
                connectivity[i] = 1.0
            
            # Relative position
            rel_positions[i] = [other.x - self.x, other.y - self.y]
        
        # vectors to goal and bs
        bs_obs = np.array([bs_location[0] - self.x, bs_location[1] - self.y], dtype=np.float32)
        goal_obs = np.array([self.goal_x - self.x, self.goal_y - self.y], dtype=np.float32)
        
        bs_idx = -1
        # the bs_connected should be updated every environment step before calling this
        # (or after??)
        if self.bs_connected:
            connectivity[bs_idx] = 1.0
        rel_positions[bs_idx] = [bs_location[0] - self.x, bs_location[1] - self.y]

        # updating this to unpack everything into the observation dict
        return {
            "energy": self.energy,
            "position": np.array([self.x, self.y], dtype=np.float32),
            "buffer_usage": np.array([dtn_state["payload_size"] / dtn_state["buffer_size"]], dtype=np.float32),
            "num_packets": np.array([dtn_state["num_packets"]], dtype=np.float32),
            "other_agent_vectors": rel_positions,
            "other_agent_connectivity": connectivity,
            "goal_vector": goal_obs,
            "terrain": terrain_obs,
            "radio_map": rm_obs
        }

    @staticmethod
    def get_signal_strength(radio_map, target_agent, width, height):
        if radio_map is None: return -np.inf 
        map_shape = radio_map.shape
        col_idx = int((target_agent.x / width) * (map_shape[1] - 1))
        row_idx = int((target_agent.y / height) * (map_shape[0] - 1))
        col_idx = np.clip(col_idx, 0, map_shape[1] - 1)
        row_idx = np.clip(row_idx, 0, map_shape[0] - 1)
        return radio_map[row_idx, col_idx]










"""
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||| BELOW IS DEPRECATED/LEGACY CODE |||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
"""




class MarlMeshAgent(MarlAgent):
    """
    A Standalone Agent that acts as a Mesh Node.
    """
    def __init__(self,
                 ue_id: int,
                 x: float = 0.0,
                 y: float = 0.0):        
        
        self.ue_id = ue_id
        # position (initialize at 0, will be set by env.reset)
        self.x = x
        self.y = y
        
        self.goal_x: float = 0.0
        self.goal_y: float = 0.0
        
        # sim state
        self.active_route = None 
        self.current_datarate = 0.0
        self.energy = 2000.0 
        self.is_moving = False 
        self.total_distance = 0.0
        
        # pathfinding state
        self.nav_path: List[Tuple[int, int]] = []

    def __str__(self):
        return f"MeshAgent_{self.ue_id}"
    
    def receive_packet(self, packet):
        return super().receive_packet(packet)

    def get_local_observation(self, 
                              env_heightmap: np.ndarray, 
                              all_agents: List['MarlMeshAgent'],
                              bs_location: Tuple[float, float],
                              radio_map: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generates the agent observation dictionary.
        1. Self State
        2. Terrain
        3. Neighbor Relative Positions
        4. Base Station Relative Position
        5. Radio Map
        """
        my_state = np.array([
            self.energy, 
            self.x, 
            self.y, 
            self.current_datarate
        ], dtype=np.float32)
        
        terrain_obs = env_heightmap.astype(np.float32)
        if len(terrain_obs.shape) == 2:
            terrain_obs = np.expand_dims(terrain_obs, axis=0)

        # rm obs
        if radio_map is None:
            # Assuming square map matching terrain size
            rm_obs = np.zeros_like(terrain_obs)
        else:
            rm_obs = radio_map.astype(np.float32)
            if len(rm_obs.shape) == 2:
                rm_obs = np.expand_dims(rm_obs, axis=0)

        # get neighbor relative positions
        neighbor_features = []
        for other in all_agents:
            if other.ue_id == self.ue_id:
                continue
            
            # calculate rel position
            dx = other.x - self.x
            dy = other.y - self.y
            neighbor_features.append([dx, dy])
        
        if len(neighbor_features) == 0:
            neighbor_features = [[0.0, 0.0]]

        neighbors_arr = np.array(neighbor_features, dtype=np.float32)
        
        # base station vector
        bs_dx = bs_location[0] - self.x
        bs_dy = bs_location[1] - self.y
        bs_obs = np.array([bs_dx, bs_dy], dtype=np.float32)
        
        # goal vector
        goal_dx = self.goal_x - self.x
        goal_dy = self.goal_y - self.y
        goal_obs = np.array([goal_dx, goal_dy], dtype=np.float32)

        return {
            "self_state": my_state,
            "terrain": terrain_obs,
            "neighbors": neighbors_arr,
            "base_station": bs_obs,
            "goal_vector": goal_obs,
            "radio_map": rm_obs
        }


    @staticmethod
    def get_signal_strength(radio_map, target_agent, width, height):
        """
        Gets the signal strength (dBm) from a given radio_map at target's location.
        """
        if radio_map is None:
            return -np.inf 
        
        map_shape = radio_map.shape
        
        col_idx = int((target_agent.x / width) * (map_shape[1] - 1))
        row_idx = int((target_agent.y / height) * (map_shape[0] - 1))
        
        col_idx = np.clip(col_idx, 0, map_shape[1] - 1)
        row_idx = np.clip(row_idx, 0, map_shape[0] - 1)
        
        return radio_map[row_idx, col_idx]


    # UNUSED
    def heuristic_resolve_route(self,
                             target_agent: 'MarlMeshAgent', 
                             all_agents: List['MarlMeshAgent'], 
                             radio_model, 
                             width: float, 
                             height: float
                            ) -> Dict[Tuple['MarlMeshAgent', 'MarlMeshAgent'], str]:
        """
        LEGACY: Agent's hardcoded logic to find a route.
        """
        HIGH_SPEED_THRESH_DBM = -90.0
        ROBUST_THRESH_DBM = -100.0 

        # try 5.8GHz
        route_5g = self.find_route(target_agent, all_agents, radio_model, 
                                   width, height, 
                                   dbm_thresh=HIGH_SPEED_THRESH_DBM, 
                                   frequency='5.8')
        
        if route_5g:
            self.active_route = (route_5g, '5.8')
            self.current_datarate = 100.0
            return {link: 'green' for link in route_5g}

        # fallback to 415MHz 
        route_415 = self.find_route(target_agent, all_agents, radio_model, 
                                    width, height, 
                                    dbm_thresh=ROBUST_THRESH_DBM, 
                                    frequency='415')
        if route_415:
            self.active_route = (route_415, '415')
            self.current_datarate = 10.0
            return {link: 'orange' for link in route_415}

        self.active_route = None
        self.current_datarate = 0.0
        return {} 

    # UNUSED
    def find_route(self, 
                   target_agent: 'MarlMeshAgent', 
                   all_agents: List['MarlMeshAgent'], 
                   radio_model, 
                   width: float, 
                   height: float, 
                   dbm_thresh: float, 
                   frequency: str
                  ) -> List[Tuple['MarlMeshAgent', 'MarlMeshAgent']]:
        """
        Internal pathfinding helper for the heuristic policy.
        """
        # NOTE: In the new system, we should try to use a cached map at some point
        map_a = radio_model.generate_map((self.x, self.y), frequency)
        
        # direct link
        dbm_a_c = MarlMeshAgent.get_signal_strength(map_a, target_agent, width, height)
        if dbm_a_c > dbm_thresh:
            return [(self, target_agent)] 

        # check simple relay
        agent_b = next((a for a in all_agents if a.ue_id == 2), None)
        
        if agent_b is None or agent_b.ue_id == self.ue_id or agent_b.ue_id == target_agent.ue_id:
            return [] 
            
        map_b = radio_model.generate_map((agent_b.x, agent_b.y), frequency)
        dbm_a_b = MarlMeshAgent.get_signal_strength(map_a, agent_b, width, height)
        dbm_b_c = MarlMeshAgent.get_signal_strength(map_b, target_agent, width, height)

        if dbm_a_b > dbm_thresh and dbm_b_c > dbm_thresh:
            return [(self, agent_b), (agent_b, target_agent)]

        return []

class MarlDtnAgent(MarlAgent):
    counter = 0  # Class-level counter for unique IDs

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0):        
        
        self.id = f'MeshAgent_{MarlDtnAgent.counter}'
        MarlDtnAgent.counter += 1
        
        # Position (initialize at provided values, will be set by env.reset)
        self.x = x
        self.y = y
        
        self.neighbors = set()
        self.payload_manager = PayloadManager(self.id)

    def __str__(self):
        return self.id

    def send_packet(self, targets:list[MarlAgent|BaseStation]):
        self.payload_manager.send_packet(targets)

    def receive_packet(self, packet:Packet):
        self.payload_manager.receive_packet(packet)

    def generate_packet(self, size, time_to_live, destination):
        self.payload_manager.generate_packet(size, time_to_live, destination)


    def update_neighbors(self, 
                   agents: List['MarlDtnAgent'], 
                   radio_model, 
                   width: float, 
                   height: float, 
                   dbm_thresh: float, 
                   frequency: str
                  ) -> set['MarlDtnAgent']:
        
        map_a = radio_model.generate_map((self.x, self.y), frequency)
        self.neighbors = set()
        for agent in agents:
            # direct link
            dbm_a_c = MarlDtnAgent.get_signal_strength(map_a, agent, width, height)
            if dbm_a_c > dbm_thresh:
                self.neighbors.add(agent)

        return self.neighbors

    def get_state(self) -> Dict[str, any]:
        state = {
            "id": self.id,
            "pos": (self.x, self.y),
            
            # Payload/bundle delivery metrics
            "payloads": self.payload_manager.get_state(),
        }
        return state
