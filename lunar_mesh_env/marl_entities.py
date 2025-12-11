# marl_entities.py

import numpy as np
from typing import List, Dict, Tuple

# forward-declare to avoid a circular import
if False:
    from .radio_model_nn import RadioMapModelNN

class MarlMeshAgent:
    """
    A Standalone Agent that acts as a Mesh Node.
    """
    def __init__(self,
                 ue_id: int,
                 velocity: float,
                 snr_tr: float,
                 noise: float,
                 height: float,
                 frequency: float, 
                 tx_power: float,   
                 bw: float):       
        
        self.ue_id = ue_id
        self.velocity = velocity
        self.height = height
        
        # position (initialize at 0, will be set by env.reset)
        self.x: float = 0.0
        self.y: float = 0.0
        
        # radio properties
        self.snr_tr = snr_tr
        self.noise = noise
        self.frequency = frequency
        self.tx_power = tx_power
        self.bw = bw
        self.snr_threshold = snr_tr
        
        # sim state
        self.active_route = None 
        self.current_datarate = 0.0
        self.energy = 1000.0 
        self.is_moving = False 
        self.total_distance = 0.0

    def __str__(self):
        return f"MeshAgent_{self.ue_id}"

    def get_local_observation(self, 
                              env_heightmap: np.ndarray, 
                              all_agents: List['MarlMeshAgent'],
                              bs_location: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """
        Generates the agent-centric observation dictionary.
        Now includes relative position to the Base Station.
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

        # gather neighbor relative positions
        neighbor_features = []
        for other in all_agents:
            if other.ue_id == self.ue_id:
                continue
            
            # calculate rel position
            dx = other.x - self.x
            dy = other.y - self.y
            neighbor_features.append([dx, dy])
        
        # if no neighbors (single agent dev), pad with zeros
        if len(neighbor_features) == 0:
            neighbor_features = [[0.0, 0.0]]

        neighbors_arr = np.array(neighbor_features, dtype=np.float32)
        
        # calculate relative vector to the global base station
        bs_dx = bs_location[0] - self.x
        bs_dy = bs_location[1] - self.y
        bs_obs = np.array([bs_dx, bs_dy], dtype=np.float32)

        return {
            "self_state": my_state,
            "terrain": terrain_obs,
            "neighbors": neighbors_arr,
            "base_station": bs_obs # Added to dictionary
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
        HIGH_SPEED_THRESH_DBM = -80.0
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
        self.ue_id = "BS_0"  

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)