import numpy as np
from typing import List, Dict, Tuple
from mobile_env.core.entities import UserEquipment

# forward-declare RadioMapModel to avoid a circular import
if False:
    from .radio_model_nn import RadioMapModelNN

class MeshAgent(UserEquipment):
    """
    A UserEquipment that can also act as a transmitter.
    It adds 'frequency', 'tx_power', and 'bw' attributes to be
    compatible with channel models like OkumuraHata when
    this agent is passed as the 'bs' (transmitter) argument.
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
        
        super().__init__(
            ue_id=ue_id,
            velocity=velocity,
            snr_tr=snr_tr,
            noise=noise,
            height=height
        )
        
        # new tx attributres
        self.frequency = frequency
        self.tx_power = tx_power
        self.bw = bw
        
        # new policy /energy params
        self.active_route = None 
        self.current_datarate = 0.0
        self.energy = 1000.0 
        self.is_moving = False 

    def __str__(self):
        return f"MeshAgent: {self.ue_id}"


    def policy_decide_route(self,
                             target_agent: 'MeshAgent', 
                             all_agents: List['MeshAgent'], 
                             radio_model: 'RadioMapModel', 
                             width: float, 
                             height: float
                            ) -> Dict[Tuple['MeshAgent', 'MeshAgent'], str]:
        """
        Agent's high-level policy. Tries high-speed, then falls back to robust.
        Sets self.active_route and self.current_datarate.
        Returns a dictionary of links { (sender, receiver): 'color' } for drawing.
        """
        
        # these thresholds determine if the link works, 
        # should be updated with real values
        # TODO
        HIGH_SPEED_THRESH_DBM = -80.0
        ROBUST_THRESH_DBM = -100.0 

        # try to find route
        
        route_5g = self.find_route(target_agent, all_agents, radio_model, 
                                   width, height, 
                                   dbm_thresh=HIGH_SPEED_THRESH_DBM, 
                                   frequency='5.8')
        
        if route_5g:
            self.active_route = (route_5g, '5.8')
            # TODO this datarate should also be made accurate
            
            self.current_datarate = 100.0
           
            return {link: 'green' for link in route_5g}

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
                   target_agent: 'MeshAgent', 
                   all_agents: List['MeshAgent'], 
                   radio_model: 'RadioMapModel', 
                   width: float, 
                   height: float, 
                   dbm_thresh: float, 
                   frequency: str
                  ) -> List[Tuple['MeshAgent', 'MeshAgent']]:
        """
        Agent's internal pathfinding logic for a *given frequency and threshold*.
        Calculates the best path to a target, either direct or via one relay.
        Returns a list of links [(sender, receiver), ...] if a path is found,
        or an empty list [] if not.
        """
        
        map_a = radio_model.generate_map((self.x, self.y), frequency)
        
        # find direct
        dbm_a_c = MeshAgent.get_signal_strength(map_a, target_agent, width, height)
        
        # find the relay
        agent_b = None
        for agent in all_agents:
            if agent.ue_id == 2: 
                agent_b = agent
                break
        
        # check direct 
        if dbm_a_c > dbm_thresh:
            return [(self, target_agent)] 

        if agent_b is None:
            return [] 
            
        # check relay
        map_b = radio_model.generate_map((agent_b.x, agent_b.y), frequency)

        dbm_a_b = MeshAgent.get_signal_strength(map_a, agent_b, width, height)
        dbm_b_c = MeshAgent.get_signal_strength(map_b, target_agent, width, height)

        if dbm_a_b > dbm_thresh and dbm_b_c > dbm_thresh:
            return [(self, agent_b), (agent_b, target_agent)]

        return []

    @staticmethod
    def get_signal_strength(radio_map, target_agent, width, height):
        """
        Gets the signal strength (dBm) from a given radio_map
        at the target_agent's current (x, y) location.
        """
        if radio_map is None:
            return -np.inf # No map, no signal
        
        map_shape = radio_map.shape
 
        x, y = target_agent.x, target_agent.y
        
        x = np.clip(x, 0, width)
        y = np.clip(y, 0, height)


        col_idx = int((x / width) * (map_shape[1] - 1))
        
        row_idx = int((y / height) * (map_shape[0] - 1))
        
        col_idx = np.clip(col_idx, 0, map_shape[1] - 1)
        row_idx = np.clip(row_idx, 0, map_shape[0] - 1)
        
        return radio_map[row_idx, col_idx]