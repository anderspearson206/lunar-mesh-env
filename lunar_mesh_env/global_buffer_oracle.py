import numpy as np
from .marl_env import LunarRoverMeshEnv
from typing import Set, Dict
from collections import Counter
from itertools import chain

# lambda is the estimated parameter of the inter-encounter times of nodes
# described by an exponential dist with parameter lambda
LAMBDA = 0.04

class GlobalBufferOracle:
    """
    This will be used as an oracle that knows all network state information
    """
    def __init__(self, env: LunarRoverMeshEnv):
        self.env = env

    def get_num_nodes(self):
        """ Gets the total number of nodes in the environment
            L
        Returns:
            int: number of nodes (rovers)
        """
        return self.env.num_agents
            
    def get_distinct_messages(self):
        """ Gets the distinct messages in the environment at this step

        returns:
            set: distinct messages         
        """
        distinct_messages = set()
        for aid in self.env.agents:
            agent = self.env.agent_map[aid]
            buf = agent.payload_manager.buffer
            distinct_messages = distinct_messages.union(buf)
        
        return distinct_messages
    
    def get_num_distinct_messages(self):
        """ Gets the number of distinct messages
            K(t)
        Returns:
            int: num distinct messages
        """
        return len(self.get_distinct_messages())
    
    def get_remaining_ttl_set(self):
        """ Gets the reaming time to live of all packets in the network
            R
        Returns:
            dict: dict of packet ids and r_ttl
        """
        packets_set = self.get_distinct_messages()
        current_step = self.env.sim_time
        packet_ttl_status = {
            p.packet_id: (p.time_to_live - (current_step - p.gen_step)) 
            for p in packets_set
        }

        return packet_ttl_status
    
    def get_num_copies(self):
        """ gets the total number of copies in the network for each packet
            n(T)
        Returns:
            dict: keys p_id, values:counts
        """
        num_copies = dict()
        bufs = [self.env.agent_map[aid].payload_manager.buffer for aid in self.env.agents]
        num_copies = Counter(p.packet_id for p in chain.from_iterable(bufs))
        num_copies = dict(num_copies)
        
        return num_copies
    
    def get_num_seen(self):
        """ Gets the number of nodes (excluding source) that have seen each message
            m(T)
        Returns:
            dict: keys pid, values counts of nodes that have seen
        """
        messages = self.get_distinct_messages()
        num_seen = {}
        for p in messages:
            bs_modifier = 1 if "BS_0" in p.touched else 0
            # m_i = total touched - source - base_station
            num_seen[p.packet_id] = len(p.touched) - 1 - bs_modifier 
            
        return num_seen
        
        
    def utility_calculator(self, messages, metric: str = 'delivery_rate'):
        """Calculates the utility of messages

        Args:
            messages: list of message p_id values
            metric (str, optional): _description_. Defaults to 'delivery rate'. The other option
            is delivery_delay
            
        Returns:
            dict: packet_id keys, metric values
        """
        _R = self.get_remaining_ttl_set()
        _m= self.get_num_seen()
        
        _n = self.get_num_copies()
        
        
        L = self.get_num_nodes()
        m = np.array([_m.get(pid, 0) for pid in messages])
        n = np.array([_n.get(pid, 1) for pid in messages])
        R = np.array([_R.get(pid, 0) for pid in messages])
        
        def rate_utility(mi, ni, Ri, l):
            return (1-(mi/(l-1)))*LAMBDA*Ri*np.exp(-LAMBDA*ni*Ri)
        
        def delay_utility(mi, ni, l):
            return (1/(np.power(ni, 2)*LAMBDA))*(1-(mi/(l-1)))
        
        final_metrics = dict()
        if metric == 'delivery_rate':
            rate_met = rate_utility(m, n, R, L)
            for i, pid in enumerate(messages):
                final_metrics[pid] = rate_met[i]
            return final_metrics
        
        elif metric == 'delivery_delay':
            delay_met = delay_utility(m, n, L)
            for i, pid in enumerate(messages):
                final_metrics[pid] = delay_met[i]
            pass
        else:
            return None
        
        