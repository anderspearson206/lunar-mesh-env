import numpy as np
from .marl_env import LunarRoverMeshEnv
from typing import Set, Dict
from collections import Counter
from itertools import chain

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

        Returns:
            int: num distinct messages
        """
        return len(self.get_distinct_messages())
    
    def get_remaining_ttl_set(self):
        """ Gets the reaming time to live of all packets in the network

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

        Returns:
            _type_: _description_
        """
        messages = self.get_distinct_messages()
        num_seen = {p.packet_id: (len(p.touched)-1) for p in messages}
        return num_seen
        