# payload_manager.py

from collections import deque, defaultdict
from time import time
from typing import Set, Dict, List
import random
import math
import copy

class Packet:
    counter = 0

    def __init__(self, source, destination, size, time_to_live, gen_step, origin_x, origin_y):
        self.packet_id = f"P_{Packet.counter}"
        Packet.counter += 1
        self.size = size
        self.gen_step = gen_step  
        self.time_to_live = time_to_live
        self.source = source
        self.destination = destination
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.touched = {source}
    
    def __str__(self):
        return str(self.get_state())
    
    def __repr__(self):
        return f"Packet({self.packet_id}, src={self.source}, dest={self.destination}, gen_step={self.gen_step})"
    
    def get_state(self):
        return {
            'id': self.packet_id, 
            'size': self.size, 
            'gen_step': self.gen_step,
            'ttl': self.time_to_live, 
            'source': self.source, 
            'dest':self.destination, 
            'origin_loc':(self.origin_x, self.origin_y), 
            'touched':self.touched
        }
        
    def clone(self):
        """Creates a safe memory copy of the packet."""
        new_p = copy.copy(self)
        new_p.touched = set(self.touched)
        return new_p


class PayloadManager:
    def __init__(self, id, buffer_size=1000):
        self.id = id
        self.buffer = deque()
        self.num_packets_generated = 0
        self.buffer_size = buffer_size
        self.payload_size = 0 
        self.seen = set()
    
    def __str__(self):
        name = f"Payload Manager {self.id}."
        name += f"Generated: {self.num_packets_generated} packets."
        name+= f"Buffer Capacity: {self.buffer_size}"    
        return name
    
    def send_packet(self, targets, current_step: int):
        if len(self.buffer) > 0:
            packet = self.buffer.popleft()
            self.payload_size -= packet.size
            for target in targets:
                target.receive_packet(packet.clone(), current_step)
                
                
    def send_packets_rate_aware(self, target_agent, target_known_packets: Set[str], 
                                data_rate_mbps: float, step_duration: float, current_step: int) -> int:
        """
        Iterates through buffer, sends what the target 
        doesn't have, up to the bandwidth limit.
        """
        # calculate capacity in bits
        # Mbps * 1e6 * seconds = bits allowed
        limit_bits = data_rate_mbps * 1e6 * step_duration
        
        sent_count = 0
        bits_sent = 0
        packets_to_deliver = []

        # iterate without popping
        for packet in self.buffer:
            if bits_sent >= limit_bits:
                break
                
            # check if target already has packet
            if packet.packet_id in target_known_packets:
                continue 

            # check bw
            if (bits_sent + packet.size) <= limit_bits:
                packets_to_deliver.append(packet)
                bits_sent += packet.size
                sent_count += 1
            else:
                break 

        # send / copy to target
        for p in packets_to_deliver:
            target_agent.receive_packet(p.clone(), current_step)
            
        return sent_count

    def cleanup_acked_packets(self, global_acked_ids: Set[str]):
        """
        Removes packets from the buffer that are known to be at the BS.
        """
        new_buffer = deque()
        new_payload_size = 0
        
        for packet in self.buffer:
            if packet.packet_id in global_acked_ids:
                continue 
            else:
                new_buffer.append(packet)
                new_payload_size += packet.size
        
        self.buffer = new_buffer
        self.payload_size = new_payload_size
                
    # we may not need this and can instead just call send packet to self
    def send_and_duplicate_packet(self, targets):
        if len(self.buffer) > 0:
            packet = self.buffer.popleft()
            self.buffer.appendleft(packet)  
            for target in targets:
                target.receive_packet(packet)
                
    def drop_expired_packets(self, current_step):
        # Update to use simulation steps instead of time()
        while len(self.buffer) > 0:
            packet = self.buffer[0]
            if current_step - packet.gen_step > packet.time_to_live:
                self.buffer.popleft()
                self.payload_size -= packet.size
            else:
                break
    
    def receive_packet(self, packet:Packet):
        # don't accept duplicates
        for p in self.buffer:
            if p.packet_id == packet.packet_id:
                return
        
        while packet.size + self.payload_size > self.buffer_size:
            old_packet = self.buffer.popleft()
            self.payload_size -= old_packet.size
        
        packet.touched.add(self.id)
        self.buffer.append(packet)
        self.payload_size += packet.size

    def generate_packet(self, size, time_to_live, destination, current_step, origin_x, origin_y):
        while size + self.payload_size > self.buffer_size:
            packet = self.buffer.popleft()
            self.payload_size -= packet.size
        
        packet = Packet(source=self.id, destination=destination, size=size, 
                        time_to_live=time_to_live, gen_step=current_step, 
                        origin_x=origin_x, origin_y=origin_y)
        self.buffer.append(packet)
        self.payload_size += size
        self.num_packets_generated += 1    
            
    def calculate_hbd_utility(packet, current_step, lambda_param, total_nodes, history_tracker, optimize="rate"):
        T_i = current_step - packet.gen_step
        R_i = packet.time_to_live - T_i
        
        if R_i <= 0:
            return -float('inf') # expired
            
        samples = history_tracker.get_samples(T_i)
        
        # if the network is young and we have no history, fallback to FBD for safety
        if not samples:
            m_i = len(packet.touched) - 1
            n_i = 1 
            samples = [(m_i, n_i)]
            
        if optimize == "rate":
            # HBD eq for rate: lambda * R_i * E[ (1 - M(T)/(L-1)) * exp(-lambda * R_i * N(T)) ]
            expected_val = 0
            for m, n in samples:
                term = (1 - (m / (total_nodes - 1))) * math.exp(-lambda_param * R_i * n)
                expected_val += term
            expected_val /= len(samples)
            
            return lambda_param * R_i * expected_val
            
        elif optimize == "delay":
            # HBD Eq for delay: E[ (L-1-M(T))/N(T) ]^2 / ( (L-1)*(L-1 - E[M(T)]) * lambda )
            sum_e_term = 0
            sum_m = 0
            for m, n in samples:
                n_safe = max(1, n)
                sum_e_term += (total_nodes - 1 - m) / n_safe
                sum_m += m
                
            E_term = sum_e_term / len(samples)
            E_m = sum_m / len(samples)
            
            # max() prevents division by zero if all nodes have seen the packet
            denominator = (total_nodes - 1) * max(0.1, (total_nodes - 1 - E_m)) * lambda_param
        
        return (E_term ** 2) / denominator
    
    
    def get_state(self):
        return {
            "buffer_size": self.buffer_size,
            "payload_size": self.payload_size,
            "num_packets": len(self.buffer),
            "num_packets_generated": self.num_packets_generated
        }
        
    
# This is used for the optimal buffer policy for DTN networks, 
# paper: https://ieeexplore.ieee.org/document/4557763
# essentially, we track the history of packets that have existed for a time T, 
# and we 
class HistoryTracker:
    def __init__(self, bucket_size_steps=100, max_samples_per_bucket=100):
        self.bucket_size = bucket_size_steps
        self.max_samples = max_samples_per_bucket
        # maps time bucket index to list of (m_value, n_value) tuples
        self.history = defaultdict(list)
        
    def record_observation(self, elapsed_steps, m, n):
        """Records an observation of a packet's state at a specific age."""
        bucket = int(elapsed_steps // self.bucket_size)
        
        # reservoir sampling to maintain a rolling statistical profile
        if len(self.history[bucket]) < self.max_samples:
            self.history[bucket].append((m, n))
        else:
            replace_idx = random.randint(0, self.max_samples - 1)
            self.history[bucket][replace_idx] = (m, n)
            
    def get_samples(self, elapsed_steps):
        """Returns historical samples for this age bucket."""
        bucket = int(elapsed_steps // self.bucket_size)
        return self.history.get(bucket, [])