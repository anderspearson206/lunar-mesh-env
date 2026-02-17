# payload_manager.py

from collections import deque
from time import time
from typing import Set, Dict, List

class Packet:
    counter = 0

    def __init__(self, source, destination, size, time_to_live):
        self.packet_id = f"P_{Packet.counter}"
        Packet.counter += 1
        self.size = size
        self.timestamp = time()
        self.time_to_live = time_to_live
        self.source = source
        self.destination = destination
        self.touched = {source}
    
    def copy(self):
        new_packet = Packet(self.source, self.destination, self.size, self.time_to_live)
        new_packet.packet_id = self.packet_id
        new_packet.timestamp = self.timestamp
        new_packet.touched = set(self.touched)
        return new_packet


class PayloadManager:
    def __init__(self, id, buffer_size=1000):
        self.id = id
        self.buffer = deque()
        self.num_packets_generated = 0
        self.buffer_size = buffer_size
        self.payload_size = 0 
    
    def send_packet(self, targets):
        if len(self.buffer) > 0:
            packet = self.buffer.popleft()
            self.payload_size -= packet.size
            for target in targets:
                target.receive_packet(packet.copy())
    
    def send_packets_rate_aware(self, target_agent, target_known_packets: Set[str], 
                                data_rate_mbps: float, step_duration: float) -> int:
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
            target_agent.receive_packet(p.copy())
            
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
                target.receive_packet(packet.copy())

    def send_all_packets(self, targets):
        current_time = time()
        new_buffer = deque()
        for packet in self.buffer:
            if current_time - packet.timestamp <= packet.time_to_live:
                new_buffer.append(packet)
                for target in targets:
                    target.receive_packet(packet.copy())

        self.buffer = new_buffer         
    
    def drop_expired_packets(self):
        current_time = time()
        while len(self.buffer) > 0:
            packet = self.buffer[0]
            if current_time - packet.timestamp > packet.time_to_live:
                # print(f"Packet {packet.packet_id} expired and dropped from agent {self.id}'s buffer.")
                # print(f"Time in buffer: {current_time - packet.timestamp:.2f}s, TTL: {packet.time_to_live}s")
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

    def generate_packet(self, size, time_to_live, destination):
        while size + self.payload_size > self.buffer_size:
            packet = self.buffer.popleft()
            self.payload_size -= packet.size
        
        packet = Packet(source=self.id, destination=destination, size=size, time_to_live=time_to_live)
        self.buffer.append(packet)
        self.payload_size += size
        self.num_packets_generated += 1
    
    def get_state(self):
        return {
            "buffer_size": self.buffer_size,
            "payload_size": self.payload_size,
            "num_packets": len(self.buffer),
            "num_packets_generated": self.num_packets_generated
        }
        
    