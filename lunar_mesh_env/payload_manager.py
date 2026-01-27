# payload_manager.py

from collections import deque
from time import time

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
                target.receive_packet(packet)
                
                
                
    # we may not need this and can instead just call send packet to self
    def send_and_duplicate_packet(self, targets):
        if len(self.buffer) > 0:
            packet = self.buffer.popleft()
            self.buffer.appendleft(packet)  
            for target in targets:
                target.receive_packet(packet)
                
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