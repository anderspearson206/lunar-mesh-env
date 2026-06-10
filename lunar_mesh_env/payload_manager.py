# payload_manager.py

import copy as _copy
from collections import deque
from time import time
from typing import Set, Dict, List

class Packet:
    counter = 0

    def __init__(self, source, destination, size, time_to_live, gen_step, origin_x, origin_y, copies=1):
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
        self.copies = copies  # spray-and-wait token count; 1 = wait mode


class PayloadManager:
    def __init__(self, id, buffer_size=1000):
        self.id = id
        self.buffer = deque()
        self.num_packets_generated = 0
        self.buffer_size = buffer_size
        self.payload_size = 0 
    
    def send_packet(self, targets, current_step: int):
        if len(self.buffer) > 0:
            packet = self.buffer.popleft()
            self.payload_size -= packet.size
            for target in targets:
                target.receive_packet(packet, current_step)
        
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
            target_agent.receive_packet(p, current_step)
            
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

    def generate_packet(self, size, time_to_live, destination, current_step, origin_x, origin_y, copies=1):
        while size + self.payload_size > self.buffer_size:
            packet = self.buffer.popleft()
            self.payload_size -= packet.size

        packet = Packet(source=self.id, destination=destination, size=size,
                        time_to_live=time_to_live, gen_step=current_step,
                        origin_x=origin_x, origin_y=origin_y, copies=copies)
        self.buffer.append(packet)
        self.payload_size += size
        self.num_packets_generated += 1

    # ------------------------------------------------------------------
    # DTN routing helpers (called by env routing protocol, not actions)
    # ------------------------------------------------------------------

    def epidemic_forward(self, target: 'PayloadManager') -> int:
        """Copy all held packets to target that it does not already have.

        The sender keeps its own copy — this is true epidemic replication.
        Returns the number of packets forwarded.
        """
        target_ids = {p.packet_id for p in target.buffer}
        forwarded = 0
        for packet in self.buffer:
            if packet.packet_id not in target_ids:
                target.receive_packet(packet)
                forwarded += 1
        return forwarded

    def spray_forward(self, target: 'PayloadManager') -> int:
        """Spray-and-wait: split token copies with target for sprayable packets.

        For each packet with copies > 1 that the target does not have:
          - give target floor(copies / 2) tokens
          - keep ceil(copies / 2) tokens
        Packets with copies == 1 are in wait mode and are not forwarded
        (they should only be delivered directly to the destination/BS).
        Returns the number of packets forwarded.
        """
        target_ids = {p.packet_id for p in target.buffer}
        forwarded = 0
        for packet in self.buffer:
            if packet.copies <= 1:
                continue
            if packet.packet_id in target_ids:
                continue
            give = packet.copies // 2
            packet.copies -= give
            p_copy = _copy.copy(packet)
            p_copy.touched = packet.touched.copy()
            p_copy.copies = give
            target.receive_packet(p_copy)
            forwarded += 1
        return forwarded
    
    def get_state(self):
        return {
            "buffer_size": self.buffer_size,
            "payload_size": self.payload_size,
            "num_packets": len(self.buffer),
            "num_packets_generated": self.num_packets_generated
        }
        
    