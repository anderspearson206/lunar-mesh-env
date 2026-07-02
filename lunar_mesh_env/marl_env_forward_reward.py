"""
Experiment B — Dense forwarding reward.

A small reward is given each time an agent successfully transfers a packet
to a peer rover that did not already have it.  This makes inter-agent
forwarding as dense a signal as the navigation reward.

Only peer-to-peer transfers are rewarded here (not agent→BS delivery,
which is already covered by REWARD_PACKET_DELIVERY in the base class).
"""

import numpy as np

from .marl_env import PACKET_SIZE_BITS, BURST_PROBABILITY, BURST_SIZE_MBITS
from .marl_env_mlp_gat import LunarRoverMeshMLPGATEnv


class LunarRoverMeshForwardRewardEnv(LunarRoverMeshMLPGATEnv):
    REWARD_FORWARD = 0.1   # reward per new packet received by a peer rover

    def _handle_communication_step(self, actions, rewards, infos):
        active_agents = [self.agent_map[aid] for aid in actions.keys()
                         if self.agent_map[aid].energy > 0]

        for agent in active_agents:
            if agent.bs_connected:
                agent.network_state["BS_0"].update(self.base_station.packets_received)

        for agent in active_agents:
            for neighbor in agent.neighbors:
                agent.merge_network_state(neighbor.network_state)

        for agent in active_agents:
            agent.cleanup_buffer()
            agent.drop_expired_packets(self.sim_time)

        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                continue

            comm_flags      = action[1:]
            peer_targets    = self._peer_targets_from_flags(agent, agent_id, comm_flags[:-1])
            for t in peer_targets:
                self.custom_links[(agent, t)] = 'green'
            targets_to_send = list(peer_targets)

            if comm_flags[-1] == 1 and agent.bs_connected:
                targets_to_send.append(self.base_station)
                self.custom_links[(agent, self.base_station)] = 'cyan'

            if targets_to_send:
                pre_delivery = self.base_station.num_packets_received
                if self.packet_mode == 'boolean':
                    agent.send_packet(targets_to_send, self.sim_time)
                else:
                    for target in targets_to_send:
                        if target is self.base_station:
                            agent.send_packets_to_target(target, self.STEP_LENGTH, self.sim_time)
                        else:
                            # Track new packets received by this peer rover
                            pre_peer = len(target.payload_manager.buffer)
                            agent.send_packets_to_target(target, self.STEP_LENGTH, self.sim_time)
                            new_fwd  = len(target.payload_manager.buffer) - pre_peer
                            if new_fwd > 0:
                                rewards[agent_id] += new_fwd * self.REWARD_FORWARD

                delivered_now = self.base_station.num_packets_received - pre_delivery
                rewards[agent_id] += delivered_now * self.REWARD_PACKET_DELIVERY

                tx_cost = self.COST_TX_5G_PER_STEP * len(targets_to_send)
                agent.energy -= tx_cost
                self.total_energy_consumed_step += tx_cost

            if self.packet_mode == 'boolean':
                if (np.random.rand() < self.PACKET_GEN_PROB
                        and not self.mission_done.get(agent_id, False)):
                    agent.generate_packet(size=10, time_to_live=50,
                                          destination="BS_0", time=self.sim_time)
            else:
                bits_needed   = self.TELEMETRY_RATE_MBPS * self.STEP_LENGTH * 1e6
                num_telemetry = int(bits_needed / PACKET_SIZE_BITS)
                for _ in range(num_telemetry):
                    agent.generate_packet(size=PACKET_SIZE_BITS, time_to_live=5000,
                                          destination="BS_0", time=self.sim_time)
                if np.random.rand() < BURST_PROBABILITY:
                    bits_burst = BURST_SIZE_MBITS * 1e6
                    num_burst  = int(bits_burst / PACKET_SIZE_BITS)
                    dtn_state  = agent.payload_manager.get_state()
                    space_left = dtn_state['buffer_size'] - dtn_state['payload_size']
                    for _ in range(min(num_burst, int(space_left / PACKET_SIZE_BITS))):
                        agent.generate_packet(size=PACKET_SIZE_BITS, time_to_live=5000,
                                              destination="BS_0", time=self.sim_time)

        return actions, rewards, infos
