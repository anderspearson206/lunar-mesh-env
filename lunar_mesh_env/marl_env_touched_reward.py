"""
Experiment D — Touched-based relay credit.

When a packet is uniquely delivered to the BS, every agent in
packet.touched (originator + all relay hops) receives REWARD_RELAY.
This gives relay agents a proper gradient tied to actual delivery
outcomes, fixing the credit assignment gap in the base reward scheme.

The BS does not store Packet objects; packets remain in the sending
agent's buffer until the next step's cleanup_acked_packets, so we
search all agent buffers immediately after delivery.
"""

from .marl_env_mlp_gat import LunarRoverMeshMLPGATEnv


class LunarRoverMeshTouchedRewardEnv(LunarRoverMeshMLPGATEnv):
    REWARD_RELAY = 10.0  # per unique packet delivered, to every agent in touched

    def _handle_communication_step(self, actions, rewards, infos):
        prev_received = set(self.base_station.packets_received)
        result = super()._handle_communication_step(actions, rewards, infos)

        new_ids = self.base_station.packets_received - prev_received
        if not new_ids:
            return result

        # Packets are shared references and stay in sender buffers until next
        # step's cleanup, so search all agent buffers for the new packet IDs.
        found = {}
        for agent in self.agent_map.values():
            for p in agent.payload_manager.buffer:
                if p.packet_id in new_ids and p.packet_id not in found:
                    found[p.packet_id] = p
            if len(found) == len(new_ids):
                break

        for packet in found.values():
            for agent_id in packet.touched:
                if agent_id in rewards:
                    rewards[agent_id] += self.REWARD_RELAY

        return result
