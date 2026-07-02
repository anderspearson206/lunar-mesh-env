"""
Experiment A — Shared delivery reward.

When any packet reaches the BS, a fraction of the delivery reward is spread
equally to all active agents, not just the one that sent to the BS.
This gives relay agents and originators a gradient toward cooperative delivery.
"""

from .marl_env_mlp_gat import LunarRoverMeshMLPGATEnv


class LunarRoverMeshSharedRewardEnv(LunarRoverMeshMLPGATEnv):
    RELAY_REWARD_SHARE = 0.5   # fraction of delivery reward redistributed to non-senders

    def _handle_communication_step(self, actions, rewards, infos):
        pre_bs = self.base_station.num_packets_received
        super()._handle_communication_step(actions, rewards, infos)
        new_deliveries = self.base_station.num_packets_received - pre_bs
        if new_deliveries > 0 and len(self.agents) > 0:
            share = (
                new_deliveries * self.REWARD_PACKET_DELIVERY
                * self.RELAY_REWARD_SHARE / len(self.agents)
            )
            for aid in self.agents:
                rewards[aid] += share
