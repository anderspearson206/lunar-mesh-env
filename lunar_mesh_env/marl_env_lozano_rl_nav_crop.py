"""
Lozano RL-nav env with a local crop of the BS radio map.

Adds one key on top of LunarRoverMeshLozanoRLNavEnv:
  bs_crop_obs  (1, 64, 64)  — 64×64 window of the BS radio map centred on
                               the agent's current position, padded with -200
                               (no-signal) at map edges, clipped to [-200, 0].

Unlike the full-map CNN (LunarRoverMeshLozanoRLNavCNNEnv), this crop changes
at every step as the agent moves, giving the policy position-aware coverage
information: "what does BS signal look like around me right now?"
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv
from .marl_env import PACKET_SIZE_BITS

CROP_SIZE = 64   # pixels; must be even
_HALF     = CROP_SIZE // 2


class LunarRoverMeshLozanoRLNavCropEnv(LunarRoverMeshLozanoRLNavEnv):
    REWARD_BS_CONNECTED  = 0.1
    REWARD_BW_UTIL       = 1.0

    def step(self, actions):
        prev_bs_packets = int(self.base_station.num_packets_received)
        obs, rewards, terms, truncs, infos = super().step(actions)

        # Bandwidth utilization reward: fraction of available BS link capacity used.
        packets_this_step = self.base_station.num_packets_received - prev_bs_packets
        bits_delivered    = packets_this_step * PACKET_SIZE_BITS
        bs_capacity_bits  = sum(
            self.radio_model.get_throughput_pos(a.x, a.y, self.base_station.x, self.base_station.y)
            * 1e6 * self.STEP_LENGTH
            for a in self.agent_map.values()
            if a.bs_connected and a.energy > 0
        )
        bw_util = min(bits_delivered / bs_capacity_bits, 1.0) if bs_capacity_bits > 0 else 0.0
        bw_reward = self.REWARD_BW_UTIL * bw_util

        for agent_id in rewards:
            agent = self.agent_map[agent_id]
            if agent.bs_connected:
                rewards[agent_id] += self.REWARD_BS_CONNECTED + bw_reward
        return obs, rewards, terms, truncs, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        spaces_dict = dict(base.spaces)
        spaces_dict["bs_crop_obs"] = spaces.Box(
            low=-200.0, high=0.0, shape=(1, CROP_SIZE, CROP_SIZE), dtype=np.float32
        )
        return spaces.Dict(spaces_dict)

    def _get_obs(self, agent_id):
        obs   = super()._get_obs(agent_id)
        agent = self.agent_map[agent_id]

        if self.bs_radio_map is not None:
            ax = int(np.clip(round(agent.x), 0, self.bs_radio_map.shape[1] - 1))
            ay = int(np.clip(round(agent.y), 0, self.bs_radio_map.shape[0] - 1))
            # Pad so out-of-bounds positions become -200 (no signal)
            padded = np.pad(
                self.bs_radio_map, _HALF,
                mode='constant', constant_values=-200.0,
            )
            # In the padded array, agent is at (ay+_HALF, ax+_HALF);
            # crop [ay : ay+CROP_SIZE] × [ax : ax+CROP_SIZE] gives a window
            # centred on the agent.
            crop = padded[ay : ay + CROP_SIZE, ax : ax + CROP_SIZE]
            crop = np.clip(crop, -200.0, 0.0).astype(np.float32)
        else:
            crop = np.full((CROP_SIZE, CROP_SIZE), -200.0, dtype=np.float32)

        obs["bs_crop_obs"] = crop[np.newaxis, :, :]   # (1, 64, 64)
        return obs
