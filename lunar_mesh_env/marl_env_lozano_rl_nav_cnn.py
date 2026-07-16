"""
Lozano RL-nav env with BS radio map observation for CNN encoding.

Adds one key on top of LunarRoverMeshLozanoRLNavEnv:
  bs_radio_obs  (1, 256, 256)  — BS radio coverage map in dBm, clipped to [-200, 0]

The model (TorchMLPGATLozanoCNNModel) compresses this via a small CNN into a
64-dim embedding that is concatenated with the MLP scalar branch.
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv


class LunarRoverMeshLozanoRLNavCNNEnv(LunarRoverMeshLozanoRLNavEnv):

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        spaces_dict = dict(base.spaces)
        spaces_dict["bs_radio_obs"] = spaces.Box(
            low=-200.0, high=0.0, shape=(1, 256, 256), dtype=np.float32
        )
        return spaces.Dict(spaces_dict)

    def _get_obs(self, agent_id):
        obs = super()._get_obs(agent_id)
        if self.bs_radio_map is not None:
            obs["bs_radio_obs"] = np.clip(
                self.bs_radio_map, -200.0, 0.0
            )[np.newaxis, :, :].astype(np.float32)
        else:
            obs["bs_radio_obs"] = np.zeros((1, 256, 256), dtype=np.float32)
        return obs
