"""
Radio-augmented graph-observation env for comparing three RSSI encoding strategies.

Extends LunarRoverMeshMLPGATEnv with a configurable ``radio_obs`` parameter:

  "rssi"  – 4 normalised RSSI scalars (one per peer + BS), shape (4,)
  "crop"  – 64×64 crop of local radio map centred on agent, shape (1, 64, 64)
  "full"  – full 256×256 radio map, shape (1, 256, 256)

Values are normalised to [0, 1] via (dBm + 200) / 200.
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env_mlp_gat import LunarRoverMeshMLPGATEnv

RADIO_MODES = ("rssi", "crop", "full")
_CROP_HALF  = 32          # half-edge of the 64×64 local crop
_DBM_SHIFT  = 200.0       # maps [-200, 0] dBm → [0, 1]


class LunarRoverMeshRadioEnv(LunarRoverMeshMLPGATEnv):
    """MLP-GAT env with one additional radio observation key."""

    def __init__(self, *args, radio_obs: str = "rssi", **kwargs):
        assert radio_obs in RADIO_MODES, f"radio_obs must be one of {RADIO_MODES}"
        self.radio_obs = radio_obs
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    def _get_obs(self, agent_id):
        obs = super()._get_obs(agent_id)
        agent = self.agent_map[agent_id]

        if self.radio_obs == "rssi":
            rssi_vals = []
            for tid in self.possible_agents:
                if tid == agent_id:
                    rssi_vals.append(1.0)
                else:
                    peer = self.agent_map[tid]
                    raw  = self.radio_model.get_signal_strength(
                        agent.x, agent.y, peer.x, peer.y)
                    rssi_vals.append(float(np.clip((raw + _DBM_SHIFT) / _DBM_SHIFT, 0.0, 1.0)))
            raw_bs = self.radio_model.get_signal_strength(
                agent.x, agent.y, self.base_station.x, self.base_station.y)
            rssi_vals.append(float(np.clip((raw_bs + _DBM_SHIFT) / _DBM_SHIFT, 0.0, 1.0)))
            obs["radio_rssi"] = np.array(rssi_vals, dtype=np.float32)

        else:
            # Both crop and full need the raw radio map from the agent
            raw_dict = agent.get_local_observation(list(self.agent_map.values()))
            rm2d     = raw_dict["radio_map"][0].astype(np.float32)  # (256, 256) dBm

            if self.radio_obs == "crop":
                ax = int(np.clip(np.round(agent.x), 0, 255))
                ay = int(np.clip(np.round(agent.y), 0, 255))
                r0, r1 = max(0, ay - _CROP_HALF), min(256, ay + _CROP_HALF)
                c0, c1 = max(0, ax - _CROP_HALF), min(256, ax + _CROP_HALF)
                patch  = rm2d[r0:r1, c0:c1]
                if patch.shape != (64, 64):
                    padded = np.full((64, 64), -200.0, dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch  = padded
                obs["radio_crop"] = np.clip(
                    (patch + _DBM_SHIFT) / _DBM_SHIFT, 0.0, 1.0
                ).astype(np.float32)[np.newaxis]   # (1, 64, 64)

            else:  # "full"
                obs["radio_map"] = np.clip(
                    (rm2d + _DBM_SHIFT) / _DBM_SHIFT, 0.0, 1.0
                ).astype(np.float32)[np.newaxis]   # (1, 256, 256)

        return obs

    # ------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base        = super().observation_space(agent)
        spaces_dict = dict(base.spaces)

        num_nodes = len(self.possible_agents) + 1
        if self.radio_obs == "rssi":
            spaces_dict["radio_rssi"] = spaces.Box(0.0, 1.0, shape=(num_nodes,),     dtype=np.float32)
        elif self.radio_obs == "crop":
            spaces_dict["radio_crop"] = spaces.Box(0.0, 1.0, shape=(1, 64, 64),      dtype=np.float32)
        else:
            spaces_dict["radio_map"]  = spaces.Box(0.0, 1.0, shape=(1, 256, 256),    dtype=np.float32)

        return spaces.Dict(spaces_dict)
