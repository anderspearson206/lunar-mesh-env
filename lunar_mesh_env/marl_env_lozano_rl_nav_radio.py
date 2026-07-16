"""
Lozano RL-nav env augmented with radio-map scalar observations.

Adds two keys on top of LunarRoverMeshLozanoRLNavEnv:
  radio_at_goal    (1,)  — BS signal strength at the agent's goal position (dBm)
  bs_grad_at_agent (2,)  — unit-vector gradient of BS radio map at agent position
                           points in the direction that maximises BS signal

These let the policy learn the joint nav+comm tradeoff without a CNN:
  "Is my goal in a coverage hole?"  → radio_at_goal
  "Which way improves BS signal?"   → bs_grad_at_agent
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv


class LunarRoverMeshLozanoRLNavRadioEnv(LunarRoverMeshLozanoRLNavEnv):

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        spaces_dict = dict(base.spaces)
        spaces_dict["radio_at_goal"]    = spaces.Box(-200.0, 0.0, shape=(1,), dtype=np.float32)
        spaces_dict["bs_grad_at_agent"] = spaces.Box(-1.0,   1.0, shape=(2,), dtype=np.float32)
        return spaces.Dict(spaces_dict)

    # ------------------------------------------------------------------
    # Lazy gradient precomputation — bs_radio_map is fixed after reset
    # ------------------------------------------------------------------

    def _ensure_bs_gradient(self):
        if not hasattr(self, "_bs_grad_x"):
            if self.bs_radio_map is not None:
                dy, dx = np.gradient(self.bs_radio_map.astype(np.float32))
                self._bs_grad_x = dx   # [y, x] indexed
                self._bs_grad_y = dy
            else:
                self._bs_grad_x = None
                self._bs_grad_y = None

    def reset(self, *args, **kwargs):
        # Clear cached gradient so it is recomputed after bs_radio_map is refreshed
        self._bs_grad_x = None
        self._bs_grad_y = None
        return super().reset(*args, **kwargs)

    # ------------------------------------------------------------------
    # Obs
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id):
        obs   = super()._get_obs(agent_id)
        agent = self.agent_map[agent_id]
        self._ensure_bs_gradient()

        # BS signal at goal position
        if self.bs_radio_map is not None:
            gy = int(np.clip(agent.goal_y, 0, self.bs_radio_map.shape[0] - 1))
            gx = int(np.clip(agent.goal_x, 0, self.bs_radio_map.shape[1] - 1))
            radio_at_goal = float(np.clip(self.bs_radio_map[gy, gx], -200.0, 0.0))
        else:
            radio_at_goal = 0.0
        obs["radio_at_goal"] = np.array([radio_at_goal], dtype=np.float32)

        # Gradient direction at agent position
        if self._bs_grad_x is not None:
            ay = int(np.clip(agent.y, 0, self._bs_grad_x.shape[0] - 1))
            ax = int(np.clip(agent.x, 0, self._bs_grad_x.shape[1] - 1))
            gx_val = float(self._bs_grad_x[ay, ax])
            gy_val = float(self._bs_grad_y[ay, ax])
            norm   = np.sqrt(gx_val ** 2 + gy_val ** 2) + 1e-8
            obs["bs_grad_at_agent"] = np.array([gx_val / norm, gy_val / norm], dtype=np.float32)
        else:
            obs["bs_grad_at_agent"] = np.zeros(2, dtype=np.float32)

        return obs
