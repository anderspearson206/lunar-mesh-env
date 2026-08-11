"""
Pure MLP policy for nav-only epidemic training.

Architecture:
  Scalar obs (21 dims): goal_dist(1) + goal_vector(2) + move_history(8) + position(2) + terrain_slopes(8)
  MLP:       Linear(21→256) → ReLU → Linear(256→256) → ReLU
  nav_head:  Linear(256→9)   — one logit per movement direction
  value:     Linear(256→1)

Flat obs layout (alphabetical, no graph keys — total 30 dims):
  [0:9]   action_mask               (9,)
  [9:10]  goal_dist                 (1,)  — dist/362, explicit nearness signal
  [10:12] goal_vector               (2,)
  [12:20] move_history              (8,)  — 4 steps × (dx,dy) in [-1,1]
  [20:22] position                  (2,)
  [22:30] terrain_slopes            (8,)  — Δheight per move dir, /MAX_INCLINE

move_history encodes displacement vectors (not raw action indices) so oscillation
is visible as zero net displacement. terrain_slopes gives continuous one-step
terrain look-ahead beyond the binary move mask, helping the policy route around
obstacles rather than discovering them on contact.

Commented out (re-enable by uncommenting in env + _SCALAR_KEYS):
  buffer_usage              (1,)
  other_agent_connectivity  (4,)
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

_SCALAR_KEYS = [
    # "buffer_usage",
    "goal_dist",
    "goal_vector",
    "move_history",
    # "other_agent_connectivity",
    "position",
    "terrain_slopes",
]
_NORM = {
    "buffer_usage":             1.0,
    "goal_dist":                1.0,    # already normalised by map diagonal in env
    "goal_vector":              256.0,
    "move_history":             1.0,    # (dx,dy) vectors already in [-1,1]
    "other_agent_connectivity": 1.0,
    "position":                 256.0,
    "terrain_slopes":           1.0,    # normalised by MAX_INCLINE in env; clipped [-5,5]
}


class TorchMLPNavOnlyModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size = int(np.prod(orig.spaces["action_mask"].shape))  # 9

        scalar_dim = sum(
            int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS
        )  # 13

        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.nav_head   = nn.Linear(256, num_outputs)  # 9 nav logits
        self.value_head = nn.Linear(256, 1)

        self._current_value = None

        # Per-dim normalisation for the scalar slice in the flat training path.
        # persistent=False → not saved in checkpoint; rebuilt from obs_space at load time.
        norm_parts = []
        for k in sorted(k for k in orig.spaces if k not in ("action_mask",)):
            dim = gym.spaces.utils.flatdim(orig.spaces[k])
            norm_parts.extend([_NORM.get(k, 256.0)] * dim)
        self.register_buffer(
            "_flat_norm",
            torch.tensor(norm_parts, dtype=torch.float32),
            persistent=False,
        )

    def _parse(self, obs):
        """Return (action_mask, scalars) as float tensors."""
        if isinstance(obs, dict):
            msk = obs["action_mask"].float()
            sc  = torch.cat(
                [obs[k].float().reshape(obs[k].shape[0], -1) / _NORM[k]
                 for k in _SCALAR_KEYS],
                dim=1,
            )
        else:
            ms = self.mask_size
            msk = obs[:, :ms].float()
            sc  = obs[:, ms:].float() / self._flat_norm
        return msk, sc

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        msk, scalars = self._parse(obs)

        hidden      = self.mlp(scalars)
        nav_logits  = self.nav_head(hidden)                        # (B, 9)

        self._current_value = self.value_head(hidden).squeeze(1)   # (B,)

        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return nav_logits + inf_mask, state

    def value_function(self):
        return self._current_value
