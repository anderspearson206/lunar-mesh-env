"""
TorchMLPGATLozanoRadioModel — extends TorchMLPGATLozanoModel with radio scalar features.

Extra scalar inputs (appended after base scalars):
  radio_at_goal    (1,)  normalised by 200  → [-1, 0]
  bs_grad_at_agent (2,)  already unit-vector → [-1, 1]

Everything else (GAT branch, comm head, masking) is inherited unchanged.
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.mlp_gat_lozano_model import TorchMLPGATLozanoModel, _SCALAR_KEYS, _SCALAR_NORMS

_RADIO_KEYS  = ["bs_grad_at_agent", "radio_at_goal"]
_RADIO_NORMS = {
    "bs_grad_at_agent": 1.0,
    "radio_at_goal":    200.0,
}


class TorchMLPGATLozanoRadioModel(TorchMLPGATLozanoModel):
    """Lozano model + radio-map scalar features."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        orig       = getattr(obs_space, "original_space", obs_space)
        extra_dim  = sum(int(np.prod(orig.spaces[k].shape)) for k in _RADIO_KEYS)
        base_dim   = self.mlp[0].in_features   # scalar_dim from parent (e.g. 12)
        new_dim    = base_dim + extra_dim

        move_dirs  = model_config.get("custom_model_config", {}).get("move_dirs", 1)

        # Rebuild MLP with wider input; value_head dims are unchanged
        self.mlp       = nn.Sequential(
            nn.Linear(new_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
        )
        self.move_head = nn.Linear(256, move_dirs)

    def _parse(self, obs):
        nf, adj, msk, sc = super()._parse(obs)

        if isinstance(obs, dict):
            extra = torch.cat([
                obs[k].float().reshape(obs[k].shape[0], -1) / _RADIO_NORMS[k]
                for k in _RADIO_KEYS
            ], dim=1)
        else:
            def _sl(key):
                s, e = self._flat_offsets[key]
                return obs[:, s:e].float()
            extra = torch.cat([_sl(k) / _RADIO_NORMS[k] for k in _RADIO_KEYS], dim=1)

        return nf, adj, msk, torch.cat([sc, extra], dim=1)
