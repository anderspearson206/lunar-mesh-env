import torch
import torch.nn as nn
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# Spatial keys excluded from the policy input.
# The action mask already encodes terrain constraints (valid moves),
# and radio connectivity is captured in other_agent_connectivity.
# Feeding raw 256x256 maps into a flat linear layer buries the useful
# 19-dim scalar signal in 131K noise features and prevents learning.
_SPATIAL_KEYS = {"terrain", "radio_map", "action_mask"}

# Scalar keys fed to the network (alphabetical = flat-obs order after mask).
_SCALAR_KEYS = [
    # -- disabled (kept for future use) --
    # "buffer_usage",             # (1,)
    # "energy",                   # (1,)
    # "num_packets",              # (1,)
    # "other_agent_connectivity", # (num_agents+1,)
    # "other_agent_vectors",      # (num_agents+1, 2)
    # ------------------------------------
    "goal_vector",              # (2,)
    "move_history",             # (HISTORY_LEN,) — last N movement actions, scaled to [0,256]
    "position",                 # (2,)
]

# Normalisation divisors per key (bring values into roughly [-1, 1]).
_NORM = {
    # -- disabled (kept for future use) --
    # "buffer_usage":             1.0,
    # "energy":                   5_000_000.0,
    # "num_packets":              1000.0,
    # "other_agent_connectivity": 1.0,
    # "other_agent_vectors":      256.0,
    # ------------------------------------
    "goal_vector":              256.0,
    "move_history":             256.0,
    "position":                 256.0,
}


class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)

        if hasattr(orig_space, "spaces") and "action_mask" in orig_space.spaces:
            self.mask_size = gym.spaces.utils.flatdim(orig_space.spaces["action_mask"])
            # feature_dim = scalar keys only (no terrain / radio_map)
            self.feature_dim = sum(
                gym.spaces.utils.flatdim(s)
                for k, s in orig_space.spaces.items()
                if k not in _SPATIAL_KEYS
            )
        else:
            # Flattened obs: mask occupies the first num_outputs slots,
            # then compact scalars. Total flat dim - mask - map dims = scalars.
            total_dim = gym.spaces.utils.flatdim(orig_space)
            map_dim = 256 * 256 * 2          # terrain + radio_map
            self.mask_size = num_outputs
            self.feature_dim = total_dim - self.mask_size - map_dim

        # Policy network
        self.internal_model = nn.Sequential(
            nn.Linear(int(self.feature_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )
        self.value_head = nn.Linear(256, 1)
        self._current_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        if isinstance(obs, dict):
            # Dict path (eval / compute_single_action): build from scalar keys only.
            parts = []
            for k in _SCALAR_KEYS:
                v = obs[k].float().reshape(obs[k].size(0), -1)
                parts.append(v / _NORM[k])
            flat_obs     = torch.cat(parts, dim=1)
            action_mask  = obs["action_mask"].float()
        else:
            # Flat path (training): scalars sit between mask and the two maps.
            # Layout (alphabetical): mask | scalars(feature_dim) | radio_map | terrain
            action_mask = obs[:, :self.mask_size].float()
            flat_obs    = obs[:, self.mask_size: self.mask_size + self.feature_dim] / 256.0

        features = self.internal_model[:-1](flat_obs)
        logits   = self.internal_model[-1](features)
        self._current_value = self.value_head(features).squeeze(1)

        inf_mask = torch.clamp(torch.log(action_mask), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value
