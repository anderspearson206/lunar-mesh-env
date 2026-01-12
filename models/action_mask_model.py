import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# models/action_mask_model.py

class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        
        # determine mask and feature sizes
        if hasattr(orig_space, "spaces") and "action_mask" in orig_space.spaces:
            self.mask_size = gym.spaces.utils.flatdim(orig_space.spaces["action_mask"])
            self.feature_dim = sum([gym.spaces.utils.flatdim(s) for k, s in orig_space.spaces.items() if k != "action_mask"])
        else:
            # sometimes rllib flattens the obs space beforehand
            self.mask_size = num_outputs 
            self.feature_dim = gym.spaces.utils.flatdim(orig_space) - self.mask_size

        # print(f"--- Model Init: Features={self.feature_dim} | Mask={self.mask_size} | Logits={num_outputs} ---")

        # base network
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
            # dict path
            feature_components = [
                obs["self_state"].float(),
                obs["terrain"].reshape(obs["terrain"].size(0), -1).float(),
                obs["neighbors"].reshape(obs["neighbors"].size(0), -1).float(),
                obs["base_station"].float(),
                obs["radio_map"].reshape(obs["radio_map"].size(0), -1).float(),
                obs["goal_vector"].float()
            ]
            flat_obs = torch.cat(feature_components, dim=1)
            action_mask = obs["action_mask"]
        else:
            # if flattened input
            action_mask = obs[:, :self.mask_size]
            flat_obs = obs[:, self.mask_size:]

        # forward pass
        features = self.internal_model[:-1](flat_obs)
        logits = self.internal_model[-1](features)
        self._current_value = self.value_head(features).squeeze(1)

        # mask
        inf_mask = torch.clamp(torch.log(action_mask.float()), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value