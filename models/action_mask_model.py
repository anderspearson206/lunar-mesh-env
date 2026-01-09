import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Calculate input dimension based on your space
        # (Self_state: 4) + (Terrain: 65536) + (Neighbors: 4) + 
        # (BaseStation: 2) + (RadioMap: 65536) + (Goal: 2) = 131,084
        input_dim = 131084 

        self.internal_model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )
        
        # Value function head
        self.value_head = nn.Linear(input_dim, 1)
        self._current_value = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the dictionary of tensors
        obs = input_dict["obs"]
        
        # 1. Manually flatten and concatenate everything except the mask
        # Note: input_dict["obs"] is an OrderedDict in multi-agent dict spaces
        flat_obs = torch.cat([
            obs["self_state"].float(),
            obs["terrain"].reshape(obs["terrain"].size(0), -1).float(),
            obs["neighbors"].reshape(obs["neighbors"].size(0), -1).float(),
            obs["base_station"].float(),
            obs["radio_map"].reshape(obs["radio_map"].size(0), -1).float(),
            obs["goal_vector"].float()
        ], dim=1)

        # 2. Compute logits and value
        logits = self.internal_model(flat_obs)
        self._current_value = self.value_head(flat_obs).squeeze(1)

        # 3. Apply Action Mask
        action_mask = obs["action_mask"]
        # Use a very large negative number for masked actions
        inf_mask = torch.clamp(torch.log(action_mask.float()), min=-1e9)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self._current_value