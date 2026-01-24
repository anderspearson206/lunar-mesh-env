import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # CNN for image data(terrain and radio map)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=3), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 256), 
            nn.ReLU()
        )

        # mlp for vector data (self state, neighbors, base station, goal vector)
        self.vector_branch = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU()
        )

        # combined heads
        self.policy_head = nn.Linear(320, num_outputs)
        self.value_head = nn.Linear(320, 1)
        self._current_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # image data
        terrain = obs["terrain"]
        radio_map = obs["radio_map"]
        img_input = torch.cat([terrain, radio_map], dim=1) 
        img_features = self.cnn_branch(img_input.float())

        # vector data
        vec_input = torch.cat([
            obs["self_state"].float(),
            obs["neighbors"].reshape(obs["neighbors"].size(0), -1).float(),
            obs["base_station"].float(),
            obs["goal_vector"].float()
        ], dim=1)
        vec_features = self.vector_branch(vec_input)

        # merge
        combined = torch.cat([img_features, vec_features], dim=1)
        logits = self.policy_head(combined)
        self._current_value = self.value_head(combined).squeeze(1)

        # mask
        action_mask = obs["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask.float()), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value