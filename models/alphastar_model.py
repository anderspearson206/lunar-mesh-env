# alphastar_model.py
"""
RLlib ModelV2 wrapper around AlphaStarPolicyNet.

- Loads SL-pretrained weights on every worker via
  custom_model_config["sl_weights"] (more robust than syncing from the local
  policy only).
- If custom_model_config["kl_coeff_sl"] > 0, keeps a frozen copy of the SL
  policy and adds KL(pi_current || pi_SL) to the PPO loss via the
  `custom_loss` hook (TorchPolicyV2 calls it right after the PPO loss on the
  same train batch, so the context cached by `forward` is reused).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch

from .alphastar_net import (
    CONTEXT_DIM,
    MASK_SLICES,
    AlphaStarPolicyNet,
    masked_logits,
)

# Canonical scalar feature order fed to the net (each divided by 256).
_SCALAR_KEYS = ["position", "goal_vector", "move_history"]
_SPATIAL_KEYS = ["screen", "minimap"]


class TorchAlphaStarModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        assert num_outputs == CONTEXT_DIM, \
            f"expected num_outputs={CONTEXT_DIM} (set by the action dist), got {num_outputs}"

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert hasattr(orig_space, "spaces"), \
            "TorchAlphaStarModel requires the Dict obs space from SpatialObsWrapper"
        # Flat layout: gymnasium flattens Dict obs in alphabetical key order.
        self._flat_slices = {}
        self._shapes = {}
        offset = 0
        for k in sorted(orig_space.spaces.keys()):
            d = gym.spaces.utils.flatdim(orig_space.spaces[k])
            self._flat_slices[k] = (offset, offset + d)
            self._shapes[k] = orig_space.spaces[k].shape
            offset += d

        cfg = model_config.get("custom_model_config", {})
        self.kl_coeff_sl = float(cfg.get("kl_coeff_sl", 0.0))
        sl_weights = cfg.get("sl_weights")

        self.net = AlphaStarPolicyNet()
        if sl_weights:
            state = torch.load(sl_weights, map_location="cpu")
            self.net.load_state_dict(state)

        # Frozen SL reference, deliberately NOT registered as a submodule:
        # keeps it out of the checkpoint state_dict (so eval models built
        # without it can restore checkpoints) and out of the optimizer.
        self._sl_ref = None
        if self.kl_coeff_sl > 0.0:
            assert sl_weights, "kl_coeff_sl > 0 requires sl_weights"
            ref = AlphaStarPolicyNet()
            ref.load_state_dict(torch.load(sl_weights, map_location="cpu"))
            ref.requires_grad_(False)
            ref.eval()
            self._sl_ref = (ref,)   # tuple wrapper defeats nn.Module registration

        self.last_action_mask = None
        self._ctx = None
        self._value = None
        self._last_obs = None
        self._sl_kl_metric = None

    # ------------------------------------------------------------------

    def _split(self, obs):
        """-> (screen, minimap, scalars, action_mask) from dict or flat obs."""
        if isinstance(obs, dict):
            screen = obs["screen"].float()
            minimap = obs["minimap"].float()
            scalars = torch.cat(
                [obs[k].float().reshape(obs[k].shape[0], -1) / 256.0
                 for k in _SCALAR_KEYS], dim=1)
            mask = obs["action_mask"].float()
        else:
            obs = obs.float()
            def part(key):
                s, e = self._flat_slices[key]
                return obs[:, s:e]
            screen = part("screen").reshape(-1, *self._shapes["screen"])
            minimap = part("minimap").reshape(-1, *self._shapes["minimap"])
            scalars = torch.cat([part(k) / 256.0 for k in _SCALAR_KEYS], dim=1)
            mask = part("action_mask")
        return screen, minimap, scalars, mask

    def forward(self, input_dict, state, seq_lens):
        screen, minimap, scalars, mask = self._split(input_dict["obs"])
        self.last_action_mask = mask
        self._last_obs = (screen, minimap, scalars)
        self._ctx = self.net.encode(screen, minimap, scalars)
        self._value = self.net.value(self._ctx)
        return self._ctx, state

    def value_function(self):
        return self._value

    # ------------------------------------------------------------------

    def custom_loss(self, policy_loss, train_batch):
        """Add kl_coeff_sl * KL(pi_current || pi_SL) to the PPO loss."""
        if self._sl_ref is None or self.kl_coeff_sl <= 0.0:
            return policy_loss

        sl_ref = self._sl_ref[0]
        actions = train_batch[SampleBatch.ACTIONS].long()
        cur_logits = self.net.all_head_logits(self._ctx, actions)
        with torch.no_grad():
            screen, minimap, scalars = self._last_obs
            if next(sl_ref.parameters()).device != screen.device:
                sl_ref.to(screen.device)
            sl_ctx = sl_ref.encode(screen, minimap, scalars)
            sl_logits = sl_ref.all_head_logits(sl_ctx, actions)

        kl_per_sample = 0.0
        for i, (s, e) in enumerate(MASK_SLICES):
            m = self.last_action_mask[:, s:e]
            cur_logp = F.log_softmax(masked_logits(cur_logits[i], m), dim=-1)
            sl_logp = F.log_softmax(masked_logits(sl_logits[i], m), dim=-1)
            kl_per_sample = kl_per_sample + (
                cur_logp.exp() * (cur_logp - sl_logp)
            ).sum(dim=-1)
        sl_kl = kl_per_sample.mean()

        # Plain float: RLlib's learner-info reduce calls np.isnan() on metric
        # values, which cannot convert CUDA tensors.
        self._sl_kl_metric = float(sl_kl.detach().cpu())
        policy_loss[0] = policy_loss[0] + self.kl_coeff_sl * sl_kl
        return policy_loss

    def metrics(self):
        # Surfaced as info.learner.<policy_id>.model.sl_kl in train results.
        if getattr(self, "_sl_kl_metric", None) is None:
            return {}
        return {"sl_kl": self._sl_kl_metric}
