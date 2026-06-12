# ar_action_dist.py
"""
Autoregressive masked action distribution for the AlphaStar-style policy.

P(a) = P(move) * P(comm_0 | move) * ... * P(comm_bs | move, comm_0..2)

Modeled on ray/rllib/examples/_old_api_stack/models/autoregressive_action_dist.py
(Ray 2.54), generalized to 5 heads with per-head action masks. Actions are
returned stacked as (B, 5) to match TorchMultiCategorical's MultiDiscrete
contract.

`self.inputs` is the 256-d context produced by the model's forward pass (this
is what RLlib stores as ACTION_DIST_INPUTS). The per-head masks are read from
`self.model.last_action_mask`, cached by the model's forward on the same
batch — valid because RLlib always constructs the distribution immediately
after a forward pass, in both the sampling and the loss paths.
"""

import torch

from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)

from .alphastar_net import CONTEXT_DIM, MASK_SLICES, NUM_HEADS, masked_logits


class TorchARMaskedDistribution(TorchDistributionWrapper):
    def _mask_slice(self, head_idx):
        s, e = MASK_SLICES[head_idx]
        return self.model.last_action_mask[:, s:e]

    def _head_dist(self, head_idx, prefix_actions):
        logits = self.model.net.head_logits(self.inputs, head_idx, prefix_actions)
        return TorchCategorical(masked_logits(logits, self._mask_slice(head_idx)))

    def _sample_chain(self, deterministic):
        actions, logp = [], 0.0
        prefix = None
        for i in range(NUM_HEADS):
            dist = self._head_dist(i, prefix)
            a = dist.deterministic_sample() if deterministic else dist.sample()
            logp = logp + dist.logp(a)
            actions.append(a)
            prefix = torch.stack(actions, dim=1)
        self._action_logp = logp
        return prefix  # (B, NUM_HEADS)

    def sample(self):
        return self._sample_chain(deterministic=False)

    def deterministic_sample(self):
        return self._sample_chain(deterministic=True)

    def sampled_action_logp(self):
        return self._action_logp

    def logp(self, actions):
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.view(-1, NUM_HEADS)
        all_logits = self.model.net.all_head_logits(self.inputs, actions)
        total = 0.0
        for i in range(NUM_HEADS):
            ml = masked_logits(all_logits[i], self._mask_slice(i))
            total = total + TorchCategorical(ml).logp(actions[:, i])
        return total

    def entropy(self):
        # Sum of per-head entropies conditioned on one sampled chain
        # (same approximation as RLlib's autoregressive example).
        actions, total = [], 0.0
        prefix = None
        for i in range(NUM_HEADS):
            dist = self._head_dist(i, prefix)
            total = total + dist.entropy()
            actions.append(dist.sample())
            prefix = torch.stack(actions, dim=1)
        return total

    def kl(self, other):
        # Condition both distributions on self's sampled chain.
        actions, total = [], 0.0
        prefix = None
        for i in range(NUM_HEADS):
            d_self = self._head_dist(i, prefix)
            d_other = other._head_dist(i, prefix)
            total = total + d_self.kl(d_other)
            actions.append(d_self.sample())
            prefix = torch.stack(actions, dim=1)
        return total

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return CONTEXT_DIM
