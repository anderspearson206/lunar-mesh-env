"""
TorchMLPGATLozanoCropModel — Lozano model with a CNN encoder for the local
BS radio map crop (64×64, position-aware).

CNN output size (input 1×64×64):
  Conv(1→4,  k=4, s=4)  → 4×16×16
  Conv(4→8,  k=4, s=4)  → 8×4×4
  Flatten                → 128
  Linear(128→8)          → 8   [zero-init, no ReLU]

The 8-dim embedding is concatenated with the 13 scalar features before
the MLP (21-dim total), giving the move head position-aware coverage context.
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from models.mlp_gat_model import GATLayer
from models.mlp_gat_lozano_model import _SCALAR_KEYS, _SCALAR_NORMS, NODE_FEAT_DIM

_CROP_SIZE    = 64
_CNN_EMBED_DIM = 8


class TorchMLPGATLozanoCropModel(TorchModelV2, nn.Module):
    """Lozano MLP-GAT policy with CNN over a local BS radio map crop."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size     = int(np.prod(orig.spaces["action_mask"].shape))
        self.max_nodes     = int(np.prod(orig.spaces["graph_adj"].shape))
        self.node_feat_dim = NODE_FEAT_DIM

        scalar_dim = sum(int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS)
        move_dirs  = model_config.get("custom_model_config", {}).get("move_dirs", 1)

        # ── CNN branch — 64×64 position-aware BS radio map crop ───────────
        # Conv(1→4, k=4, s=4):  (64-4)/4+1 = 16  → 4×16×16
        # Conv(4→8, k=4, s=4):  (16-4)/4+1 =  4  → 8×4×4
        # Flatten: 128  →  Linear: 8
        cnn_linear = nn.Linear(128, _CNN_EMBED_DIM)
        nn.init.zeros_(cnn_linear.weight)
        nn.init.zeros_(cnn_linear.bias)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=4, stride=4),    # → 4×16×16
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=4, stride=4),    # → 8×4×4
            nn.ReLU(),
            nn.Flatten(),                                  # → 128
            cnn_linear,
            # No ReLU here: ReLU(0)=0 blocks gradients when zero-init,
            # causing the CNN to never activate. Linear output lets
            # gradients flow from step 1 so weights grow from zero.
        )

        # ── MLP branch — scalars + CNN embedding ──────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim + _CNN_EMBED_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),                          nn.ReLU(),
        )
        self.move_head = nn.Linear(256, move_dirs)

        # ── GAT branch ────────────────────────────────────────────────────
        self.gat1 = GATLayer(NODE_FEAT_DIM, 32, num_heads=4, concat=True,  dropout=0.2)
        self.gat2 = GATLayer(128,            64, num_heads=1, concat=False, dropout=0.2)
        self.elu  = nn.ELU()

        # ── Lozano comm head ──────────────────────────────────────────────
        self.comm_score = nn.Linear(64, 1)
        self.hold_logit = nn.Parameter(torch.zeros(1))

        # ── Value head ────────────────────────────────────────────────────
        self.value_head = nn.Linear(256 + 64, 1)

        self._current_value = None
        self._flat_offsets  = self._build_offsets(orig)

    def _build_offsets(self, orig_space):
        offsets = {}
        cursor  = 0
        for k in sorted(orig_space.spaces):
            dim        = int(np.prod(orig_space.spaces[k].shape))
            offsets[k] = (cursor, cursor + dim)
            cursor    += dim
        return offsets

    def _parse(self, obs):
        if isinstance(obs, dict):
            nf    = obs["graph_node_features"].float()
            adj   = obs["graph_adj"].float()
            msk   = obs["action_mask"].float()
            sc    = torch.cat([
                obs[k].float().reshape(obs[k].shape[0], -1) / _SCALAR_NORMS[k]
                for k in _SCALAR_KEYS
            ], dim=1)
            crop  = obs["bs_crop_obs"].float() / 200.0
        else:
            def _sl(key):
                s, e = self._flat_offsets[key]
                return obs[:, s:e].float()

            msk  = _sl("action_mask")
            adj  = _sl("graph_adj")
            nf   = _sl("graph_node_features").reshape(-1, self.max_nodes, self.node_feat_dim)
            sc   = torch.cat([_sl(k) / _SCALAR_NORMS[k] for k in _SCALAR_KEYS], dim=1)
            crop = _sl("bs_crop_obs").reshape(-1, 1, _CROP_SIZE, _CROP_SIZE).float() / 200.0

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)
        return nf, adj, msk, sc, crop

    def _build_adj(self, adj_1d):
        B, N = adj_1d.shape
        adj  = torch.zeros(B, N, N, device=adj_1d.device)
        adj[:, 0, :] = adj_1d
        adj[:, :, 0] = adj_1d
        eye  = torch.eye(N, device=adj_1d.device).unsqueeze(0)
        return torch.clamp(adj + eye, max=1.0)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nf, adj_1d, msk, scalars, crop = self._parse(obs)
        B = nf.shape[0]

        # ── CNN embedding (position-aware) ────────────────────────────────
        cnn_embed = self.map_cnn(crop)               # (B, 8)

        # ── MLP branch ────────────────────────────────────────────────────
        mlp_hidden  = self.mlp(torch.cat([scalars, cnn_embed], dim=1))
        move_logits = self.move_head(mlp_hidden)

        # ── GAT branch ────────────────────────────────────────────────────
        adj  = self._build_adj(adj_1d)
        h    = self.elu(self.gat1(nf,  adj))
        h    = self.elu(self.gat2(h,   adj))
        focal_embed = h[:, 0, :]

        # ── Lozano comm logits ────────────────────────────────────────────
        node_scores = self.comm_score(h[:, 1:, :]).squeeze(-1)
        hold_scores = self.hold_logit.expand(B, 1)
        comm_logits = torch.cat([hold_scores, node_scores], dim=1)
        avail = torch.cat(
            [torch.ones(B, 1, device=adj_1d.device), adj_1d[:, 1:]], dim=1
        )
        comm_logits = comm_logits.masked_fill(avail == 0, float("-inf"))

        # ── Value head ────────────────────────────────────────────────────
        self._current_value = self.value_head(
            torch.cat([mlp_hidden, focal_embed], dim=1)
        ).squeeze(1)

        # ── Combine and apply obs action mask ─────────────────────────────
        if self.num_outputs == self.max_nodes:
            logits = comm_logits
        else:
            logits = torch.cat([move_logits, comm_logits], dim=1)
        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value
