"""
TorchMLPGATLozanoCNNModel — Lozano model with a CNN encoder for the BS radio map.

Architecture:
  CNN branch:    bs_radio_obs (1×256×256) → 3 conv layers → 64-dim embedding
  Scalar branch: same 4 scalars as base model → normalised and concatenated
  MLP:           [scalars (12) + CNN embedding (64)] → 256 → 256
  GAT branch:    unchanged — graph_node_features + graph_adj
  Move head:     MLP output → 9 (RL nav)
  Comm head:     GAT focal embedding → per-node score + hold logit
  Value head:    [MLP output + GAT focal] → 1

CNN output size verification (input 1×256×256):
  Conv(1→16, k=8, s=8)  → 16×32×32
  Conv(16→32, k=4, s=4) → 32×8×8
  Conv(32→32, k=4, s=4) → 32×2×2
  Flatten               → 128
  Linear(128→64)        → 64
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from models.mlp_gat_model import GATLayer
from models.mlp_gat_lozano_model import _SCALAR_KEYS, _SCALAR_NORMS, NODE_FEAT_DIM

_CNN_EMBED_DIM = 64


class TorchMLPGATLozanoCNNModel(TorchModelV2, nn.Module):
    """Lozano MLP-GAT policy with CNN-encoded BS radio map."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size     = int(np.prod(orig.spaces["action_mask"].shape))
        self.max_nodes     = int(np.prod(orig.spaces["graph_adj"].shape))
        self.node_feat_dim = NODE_FEAT_DIM

        scalar_dim = sum(int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS)
        move_dirs  = model_config.get("custom_model_config", {}).get("move_dirs", 1)

        # ── CNN branch — compresses 1×256×256 BS radio map ────────────────
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=8),   # → 16×32×32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4),  # → 32×8×8
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=4),  # → 32×2×2
            nn.ReLU(),
            nn.Flatten(),                                  # → 128
            nn.Linear(128, _CNN_EMBED_DIM),
            nn.ReLU(),
        )

        # ── MLP branch — scalars + CNN embedding ──────────────────────────
        mlp_in = scalar_dim + _CNN_EMBED_DIM
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_offsets(self, orig_space):
        offsets = {}
        cursor  = 0
        for k in sorted(orig_space.spaces):
            dim        = int(np.prod(orig_space.spaces[k].shape))
            offsets[k] = (cursor, cursor + dim)
            cursor    += dim
        return offsets

    def _parse(self, obs):
        """Return (node_feats, adj_1d, action_mask, scalars, bs_map) tensors."""
        if isinstance(obs, dict):
            nf    = obs["graph_node_features"].float()
            adj   = obs["graph_adj"].float()
            msk   = obs["action_mask"].float()
            sc    = torch.cat([
                obs[k].float().reshape(obs[k].shape[0], -1) / _SCALAR_NORMS[k]
                for k in _SCALAR_KEYS
            ], dim=1)
            bsmap = obs["bs_radio_obs"].float() / 200.0  # → [-1, 0]
        else:
            def _sl(key):
                s, e = self._flat_offsets[key]
                return obs[:, s:e].float()

            msk   = _sl("action_mask")
            adj   = _sl("graph_adj")
            nf    = _sl("graph_node_features").reshape(-1, self.max_nodes, self.node_feat_dim)
            sc    = torch.cat([_sl(k) / _SCALAR_NORMS[k] for k in _SCALAR_KEYS], dim=1)
            bsmap = _sl("bs_radio_obs").reshape(-1, 1, 256, 256).float() / 200.0

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)

        return nf, adj, msk, sc, bsmap

    def _build_adj(self, adj_1d):
        B, N = adj_1d.shape
        adj  = torch.zeros(B, N, N, device=adj_1d.device)
        adj[:, 0, :] = adj_1d
        adj[:, :, 0] = adj_1d
        eye  = torch.eye(N, device=adj_1d.device).unsqueeze(0)
        return torch.clamp(adj + eye, max=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nf, adj_1d, msk, scalars, bsmap = self._parse(obs)
        B = nf.shape[0]

        # ── CNN embedding ─────────────────────────────────────────────────
        cnn_embed = self.map_cnn(bsmap)              # (B, 64)

        # ── MLP branch ────────────────────────────────────────────────────
        mlp_in      = torch.cat([scalars, cnn_embed], dim=1)
        mlp_hidden  = self.mlp(mlp_in)
        move_logits = self.move_head(mlp_hidden)     # (B, move_dirs)

        # ── GAT branch ────────────────────────────────────────────────────
        adj  = self._build_adj(adj_1d)
        h    = self.elu(self.gat1(nf,  adj))         # (B, N, 128)
        h    = self.elu(self.gat2(h,   adj))         # (B, N, 64)
        focal_embed = h[:, 0, :]                     # (B, 64)

        # ── Lozano comm logits ────────────────────────────────────────────
        non_self_embeds = h[:, 1:, :]
        node_scores = self.comm_score(non_self_embeds).squeeze(-1)
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
