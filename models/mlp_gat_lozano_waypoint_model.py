"""
MLP-GAT Lozano model for waypoint-residual navigation.

Identical architecture to TorchMLPGATLozanoModel but with extended scalar input:
  - bs_signal       (1,)   normalised dBm at current position
  - waypoint_signals (16,) normalised dBm at each of the 16 codebook waypoints
  - decision_due    (1,)   binary: 1 when nav action is live this step

Scalar dim: existing 13 + 18 new = 31 total → MLP(31→256→256) → move_head(256→N_NAV).

move_dirs is read from custom_model_config["move_dirs"]; default 17 (N_NAV).
"""

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from models.mlp_gat_model import GATLayer

NODE_FEAT_DIM = 7

# Scalar keys processed by the MLP branch (must match obs dict keys exactly)
_SCALAR_KEYS = [
    "bs_signal",
    "buffer_usage",
    "decision_due",
    "goal_vector",
    "move_history",
    "other_agent_connectivity",
    "position",
    "waypoint_signals",
]
_SCALAR_NORMS = {
    "bs_signal":               1.0,
    "buffer_usage":            1.0,
    "decision_due":            1.0,
    "goal_vector":             256.0,
    "move_history":            256.0,
    "other_agent_connectivity": 1.0,
    "position":                256.0,
    "waypoint_signals":        1.0,
}


class TorchMLPGATLozanoWaypointModel(TorchModelV2, nn.Module):
    """
    Hybrid MLP-GAT policy for waypoint-residual Lozano navigation.

    Navigation head: MLP(31) → Linear(256) → ReLU → Linear(256) → ReLU
                     → move_head(256, N_NAV)   (N_NAV = 17 by default)
    Comm head:       GAT → comm_score per non-self node + hold_logit
    Value head:      Linear(cat(mlp_hidden, focal_embed)) → scalar
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size     = int(np.prod(orig.spaces["action_mask"].shape))
        self.max_nodes     = int(np.prod(orig.spaces["graph_adj"].shape))
        self.node_feat_dim = NODE_FEAT_DIM

        scalar_dim = sum(
            int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS
        )

        move_dirs = model_config.get("custom_model_config", {}).get("move_dirs", 17)

        # ── MLP branch ────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
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
        """Return (node_feats, adj_1d, action_mask, mlp_scalars) as float tensors."""
        if isinstance(obs, dict):
            nf  = obs["graph_node_features"].float()
            adj = obs["graph_adj"].float()
            msk = obs["action_mask"].float()
            sc  = torch.cat([
                obs[k].float().reshape(obs[k].shape[0], -1) / _SCALAR_NORMS[k]
                for k in _SCALAR_KEYS
            ], dim=1)
        else:
            def _sl(key):
                s, e = self._flat_offsets[key]
                return obs[:, s:e].float()

            msk = _sl("action_mask")
            adj = _sl("graph_adj")
            nf  = _sl("graph_node_features").reshape(
                -1, self.max_nodes, self.node_feat_dim
            )
            sc  = torch.cat([_sl(k) / _SCALAR_NORMS[k] for k in _SCALAR_KEYS], dim=1)

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)

        return nf, adj, msk, sc

    def _build_adj(self, adj_1d: torch.Tensor) -> torch.Tensor:
        B, N = adj_1d.shape
        adj  = torch.zeros(B, N, N, device=adj_1d.device)
        adj[:, 0, :] = adj_1d
        adj[:, :, 0] = adj_1d
        eye  = torch.eye(N, device=adj_1d.device).unsqueeze(0)
        return torch.clamp(adj + eye, max=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nf, adj_1d, msk, scalars = self._parse(obs)
        B = nf.shape[0]

        # ── MLP branch ───────────────────────────────────────────────────
        mlp_hidden  = self.mlp(scalars)
        move_logits = self.move_head(mlp_hidden)        # (B, move_dirs)

        # ── GAT branch ───────────────────────────────────────────────────
        adj = self._build_adj(adj_1d)                   # (B, N, N)
        h   = self.elu(self.gat1(nf,  adj))             # (B, N, 128)
        h   = self.elu(self.gat2(h,   adj))             # (B, N, 64)
        focal_embed = h[:, 0, :]                        # (B, 64)

        # ── Lozano comm logits ───────────────────────────────────────────
        non_self_embeds = h[:, 1:, :]
        node_scores = self.comm_score(non_self_embeds).squeeze(-1)  # (B, N-1)
        hold_scores = self.hold_logit.expand(B, 1)
        comm_logits = torch.cat([hold_scores, node_scores], dim=1)  # (B, N)

        avail = torch.cat(
            [torch.ones(B, 1, device=adj_1d.device), adj_1d[:, 1:]], dim=1
        )
        comm_logits = comm_logits.masked_fill(avail == 0, float("-inf"))

        # ── Value head ───────────────────────────────────────────────────
        self._current_value = self.value_head(
            torch.cat([mlp_hidden, focal_embed], dim=1)
        ).squeeze(1)

        # ── Combine and apply obs action mask ────────────────────────────
        if self.num_outputs == self.max_nodes:
            logits = comm_logits                        # DQN discrete mode
        else:
            logits = torch.cat([move_logits, comm_logits], dim=1)

        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value