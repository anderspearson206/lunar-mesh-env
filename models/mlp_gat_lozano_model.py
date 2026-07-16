"""
Lozano-style hybrid MLP-GAT policy model for RLlib (old API stack).

Movement is handled by A*; the policy only controls the comm edge selection.
The movement slot in action space is MultiDiscrete([1, MAX_NODES]) — the single
movement option (idle) is always masked valid, and A* overrides it in step().

Architecture:
  MLP branch  : scalars(12) → Linear(256)+ReLU → Linear(256)+ReLU → idle_logit(1)
  GAT branch  : node_features(N,7) + adj →
                  GATLayer(7→128, heads=4, concat) + ELU →
                  GATLayer(128→64, heads=1, mean)  + ELU →
                  comm_score(64→1) per non-self node + hold_logit param →
                  comm_logits(MAX_NODES=4)  with adj masking
  Value head  : cat(mlp_hidden(256), focal_embed(64)) → Linear(320,1)
  Output      : cat(idle_logit(1), comm_logits(4)) = 5 logits

Comm logit ordering (action[1]):
  comm_logits[0]  → hold
  comm_logits[1]  → other_rover_0 (graph node 1, sorted by ue_id)
  comm_logits[2]  → other_rover_1 (graph node 2)
  comm_logits[3]  → base station  (graph node 3)

Flat obs layout (alphabetical key order, total 50 dims):
  [0:5]   action_mask                (5,)
  [5:6]   buffer_usage               (1,)
  [6:8]   goal_vector                (2,)
  [8:12]  graph_adj                  (4,)
  [12:40] graph_node_features        (4×7 = 28,)
  [40:44] move_history               (4,)
  [44:48] other_agent_connectivity   (4,)
  [48:50] position                   (2,)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.mlp_gat_model import GATLayer   # reuse identical GAT implementation

NODE_FEAT_DIM = 7

_SCALAR_KEYS  = ["buffer_usage", "goal_vector", "move_history", "other_agent_connectivity", "position"]
_SCALAR_NORMS = {
    "buffer_usage":             1.0,
    "goal_vector":              256.0,
    "move_history":             256.0,
    "other_agent_connectivity": 1.0,
    "position":                 256.0,
}


class TorchMLPGATLozanoModel(TorchModelV2, nn.Module):
    """
    Hybrid MLP-GAT policy for Lozano single-edge comm action space.

    Observing agent is always placed at graph node index 0.
    Comm head scores each non-self node; hold is an additional learnable scalar.
    Unreachable nodes are masked to -inf before softmax.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size     = int(np.prod(orig.spaces["action_mask"].shape))   # 13
        self.max_nodes     = int(np.prod(orig.spaces["graph_adj"].shape))      # 4
        self.node_feat_dim = NODE_FEAT_DIM

        scalar_dim = sum(
            int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS
        )  # 1+2+4+4+2 = 13

        # move_dirs=1  → A* nav (idle-only slot, A* overrides)
        # move_dirs=9  → RL nav (full 9-direction head)
        move_dirs = model_config.get("custom_model_config", {}).get("move_dirs", 1)

        # ── MLP branch ────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.move_head = nn.Linear(256, move_dirs)

        # ── GAT branch (unchanged) ────────────────────────────────────────
        self.gat1 = GATLayer(NODE_FEAT_DIM, 32, num_heads=4, concat=True,  dropout=0.2)
        self.gat2 = GATLayer(128,            64, num_heads=1, concat=False, dropout=0.2)
        self.elu  = nn.ELU()

        # ── Lozano comm head ──────────────────────────────────────────────
        # Scores the attractiveness of forwarding to each non-self node.
        self.comm_score = nn.Linear(64, 1)
        # Learnable logit for the "hold" action (index 0 in edge selection).
        self.hold_logit = nn.Parameter(torch.zeros(1))

        # ── Value head (unchanged) ────────────────────────────────────────
        self.value_head = nn.Linear(256 + 64, 1)

        self._current_value = None
        self._flat_offsets  = self._build_offsets(orig)

    # ------------------------------------------------------------------
    # Helpers (same as base)
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
        move_logits = self.move_head(mlp_hidden)     # (B, 9)

        # ── GAT branch ───────────────────────────────────────────────────
        adj  = self._build_adj(adj_1d)               # (B, N, N)
        h    = self.elu(self.gat1(nf,  adj))         # (B, N, 128)
        h    = self.elu(self.gat2(h,   adj))         # (B, N, 64)
        focal_embed = h[:, 0, :]                     # (B, 64)

        # ── Lozano comm logits ───────────────────────────────────────────
        non_self_embeds = h[:, 1:, :]                # (B, N-1, 64)
        node_scores = self.comm_score(non_self_embeds).squeeze(-1)  # (B, N-1)
        hold_scores = self.hold_logit.expand(B, 1)   # (B, 1)
        comm_logits = torch.cat([hold_scores, node_scores], dim=1)  # (B, N)

        # Mask unavailable targets: adj_1d[:,1:] = 0 if unreachable, hold always reachable
        avail = torch.cat(
            [torch.ones(B, 1, device=adj_1d.device), adj_1d[:, 1:]], dim=1
        )  # (B, N)
        comm_logits = comm_logits.masked_fill(avail == 0, float("-inf"))

        # ── Value head ───────────────────────────────────────────────────
        self._current_value = self.value_head(
            torch.cat([mlp_hidden, focal_embed], dim=1)
        ).squeeze(1)

        # ── Combine and apply obs action mask ────────────────────────────
        # num_outputs == max_nodes → Discrete mode (DQN): comm logits only
        # num_outputs == 1+max_nodes → MultiDiscrete mode (PPO): prepend move logit
        if self.num_outputs == self.max_nodes:
            logits = comm_logits                                   # (B, 4)
        else:
            logits = torch.cat([move_logits, comm_logits], dim=1)  # (B, 5)
        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value
