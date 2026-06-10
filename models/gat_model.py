"""
Graph Attention Network policy model for RLlib (old API stack).

Implements the GAT-MARL architecture from:
  Lozano-Cuadra et al., "Learning Decentralized Routing Policies via
  Graph Attention-based Multi-Agent Reinforcement Learning in Lunar
  Delay-Tolerant Networks", arXiv:2510.20436

Architecture:
  - GAT layer 1 : in_feats=7  → 8 heads × 64  = 512 (concat)
  - GAT layer 2 : in_feats=512 → 1 head  × 64  =  64 (mean)
  - Policy head  : self-node embedding + scalar context → num_outputs
  - Value head   : mean-pooled graph embedding → 1  (centralized critic)

Node features (NODE_FEAT_DIM = 7) per node in the local 1-hop subgraph:
  [is_self, is_lander, bs_connected, buffer_fill, dist_to_lander_norm, x_norm, y_norm]

Observation keys consumed (others are ignored):
  action_mask, graph_adj, graph_node_features, goal_vector, move_history, position
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

NODE_FEAT_DIM = 7

_SCALAR_KEYS = ["goal_vector", "move_history", "position"]
_SCALAR_NORMS = {"goal_vector": 256.0, "move_history": 256.0, "position": 256.0}


# ---------------------------------------------------------------------------
# GAT layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Batched multi-head graph attention (Veličković et al., 2018).

    Attention scores use the decomposed form:
        e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
    which avoids the O(N²·F) memory of concatenating all pairs.
    """

    def __init__(self, in_feats: int, out_feats: int, num_heads: int,
                 concat: bool = True, dropout: float = 0.2):
        super().__init__()
        self.H = num_heads
        self.F = out_feats
        self.concat = concat

        self.W      = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.a_src  = nn.Parameter(torch.empty(num_heads, out_feats))
        self.a_dst  = nn.Parameter(torch.empty(num_heads, out_feats))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky  = nn.LeakyReLU(0.2)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N, in_feats)
        adj : (B, N, N)  — 1 where edge exists (including self-loops), 0 elsewhere
        Returns (B, N, H*F) if concat else (B, N, F)
        """
        B, N, _ = x.shape
        H_dim, F_dim = self.H, self.F

        Wh = self.W(x).view(B, N, H_dim, F_dim)   # (B, N, H, F)

        # e_src[b,i,h] = a_src[h] · Wh[b,i,h]
        e_src = (Wh * self.a_src).sum(-1)          # (B, N, H)
        e_dst = (Wh * self.a_dst).sum(-1)          # (B, N, H)

        # broadcast: e[b,i,j,h] = LeakyReLU(e_src[b,i,h] + e_dst[b,j,h])
        e = self.leaky(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1)   # (B, N, N, H)
        )

        # mask non-edges → -inf so softmax ignores them
        mask  = (adj == 0).unsqueeze(-1)               # (B, N, N, 1)
        e     = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=2)                    # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)       # isolated nodes → 0
        alpha = self.drop(alpha)

        # aggregate: out[b,i] = Σ_j alpha[b,i,j] * Wh[b,j]
        # rearrange to (B, H, N, N) @ (B, H, N, F) → (B, H, N, F)
        alpha_t = alpha.permute(0, 3, 1, 2)            # (B, H, N, N)
        Wh_t    = Wh.permute(0, 2, 1, 3)              # (B, H, N, F)
        out     = torch.matmul(alpha_t, Wh_t)          # (B, H, N, F)
        out     = out.permute(0, 2, 1, 3)              # (B, N, H, F)

        if self.concat:
            return out.reshape(B, N, H_dim * F_dim)
        return out.mean(dim=2)                         # (B, N, F)


# ---------------------------------------------------------------------------
# RLlib model
# ---------------------------------------------------------------------------

class TorchGATModel(TorchModelV2, nn.Module):
    """
    GAT-MARL model for RLlib's old API stack (TorchModelV2).

    Observing agent is always placed at node index 0.
    Policy  : self-node (index 0) embedding + scalar context → logits
    Value   : mean-pooled graph embedding (centralized critic)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size = int(np.prod(orig.spaces["action_mask"].shape))
        self.max_nodes = int(np.prod(orig.spaces["graph_adj"].shape))

        # GAT: 7 → (8×64=512) → 64
        self.gat1 = GATLayer(NODE_FEAT_DIM, 64, num_heads=8, concat=True,  dropout=0.2)
        self.gat2 = GATLayer(512,            64, num_heads=1, concat=False, dropout=0.2)
        self.elu  = nn.ELU()

        # Scalar context fed alongside self-node embedding
        scalar_dim = sum(
            int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS
        )

        # Policy head (decentralized: self-node only)
        self.policy_head = nn.Sequential(
            nn.Linear(64 + scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )

        # Value head (centralized: all nodes)
        self.value_head = nn.Sequential(
            nn.Linear(64 * self.max_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._value = None
        self._flat_offsets = self._build_offsets(orig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_offsets(self, orig_space):
        offsets = {}
        cursor = 0
        for k in sorted(orig_space.spaces):
            dim = int(np.prod(orig_space.spaces[k].shape))
            offsets[k] = (cursor, cursor + dim)
            cursor += dim
        return offsets

    def _parse(self, obs):
        """Return (node_feats, adj_1d, action_mask, scalars) as float tensors."""
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
            nf  = _sl("graph_node_features").view(-1, self.max_nodes, NODE_FEAT_DIM)
            adj = _sl("graph_adj")
            msk = _sl("action_mask")
            sc  = torch.cat([_sl(k) / _SCALAR_NORMS[k] for k in _SCALAR_KEYS], dim=1)

        # ensure (B, N, F) shape
        if nf.dim() == 2:
            nf = nf.unsqueeze(0)

        return nf, adj, msk, sc

    def _build_adj(self, adj_1d: torch.Tensor) -> torch.Tensor:
        """
        Build full N×N adjacency from the observing agent's 1-hop view.
        Node 0 is always self. Star topology: self ↔ each connected node.
        Self-loops added for all nodes.
        """
        B, N   = adj_1d.shape
        adj    = torch.zeros(B, N, N, device=adj_1d.device)
        adj[:, 0, :] = adj_1d           # self → neighbors
        adj[:, :, 0] = adj_1d           # neighbors → self (symmetric)
        eye    = torch.eye(N, device=adj_1d.device).unsqueeze(0)
        return torch.clamp(adj + eye, max=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nf, adj_1d, msk, scalars = self._parse(obs)

        adj = self._build_adj(adj_1d)           # (B, N, N)

        h = self.elu(self.gat1(nf, adj))        # (B, N, 512)
        h = self.elu(self.gat2(h,  adj))        # (B, N, 64)

        # Value: mean-pool → flatten → MLP
        self._value = self.value_head(
            h.reshape(h.shape[0], -1)           # (B, N*64)
        ).squeeze(1)

        # Policy: self-node (index 0) + scalar context
        self_emb = h[:, 0, :]                   # (B, 64)
        logits   = self.policy_head(torch.cat([self_emb, scalars], dim=1))

        # Apply action mask
        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._value