"""
Hybrid MLP-GAT policy model for RLlib (old API stack).

Movement decisions (9 logits) come from an MLP over scalar features.
Communication decisions (6 logits = 3 targets × 2) come from a GAT over
the local agent-BS graph.  Both branches share a value head.

Architecture:
  MLP branch  : scalars(12) → Linear(256)+ReLU → Linear(256)+ReLU → move_logits(9)
  GAT branch  : node_features(N,7) + adj(N,N) →
                  GATLayer(7→128, heads=4, concat) + ELU →
                  GATLayer(128→64, heads=1, mean)  + ELU →
                  per_node_comm_head(64→2) applied to nodes[1:] →
                  comm_logits(6)          [node 0 = self, excluded from output]
  Value head  : cat(mlp_hidden(256), focal_embed(64)) → Linear(320,1)
  Output      : cat(move_logits(9), comm_logits(6)) = 15 logits + action mask

Comm logit ordering matches action space (self excluded, others sorted by ue_id then BS):
  comm_logits[0:2]  → other_rover_0 (graph node 1)
  comm_logits[2:4]  → other_rover_1 (graph node 2)
  comm_logits[4:6]  → base station  (graph node 3)

Flat obs layout (alphabetical key order, total 59 dims):
  [0:15]  action_mask                (15,)
  [15:17] goal_vector                (2,)
  [17:21] graph_adj                  (4,)
  [21:49] graph_node_features        (4×7 = 28,)
  [49:53] move_history               (4,)
  [53:57] other_agent_connectivity   (4,)
  [57:59] position                   (2,)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

NODE_FEAT_DIM = 7
MAX_NODES     = 4   # 3 rovers + 1 BS (for default num_agents=3)

_SCALAR_KEYS  = ["goal_vector", "move_history", "other_agent_connectivity", "position"]
_SCALAR_NORMS = {
    "goal_vector":              256.0,
    "move_history":             256.0,
    "other_agent_connectivity": 1.0,
    "position":                 256.0,
}


# ---------------------------------------------------------------------------
# GAT layer (Veličković et al., 2018 — decomposed attention form)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Batched multi-head graph attention layer.

    Attention scores:  e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
    This decomposed form avoids O(N²·F) memory from explicit pair-concatenation.
    """

    def __init__(self, in_feats: int, out_feats: int, num_heads: int,
                 concat: bool = True, dropout: float = 0.2):
        super().__init__()
        self.H      = num_heads
        self.F      = out_feats
        self.concat = concat

        self.W     = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_feats))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_feats))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N, in_feats)
        adj : (B, N, N)  — 1 where edge exists (including self-loops), else 0
        Returns (B, N, H*F) if concat=True else (B, N, F)
        """
        B, N, _ = x.shape
        H, Fd = self.H, self.F  # Fd avoids shadowing torch.nn.functional F

        Wh    = self.W(x).view(B, N, H, Fd)        # (B, N, H, F)
        e_src = (Wh * self.a_src).sum(-1)          # (B, N, H)
        e_dst = (Wh * self.a_dst).sum(-1)          # (B, N, H)

        # e[b,i,j,h] = LeakyReLU(e_src[b,i,h] + e_dst[b,j,h])
        e     = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # (B, N, N, H)

        mask  = (adj == 0).unsqueeze(-1)            # (B, N, N, 1)
        e     = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=2)                 # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)    # isolated nodes → 0 weight
        alpha = self.drop(alpha)

        # out[b,i,h] = Σ_j alpha[b,i,j,h] · Wh[b,j,h,:]
        alpha_t = alpha.permute(0, 3, 1, 2)         # (B, H, N, N)
        Wh_t    = Wh.permute(0, 2, 1, 3)           # (B, H, N, F)
        out     = torch.matmul(alpha_t, Wh_t)       # (B, H, N, F)
        out     = out.permute(0, 2, 1, 3)           # (B, N, H, F)

        if self.concat:
            return out.reshape(B, N, H * Fd)
        return out.mean(dim=2)                      # (B, N, Fd)


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------

class TorchMLPGATModel(TorchModelV2, nn.Module):
    """
    Hybrid MLP-GAT policy for RLlib's old API stack.

    Observing agent is always placed at graph node index 0.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size    = int(np.prod(orig.spaces["action_mask"].shape))
        self.max_nodes    = int(np.prod(orig.spaces["graph_adj"].shape))
        self.node_feat_dim = NODE_FEAT_DIM

        scalar_dim = sum(
            int(np.prod(orig.spaces[k].shape)) for k in _SCALAR_KEYS
        )  # = 8  (goal_vector:2 + move_history:4 + position:2)

        # ── MLP branch ───────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.move_head = nn.Linear(256, 9)

        # ── GAT branch ───────────────────────────────────────────────────
        self.gat1 = GATLayer(NODE_FEAT_DIM, 32, num_heads=4, concat=True,  dropout=0.2)
        self.gat2 = GATLayer(128,            64, num_heads=1, concat=False, dropout=0.2)
        self.elu  = nn.ELU()
        # Shared binary head applied independently to each non-self node embedding.
        # Logit ordering: [other_0, other_1, BS] × 2 = 6 comm logits.
        self.comm_head = nn.Linear(64, 2)

        # ── Value head ───────────────────────────────────────────────────
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
            dim           = int(np.prod(orig_space.spaces[k].shape))
            offsets[k]    = (cursor, cursor + dim)
            cursor       += dim
        return offsets

    def _parse(self, obs):
        """Return (node_feats, adj_1d, action_mask, mlp_scalars) as float tensors."""
        if isinstance(obs, dict):
            nf  = obs["graph_node_features"].float()    # (B, N, 7) or (N, 7)
            adj = obs["graph_adj"].float()              # (B, N) or (N,)
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
                -1, self.max_nodes, self.node_feat_dim   # .reshape not .view: may be non-contiguous
            )
            sc  = torch.cat([_sl(k) / _SCALAR_NORMS[k] for k in _SCALAR_KEYS], dim=1)

        if nf.dim() == 2:                               # single-sample without batch dim
            nf = nf.unsqueeze(0)

        return nf, adj, msk, sc

    def _build_adj(self, adj_1d: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full N×N adjacency from focal agent's 1-hop reachability vector.
        Star topology: self ↔ each reachable node.  Self-loops added for all nodes.

        adj_1d : (B, N)  — 1 if focal can reach node j, else 0 (adj_1d[:,0]=1 always)
        Returns : (B, N, N)
        """
        B, N = adj_1d.shape
        adj  = torch.zeros(B, N, N, device=adj_1d.device)
        adj[:, 0, :] = adj_1d           # focal → neighbors
        adj[:, :, 0] = adj_1d           # neighbors → focal (symmetric)
        eye  = torch.eye(N, device=adj_1d.device).unsqueeze(0)
        return torch.clamp(adj + eye, max=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nf, adj_1d, msk, scalars = self._parse(obs)

        # ── MLP branch: scalar features → movement logits ─────────────
        mlp_hidden  = self.mlp(scalars)             # (B, 256)
        move_logits = self.move_head(mlp_hidden)    # (B, 9)

        # ── GAT branch: graph → per-node embeddings → comm logits ────
        adj  = self._build_adj(adj_1d)              # (B, N, N)
        h    = self.elu(self.gat1(nf,  adj))        # (B, N, 128)
        h    = self.elu(self.gat2(h,   adj))        # (B, N, 64)
        focal_embed = h[:, 0, :]                    # (B, 64) — self is always node 0
        # Apply shared comm head to each non-self node; logit order matches action space.
        comm_logits = self.comm_head(h[:, 1:, :]).reshape(h.shape[0], -1)  # (B, (N-1)*2)

        # ── Value head: fuse both branches ────────────────────────────
        self._current_value = self.value_head(
            torch.cat([mlp_hidden, focal_embed], dim=1)   # (B, 320)
        ).squeeze(1)

        # ── Combine and mask ──────────────────────────────────────────
        logits   = torch.cat([move_logits, comm_logits], dim=1)  # (B, 17)
        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value
