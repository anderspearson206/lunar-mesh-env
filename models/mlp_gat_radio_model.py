"""
MLP-GAT policy with configurable radio-map encoding (old RLlib API stack).

Set ``custom_model_config["radio_mode"]`` to one of:

  "rssi"  – 4 RSSI scalars fed into the MLP branch (scalar_dim 12→16).
             Context dim 320; head sizes unchanged from mlp_gat_model.
  "crop"  – 64×64 radio-map crop → small CNN → 64-dim embed.
             Context dim 384 = 256 (mlp) + 64 (gat) + 64 (cnn).
  "full"  – 256×256 radio map → deep CNN → 128-dim embed.
             Context dim 448 = 256 + 64 + 128.

In all modes move, comm, and value heads share the same fused context.

Flat obs layout (alphabetical, for each mode):
  rssi  65  dims: base(61) + radio_rssi(4)
  crop 4157  dims: base(61) + radio_crop(1×64×64=4096)
  full 65597 dims: base(61) + radio_map(1×256×256=65536)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.mlp_gat_model import GATLayer  # reuse identical GAT layer

NODE_FEAT_DIM = 7
MAX_NODES     = 4

_BASE_SCALAR_KEYS  = ["goal_vector", "move_history", "other_agent_connectivity", "position"]
_BASE_SCALAR_NORMS = {
    "goal_vector":              256.0,
    "move_history":             256.0,
    "other_agent_connectivity": 1.0,
    "position":                 256.0,
}

_CROP_HW = 64
_FULL_HW = 256


def _crop_cnn() -> nn.Sequential:
    # 1×64×64 → (16,31,31) → (32,15,15) → (32,7,7) → 1568 → 64
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2), nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2), nn.ELU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
    )


def _full_cnn() -> nn.Sequential:
    # 1×256×256 → (16,84,84) → (32,41,41) → (64,20,20) → (64,9,9) → 5184 → 128
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=3), nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ELU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ELU(),
        nn.Flatten(),
        nn.Linear(64 * 9 * 9, 128),
    )


class TorchMLPGATRadioModel(TorchModelV2, nn.Module):
    """MLP-GAT policy with an optional radio-map encoding branch."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        ccfg            = model_config.get("custom_model_config", {})
        self.radio_mode = ccfg.get("radio_mode", "rssi")
        assert self.radio_mode in ("rssi", "crop", "full"), \
            f"radio_mode must be 'rssi', 'crop', or 'full', got {self.radio_mode!r}"

        orig = getattr(obs_space, "original_space", obs_space)
        self.mask_size     = int(np.prod(orig.spaces["action_mask"].shape))
        self.max_nodes     = int(np.prod(orig.spaces["graph_adj"].shape))
        self.node_feat_dim = NODE_FEAT_DIM

        if self.radio_mode == "rssi":
            self._scalar_keys  = _BASE_SCALAR_KEYS + ["radio_rssi"]
            self._scalar_norms = dict(_BASE_SCALAR_NORMS, radio_rssi=1.0)
        else:
            self._scalar_keys  = list(_BASE_SCALAR_KEYS)
            self._scalar_norms = dict(_BASE_SCALAR_NORMS)

        scalar_dim = sum(int(np.prod(orig.spaces[k].shape)) for k in self._scalar_keys)

        # ── Radio CNN (crop / full only) ─────────────────────────────────
        if self.radio_mode == "crop":
            self.radio_cnn = _crop_cnn()
            radio_emb_dim  = 64
        elif self.radio_mode == "full":
            self.radio_cnn = _full_cnn()
            radio_emb_dim  = 128
        else:
            self.radio_cnn = None
            radio_emb_dim  = 0

        # ── MLP branch ───────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )

        # ── GAT branch ───────────────────────────────────────────────────
        self.gat1 = GATLayer(NODE_FEAT_DIM, 32, num_heads=4, concat=True,  dropout=0.2)
        self.gat2 = GATLayer(128,            64, num_heads=1, concat=False, dropout=0.2)
        self.elu  = nn.ELU()

        # ── Fused context heads ──────────────────────────────────────────
        context_dim = 256 + 64 + radio_emb_dim
        self.move_head  = nn.Linear(context_dim, 9)
        self.comm_head  = nn.Linear(context_dim, 8)
        self.value_head = nn.Linear(context_dim, 1)

        self._current_value = None
        self._flat_offsets  = self._build_offsets(orig)

    # ------------------------------------------------------------------
    def _build_offsets(self, orig_space):
        offsets, cursor = {}, 0
        for k in sorted(orig_space.spaces):
            dim        = int(np.prod(orig_space.spaces[k].shape))
            offsets[k] = (cursor, cursor + dim)
            cursor    += dim
        return offsets

    def _parse(self, obs):
        """Return (nf, adj_1d, mask, scalars, radio_img_or_None)."""
        if isinstance(obs, dict):
            nf  = obs["graph_node_features"].float()
            adj = obs["graph_adj"].float()
            msk = obs["action_mask"].float()
            sc  = torch.cat([
                obs[k].float().reshape(obs[k].shape[0], -1) / self._scalar_norms[k]
                for k in self._scalar_keys
            ], dim=1)
            if self.radio_mode == "crop":
                radio = obs["radio_crop"].float()
                if radio.dim() == 3:
                    radio = radio.unsqueeze(0)
            elif self.radio_mode == "full":
                radio = obs["radio_map"].float()
                if radio.dim() == 3:
                    radio = radio.unsqueeze(0)
            else:
                radio = None

        else:
            def _sl(key):
                s, e = self._flat_offsets[key]
                return obs[:, s:e].float()

            msk = _sl("action_mask")
            adj = _sl("graph_adj")
            nf  = _sl("graph_node_features").reshape(-1, self.max_nodes, self.node_feat_dim)
            sc  = torch.cat([_sl(k) / self._scalar_norms[k] for k in self._scalar_keys], dim=1)
            if self.radio_mode == "crop":
                radio = _sl("radio_crop").reshape(-1, 1, _CROP_HW, _CROP_HW)
            elif self.radio_mode == "full":
                radio = _sl("radio_map").reshape(-1, 1, _FULL_HW, _FULL_HW)
            else:
                radio = None

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)

        return nf, adj, msk, sc, radio

    def _build_adj(self, adj_1d: torch.Tensor) -> torch.Tensor:
        B, N = adj_1d.shape
        adj  = torch.zeros(B, N, N, device=adj_1d.device)
        adj[:, 0, :] = adj_1d
        adj[:, :, 0] = adj_1d
        return torch.clamp(adj + torch.eye(N, device=adj_1d.device).unsqueeze(0), max=1.0)

    # ------------------------------------------------------------------
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        nf, adj_1d, msk, scalars, radio = self._parse(obs)

        mlp_hidden  = self.mlp(scalars)              # (B, 256)

        adj         = self._build_adj(adj_1d)        # (B, N, N)
        h           = self.elu(self.gat1(nf,  adj))  # (B, N, 128)
        h           = self.elu(self.gat2(h,   adj))  # (B, N, 64)
        focal_embed = h[:, 0, :]                     # (B, 64)

        context = torch.cat([mlp_hidden, focal_embed], dim=1)  # (B, 320)
        if radio is not None:
            context = torch.cat([context, self.elu(self.radio_cnn(radio))], dim=1)

        move_logits = self.move_head(context)
        comm_logits = self.comm_head(context)
        self._current_value = self.value_head(context).squeeze(1)

        logits   = torch.cat([move_logits, comm_logits], dim=1)
        inf_mask = torch.clamp(torch.log(msk), min=-1e9)
        return logits + inf_mask, state

    def value_function(self):
        return self._current_value
