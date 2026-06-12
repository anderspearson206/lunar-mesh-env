# alphastar_net.py
"""
AlphaStar-style policy network for LunarRoverMeshEnv (pure PyTorch).

No RLlib imports: the same module is trained standalone during supervised
pretraining (examples/train_sl_policy.py) and embedded in the RLlib ModelV2
wrapper (models/alphastar_model.py) for PPO fine-tuning.

Structure (miniaturized AlphaStar):
    screen (3,64,64) --SpatialEncoder--> 256
    minimap (7,64,64) --SpatialEncoder--> 256          (scatter planes inside)
    scalars (8,)      --MLP-->            64
    concat(576) --core (Linear + residual block)--> context (256)
    context --> value head (1)
    context --> autoregressive heads: move(9), then comm heads (2 each)
                conditioned on the move embedding and previous comm choices.
"""

import torch
import torch.nn as nn

CONTEXT_DIM = 256
SCALAR_DIM = 8                    # position(2) + goal_vector(2) + move_history(4)
MOVE_EMBED_DIM = 16
HEAD_SIZES = [9, 2, 2, 2, 2]      # move, comm_rover0..2, comm_bs
NUM_HEADS = len(HEAD_SIZES)
NUM_COMM_HEADS = NUM_HEADS - 1
MASK_DIM = sum(HEAD_SIZES)        # 17
# (start, end) slices of the flat action mask per head, matching
# marl_env._get_obs: [move(9) | rover0(2) | rover1(2) | rover2(2) | bs(2)]
MASK_SLICES = []
_off = 0
for _h in HEAD_SIZES:
    MASK_SLICES.append((_off, _off + _h))
    _off += _h


class SpatialEncoder(nn.Module):
    """(B, C, 64, 64) -> (B, 256)"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),           # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),           # 4x4
            nn.ReLU(),
            nn.Flatten(),                                                    # 1024
            nn.Linear(64 * 4 * 4, CONTEXT_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class AlphaStarPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.screen_encoder = SpatialEncoder(3)
        self.minimap_encoder = SpatialEncoder(7)
        self.scalar_mlp = nn.Sequential(nn.Linear(SCALAR_DIM, 64), nn.ReLU())

        fused = CONTEXT_DIM + CONTEXT_DIM + 64
        self.core_in = nn.Sequential(nn.Linear(fused, CONTEXT_DIM), nn.ReLU())
        self.core_res = nn.Sequential(
            nn.Linear(CONTEXT_DIM, CONTEXT_DIM), nn.ReLU(),
            nn.Linear(CONTEXT_DIM, CONTEXT_DIM),
        )

        self.value_head = nn.Linear(CONTEXT_DIM, 1)
        self.move_head = nn.Linear(CONTEXT_DIM, HEAD_SIZES[0])
        self.move_embed = nn.Embedding(HEAD_SIZES[0], MOVE_EMBED_DIM)
        # comm trunk input: context + move embedding + previous comm bits
        # (zero-padded to NUM_COMM_HEADS so all comm heads share one trunk)
        self.comm_trunk = nn.Sequential(
            nn.Linear(CONTEXT_DIM + MOVE_EMBED_DIM + NUM_COMM_HEADS, 64),
            nn.ReLU(),
        )
        self.comm_heads = nn.ModuleList(
            [nn.Linear(64, h) for h in HEAD_SIZES[1:]]
        )

    # ------------------------------------------------------------------

    def encode(self, screen, minimap, scalars):
        """-> context (B, 256)"""
        fused = torch.cat([
            self.screen_encoder(screen),
            self.minimap_encoder(minimap),
            self.scalar_mlp(scalars),
        ], dim=1)
        ctx = self.core_in(fused)
        return torch.relu(ctx + self.core_res(ctx))

    def value(self, context):
        """-> (B,)"""
        return self.value_head(context).squeeze(1)

    def head_logits(self, context, head_idx, prefix_actions=None):
        """Logits for one head given the chosen prefix actions.

        prefix_actions: (B, head_idx) long tensor of earlier head choices
        (unused for head 0).
        """
        if head_idx == 0:
            return self.move_head(context)
        move = self.move_embed(prefix_actions[:, 0].long())
        prev_comm = torch.zeros(
            context.shape[0], NUM_COMM_HEADS,
            dtype=context.dtype, device=context.device,
        )
        n_prev = head_idx - 1
        if n_prev > 0:
            prev_comm[:, :n_prev] = prefix_actions[:, 1:head_idx].float()
        h = self.comm_trunk(torch.cat([context, move, prev_comm], dim=1))
        return self.comm_heads[head_idx - 1](h)

    def all_head_logits(self, context, actions):
        """Teacher-forced logits for all heads given full actions (B, 5).

        Returns a list of NUM_HEADS tensors [(B, 9), (B, 2) x 4].
        """
        actions = actions.long()
        out = [self.move_head(context)]
        move = self.move_embed(actions[:, 0])
        for i in range(1, NUM_HEADS):
            prev_comm = torch.zeros(
                context.shape[0], NUM_COMM_HEADS,
                dtype=context.dtype, device=context.device,
            )
            if i > 1:
                prev_comm[:, :i - 1] = actions[:, 1:i].float()
            h = self.comm_trunk(torch.cat([context, move, prev_comm], dim=1))
            out.append(self.comm_heads[i - 1](h))
        return out


def masked_logits(logits, mask_slice):
    """Apply an action mask slice (B, head_size) to head logits."""
    return logits + torch.clamp(torch.log(mask_slice.float()), min=-1e9)
