"""
Supervised pretraining of AlphaStarPolicyNet from expert demonstrations
(AlphaStar pipeline, phase 2). Plain PyTorch — no RLlib.

Loads compact demo shards from collect_demos.py and reconstructs the
screen/minimap tensors on the fly via spatial_features.build_spatial_obs
(the same featurizer used by SpatialObsWrapper during RL — guaranteeing
train/rollout feature parity). Radio maps come from the mmap'd lookup array.

Loss: teacher-forced per-head masked cross-entropy over the 5 autoregressive
heads. The value head is left untouched (PPO refits it during fine-tuning).

Usage:
    python examples/train_sl_policy.py --epochs 1 --max-samples 5000   # smoke
    python examples/train_sl_policy.py --epochs 8                      # full
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env.spatial_features import build_spatial_obs, pool4
from models.alphastar_net import (
    AlphaStarPolicyNet, HEAD_SIZES, MASK_SLICES, NUM_HEADS, masked_logits,
)

DATA_ROOT    = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH      = f"{DATA_ROOT}/hm/hm_18.npy"
DEFAULT_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "radio_maps_hm_18.npy")
DEFAULT_DEMOS = os.path.join(_REPO_ROOT, "results", "demos")
DEFAULT_OUT   = os.path.join(_REPO_ROOT, "results", "sl_policy.pt")

HEAD_NAMES = ["move", "comm_r0", "comm_r1", "comm_r2", "comm_bs"]


class DemoDataset(Dataset):
    """Reconstructs (screen, minimap, scalars, mask, action) from compact demos."""

    def __init__(self, demo_dir, maps_path, hm_path, max_samples=None):
        shards = sorted(glob.glob(os.path.join(demo_dir, "shard_*.npz")))
        if not shards:
            raise FileNotFoundError(f"no demo shards in {demo_dir} — "
                                    "run collect_demos.py first")
        data = [np.load(s) for s in shards]
        self.fields = {k: np.concatenate([d[k] for d in data])
                       for k in data[0].keys()}
        n = len(self.fields["action"])
        if max_samples is not None and max_samples < n:
            idx = np.random.RandomState(0).permutation(n)[:max_samples]
            self.fields = {k: v[idx] for k, v in self.fields.items()}
        self._n = len(self.fields["action"])

        self.maps_path = maps_path
        self._maps = None              # lazy mmap (safe across DataLoader workers)

        hm = np.load(hm_path).astype(np.float32)
        std = float(hm.std())
        self._hm_norm = (hm - float(hm.mean())) / (std if std > 1e-6 else 1.0)
        self._hm_pool_norm = pool4(self._hm_norm)

    @property
    def maps(self):
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r")
        return self._maps

    def _radio_map(self, xy):
        x = int(round(float(np.clip(xy[0], 0, 255))))
        y = int(round(float(np.clip(xy[1], 0, 255))))
        return self.maps[y, x].astype(np.float32)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        f = self.fields
        self_xy, goal_xy, bs_xy = f["self_xy"][i], f["goal_xy"][i], f["bs_xy"][i]
        other_xys = f["other_xys"][i].reshape(-1, 2)

        spatial = build_spatial_obs(
            self._hm_norm, self._hm_pool_norm,
            own_radio_dbm=self._radio_map(self_xy),
            bs_radio_dbm=self._radio_map(bs_xy),
            self_xy=self_xy, goal_xy=goal_xy, bs_xy=bs_xy, other_xys=other_xys,
        )
        # Canonical scalar order (position, goal_vector, move_history) / 256,
        # matching TorchAlphaStarModel._split.
        goal_vector = np.clip(goal_xy - self_xy, -256.0, 256.0)
        scalars = np.concatenate([
            self_xy / 256.0, goal_vector / 256.0, f["move_history"][i] / 256.0,
        ]).astype(np.float32)

        return (spatial["screen"], spatial["minimap"], scalars,
                f["action_mask"][i].astype(np.float32),
                f["action"][i].astype(np.int64))


def evaluate(net, loader, device, comm_weight):
    net.eval()
    loss_sum, n = 0.0, 0
    correct = np.zeros(NUM_HEADS)
    with torch.no_grad():
        for screen, minimap, scalars, mask, action in loader:
            screen, minimap, scalars = (screen.to(device), minimap.to(device),
                                        scalars.to(device))
            mask, action = mask.to(device), action.to(device)
            ctx = net.encode(screen, minimap, scalars)
            logits = net.all_head_logits(ctx, action)
            loss = 0.0
            for k, (s, e) in enumerate(MASK_SLICES):
                ml = masked_logits(logits[k], mask[:, s:e])
                w = comm_weight if k > 0 else 1.0
                loss = loss + w * F.cross_entropy(ml, action[:, k])
                correct[k] += (ml.argmax(dim=1) == action[:, k]).sum().item()
            loss_sum += float(loss) * len(action)
            n += len(action)
    net.train()
    return loss_sum / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default=DEFAULT_DEMOS)
    parser.add_argument("--maps", default=DEFAULT_MAPS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--comm-weight", type=float, default=1.0,
                        help="Loss weight for comm heads (counter no-send imbalance)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = DemoDataset(args.demos, args.maps, HM_PATH, args.max_samples)
    n_val = max(int(len(ds) * args.val_fraction), args.batch_size)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"samples: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    net = AlphaStarPolicyNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(len(train_loader), 1))

    # Majority-class baselines for context on the reported accuracies.
    acts = ds.fields["action"]
    for k in range(NUM_HEADS):
        maj = np.bincount(acts[:, k], minlength=HEAD_SIZES[k]).max() / len(acts)
        print(f"  {HEAD_NAMES[k]}: majority baseline {maj:.3f}")

    for epoch in range(args.epochs):
        t0, running = time.time(), 0.0
        for step, (screen, minimap, scalars, mask, action) in enumerate(train_loader):
            screen, minimap, scalars = (screen.to(device), minimap.to(device),
                                        scalars.to(device))
            mask, action = mask.to(device), action.to(device)

            ctx = net.encode(screen, minimap, scalars)
            logits = net.all_head_logits(ctx, action)
            loss = 0.0
            for k, (s, e) in enumerate(MASK_SLICES):
                ml = masked_logits(logits[k], mask[:, s:e])
                w = args.comm_weight if k > 0 else 1.0
                loss = loss + w * F.cross_entropy(ml, action[:, k])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss)

            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(train_loader)} "
                      f"loss {running / (step + 1):.4f}")

        val_loss, val_acc = evaluate(net, val_loader, device, args.comm_weight)
        acc_str = "  ".join(f"{HEAD_NAMES[k]}={val_acc[k]:.3f}"
                            for k in range(NUM_HEADS))
        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"train_loss {running / max(len(train_loader), 1):.4f}  "
              f"val_loss {val_loss:.4f}  val_acc: {acc_str}  "
              f"({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
