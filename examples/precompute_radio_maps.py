"""
Precompute radio maps for every integer TX position on a heightmap.

Output
------
  <out>.npy  – float16 array, shape (256, 256, 256, 256)
               index as: maps[tx_y, tx_x]  → 256×256 dBm signal map

Usage
-----
  python examples/precompute_radio_maps.py
  python examples/precompute_radio_maps.py --hm DATA/radio_data_2/radio_data_2/hm/hm_18.npy
  python examples/precompute_radio_maps.py --batch 64 --out radio_maps_hm18

Storage: 256^4 * 2 bytes (float16) ≈ 8.6 GB per map
Time   : ~256*256/batch × t_per_batch  (≈20-30 min at batch=32 on RTX 3090)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lunar_mesh_env import RadioMapModelNN

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_ROOT   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
DEFAULT_HM  = f"{DATA_ROOT}/hm/hm_18.npy"
_PRETRAINED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}
FREQ        = "5.8"
MAP_SIZE    = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hm",    default=DEFAULT_HM,
                        help="Path to heightmap .npy")
    parser.add_argument("--out",   default=None,
                        help="Output file stem (no extension). "
                             "Default: radio_maps_<hm_stem>")
    parser.add_argument("--batch", type=int, default=32,
                        help="TX positions per GPU batch (default 32)")
    parser.add_argument("--freq",  default=FREQ,
                        help="Radio frequency band (default 5.8)")
    args = parser.parse_args()

    hm_stem = os.path.splitext(os.path.basename(args.hm))[0]
    out_path = (args.out or f"radio_maps_{hm_stem}") + ".npy"
    progress_path = out_path + ".progress"   # stores last completed tx_y row

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"HM     : {args.hm}")
    print(f"Output : {out_path}")
    print(f"Batch  : {args.batch}  TX positions per GPU call")

    # ── Load NN ──────────────────────────────────────────────────────────────
    hm = np.load(args.hm)
    model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=hm,
        env_width=MAP_SIZE,
        env_height=MAP_SIZE,
        dummy_mode=False,
        device=device,
    )

    # ── Allocate output (memory-mapped so RAM is not saturated) ───────────────
    shape = (MAP_SIZE, MAP_SIZE, MAP_SIZE, MAP_SIZE)   # [tx_y, tx_x, rx_y, rx_x]
    size_gb = np.prod(shape) * 2 / 1e9                 # float16 = 2 bytes
    print(f"Output : {shape}  ({size_gb:.1f} GB float16)")

    if not os.path.exists(out_path):
        print("Allocating output file...")
        fp = np.lib.format.open_memmap(out_path, mode="w+",
                                       dtype=np.float16, shape=shape)
        del fp
    else:
        print("Output file already exists — resuming.")

    fp = np.lib.format.open_memmap(out_path, mode="r+",
                                   dtype=np.float16, shape=shape)

    # ── Resume support ────────────────────────────────────────────────────────
    start_row = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            start_row = int(f.read().strip()) + 1
        print(f"Resuming from tx_y = {start_row}")

    total_rows   = MAP_SIZE
    remaining    = total_rows - start_row
    total_pos    = MAP_SIZE * MAP_SIZE
    done_pos     = start_row * MAP_SIZE

    t_start = time.time()

    # ── Main loop: iterate over all (tx_y, tx_x) ─────────────────────────────
    for ty in range(start_row, MAP_SIZE):
        # Build list of all TX positions in this row
        row_positions = [(float(tx), float(ty)) for tx in range(MAP_SIZE)]

        # Process row in sub-batches
        for b_start in range(0, MAP_SIZE, args.batch):
            batch_pos = row_positions[b_start: b_start + args.batch]
            maps = model._run_batch_inference(batch_pos, args.freq)
            # maps: (batch, 256, 256) float64 → store as float16
            for i, (tx, _) in enumerate(batch_pos):
                fp[ty, int(tx)] = maps[i].astype(np.float16)

        fp.flush()

        # ── Progress ──────────────────────────────────────────────────────
        done_pos += MAP_SIZE
        elapsed  = time.time() - t_start
        rows_done = ty - start_row + 1
        rate_rows = rows_done / elapsed           # rows per second
        eta_s     = (remaining - rows_done) / rate_rows if rate_rows > 0 else 0
        pct       = done_pos / total_pos * 100
        print(f"  tx_y={ty:>3}  {pct:5.1f}%  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta_s/60:.1f}m", flush=True)

        # Save resume checkpoint
        with open(progress_path, "w") as f:
            f.write(str(ty))

    del fp

    # Clean up progress file on success
    if os.path.exists(progress_path):
        os.remove(progress_path)

    elapsed = time.time() - t_start
    print(f"\nDone. {out_path}  ({size_gb:.1f} GB)  "
          f"in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()