"""
Diagnostic: are the waypoint_signals in the obs actually informative?

Runs one episode with a random policy, records the waypoint_signals vector
on every DECISION step, and prints:
  - mean signal per codebook entry (NULL baseline vs each offset)
  - stddev across decision steps (how much variation there is)
  - which codebook entries consistently beat the NULL/straight path

If all 16 entries have similar mean signal → the terrain is too flat for
the signal obs to guide nav decisions. If some entries consistently dominate →
the obs is informative but training failed to use it.

Usage:
    python examples/inspect_waypoint_signals.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lunar_mesh_env.marl_env_lozano_waypoint_nav import LunarRoverMeshLozanoWaypointNavEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup

DATA_ROOT = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH   = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radio_maps_hm_18.npy")

N_NAV      = LunarRoverMeshLozanoWaypointNavEnv.N_NAV
NUM_AGENTS = 3

_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_CB_LABELS = [f"{d}×{m}" for m in ["R/2", "R"] for d in _COMPASS]


def main():
    hm = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(
        maps_path=MAPS_PATH, heightmap=hm, env_width=256, env_height=256,
    )

    # Collect signals across multiple seeds
    seeds = [42, 99, 137, 7, 21]
    episodes_per_seed = 3

    all_bs_signal: list[float] = []
    all_wp_signals: list[np.ndarray] = []   # shape (N_NAV-1,) each

    for seed in seeds:
        env = LunarRoverMeshLozanoWaypointNavEnv(
            hm_path=HM_PATH,
            radio_model=radio_model,
            num_agents=NUM_AGENTS,
            packet_mode="rate",
            seed=seed,
        )

        for _ in range(episodes_per_seed):
            obs, _ = env.reset()
            done = False
            step = 0

            while not done and step < 300:
                for aid, o in obs.items():
                    if o["decision_due"][0]:
                        all_bs_signal.append(float(o["bs_signal"][0]))
                        all_wp_signals.append(o["waypoint_signals"].copy())

                # random actions
                actions = {}
                for aid in obs:
                    mask = obs[aid]["action_mask"]
                    valid_nav  = np.where(mask[:N_NAV])[0]
                    valid_comm = np.where(mask[N_NAV:])[0]
                    nav_a  = int(np.random.choice(valid_nav))
                    comm_a = int(np.random.choice(valid_comm)) if len(valid_comm) else 0
                    actions[aid] = np.array([nav_a, comm_a])

                obs, _, terms, truncs, _ = env.step(actions)
                done = all(terms.values()) or all(truncs.values()) or not obs
                step += 1

        env.close()

    n = len(all_wp_signals)
    print(f"Collected {n} decision-step observations across "
          f"{len(seeds)} seeds × {episodes_per_seed} episodes\n")

    if n == 0:
        print("No decision steps observed — check WAYPOINT_INTERVAL.")
        return

    bs_arr = np.array(all_bs_signal)          # (n,)
    wp_arr = np.vstack(all_wp_signals)         # (n, 16)

    print(f"BS signal at current position:")
    print(f"  mean={bs_arr.mean():.3f}  std={bs_arr.std():.3f}  "
          f"min={bs_arr.min():.3f}  max={bs_arr.max():.3f}\n")

    print(f"Waypoint signals (normalised dBm, 0=threshold, 1=threshold+40dB):")
    print(f"{'#':<4} {'Label':<12} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}  vs_bs")
    print("-" * 56)
    for i, label in enumerate(_CB_LABELS):
        col  = wp_arr[:, i]
        diff = col.mean() - bs_arr.mean()
        sign = "+" if diff >= 0 else ""
        print(f"  {i+1:<3} {label:<12} {col.mean():>7.3f} {col.std():>7.3f} "
              f"{col.min():>7.3f} {col.max():>7.3f}  {sign}{diff:.3f}")

    # How often does each waypoint beat the current position signal?
    print(f"\nFraction of steps where waypoint signal > current bs_signal:")
    for i, label in enumerate(_CB_LABELS):
        frac = (wp_arr[:, i] > bs_arr).mean()
        bar  = "#" * int(frac * 40)
        print(f"  {i+1:<3} {label:<12}  {frac:>5.1%}  {bar}")

    # Overall range check
    signal_range = wp_arr.max(axis=1) - wp_arr.min(axis=1)
    print(f"\nPer-step signal range across 16 waypoints:")
    print(f"  mean={signal_range.mean():.3f}  std={signal_range.std():.3f}  "
          f"max={signal_range.max():.3f}")
    print(f"  (>0.05 range on >50% of steps means the obs has real structure)")
    frac_useful = (signal_range > 0.05).mean()
    print(f"  Steps with >0.05 range: {frac_useful:.1%}")


if __name__ == "__main__":
    main()