"""
Navigation comparison eval harness.

Runs multiple nav variants back-to-back on the same terrain/seed and outputs:
  - per-episode metrics: unique packets received, goals completed, steps taken,
    path extra vs A* baseline (proxy: total_distance / A*_distance)
  - mean ± 95% CI across seeds
  - SQLite table + matplotlib summary plot

Nav variants compared (spec §9):
  1. lozano            — A* terrain nav (reference floor), Lozano comm
  2. lozano_conn_nav   — connectivity-aware A* nav (the "bar to beat"), Lozano comm
  3. lozano_res_nav    — per-step ±1 sector residual RL, Lozano comm
  4. lozano_waypoint_nav — slow-timescale waypoint-residual RL (this work), Lozano comm
  5. lozano_rl_nav     — full RL nav + comm (expressiveness ceiling)

Checkpoints are passed via --checkpoints (one per variant, order matching --models).
Variants with --routing_only (no checkpoint) use heuristic nav only.

Usage:
    python examples/eval_nav_comparison.py \\
        --models lozano lozano_conn_nav lozano_res_nav lozano_waypoint_nav \\
        --checkpoints /path/ck1 none /path/ck3 /path/ck4 \\
        --episodes 30 --steps 500 --seeds 42 99 137 \\
        --db examples/nav_comparison.db \\
        --plot examples/nav_comparison.png

Pass "none" as a checkpoint to run that variant without a trained policy
(useful for the A* baselines which don't need RL).
"""

import argparse
import os
import sys
import sqlite3
import warnings
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from lunar_mesh_env import LunarRoverMeshEnv
from lunar_mesh_env.marl_env_lozano import LunarRoverMeshLozanoEnv
from lunar_mesh_env.marl_env_lozano_conn_nav import LunarRoverMeshLozanoConnNavEnv
from lunar_mesh_env.marl_env_lozano_res_nav import LunarRoverMeshLozanoResNavEnv
from lunar_mesh_env.marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv
from lunar_mesh_env.marl_env_lozano_waypoint_nav import LunarRoverMeshLozanoWaypointNavEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from models.mlp_gat_lozano_model import TorchMLPGATLozanoModel
from models.mlp_gat_lozano_waypoint_model import TorchMLPGATLozanoWaypointModel

DATA_ROOT = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH   = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "radio_maps_hm_18.npy")

NUM_AGENTS = 3

_ENV_CLS = {
    "lozano":             LunarRoverMeshLozanoEnv,
    "lozano_conn_nav":    LunarRoverMeshLozanoConnNavEnv,
    "lozano_res_nav":     LunarRoverMeshLozanoResNavEnv,
    "lozano_rl_nav":      LunarRoverMeshLozanoRLNavEnv,
    "lozano_waypoint_nav": LunarRoverMeshLozanoWaypointNavEnv,
}
_MODEL_NAME = {
    "lozano":             "mlp_gat_lozano_model",
    "lozano_conn_nav":    "mlp_gat_lozano_model",
    "lozano_res_nav":     "mlp_gat_lozano_model",
    "lozano_rl_nav":      "mlp_gat_lozano_model",
    "lozano_waypoint_nav": "mlp_gat_lozano_waypoint_model",
}
_MOVE_DIRS = {
    "lozano":             1,
    "lozano_conn_nav":    1,
    "lozano_res_nav":     9,
    "lozano_rl_nav":      9,
    "lozano_waypoint_nav": LunarRoverMeshLozanoWaypointNavEnv.N_NAV,
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nav_comparison (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            variant         TEXT    NOT NULL,
            seed            INTEGER NOT NULL,
            episode         INTEGER NOT NULL,
            steps           INTEGER,
            unique_pkts     INTEGER,
            goals_completed INTEGER,
            total_distance  REAL,
            run_ts          TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def insert_episode(conn, variant, seed, episode, steps, unique_pkts,
                   goals, distance):
    conn.execute(
        "INSERT INTO nav_comparison "
        "(variant, seed, episode, steps, unique_pkts, goals_completed, total_distance) "
        "VALUES (?,?,?,?,?,?,?)",
        (variant, seed, episode, steps, unique_pkts, goals, distance),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(model_name: str, seed: int, maps_path: str) -> tuple:
    hm = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(
        maps_path=maps_path, heightmap=hm,
        env_width=256, env_height=256,
    )
    env_cls = _ENV_CLS[model_name]
    kwargs = dict(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        packet_mode="rate",
        render_mode="rgb_array",
        seed=seed,
    )
    if model_name in ("lozano", "lozano_conn_nav"):
        kwargs["action_type"] = "multidiscrete"
    raw_env = env_cls(**kwargs)
    if model_name in ("lozano_rl_nav",):
        raw_env.eps_nav = 0.0
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space      = raw_env.action_space(raw_env.possible_agents[0])
    return env, raw_env


# ---------------------------------------------------------------------------
# Run one episode; return metrics dict
# ---------------------------------------------------------------------------

def run_episode(policy, env, raw_env, max_steps: int) -> dict:
    obs, _ = env.reset()
    done   = False
    step   = 0
    total_dist = sum(a.total_distance for a in raw_env.agent_map.values())

    while not done and step < max_steps:
        actions = {}
        for agent_id, o in obs.items():
            action, _, _ = policy.compute_single_action(
                o, explore=False, policy_id="shared_policy"
            )
            actions[agent_id] = action
        obs, _, terminations, truncations, _ = env.step(actions)
        done = all(terminations.values()) or all(truncations.values()) or not obs
        step += 1

    dist_delta = sum(a.total_distance for a in raw_env.agent_map.values()) - total_dist
    return {
        "steps":           step,
        "unique_pkts":     raw_env.unique_packets_rcvd,
        "goals_completed": sum(raw_env.agent_goals_completed.values()),
        "total_distance":  dist_delta,
    }


# ---------------------------------------------------------------------------
# Build algo for a checkpoint (or None for heuristic-only)
# ---------------------------------------------------------------------------

def build_algo(model_name: str, checkpoint_path: str | None,
               maps_path: str, seed: int):
    env_tmp, _ = make_env(model_name, seed, maps_path)
    obs_space  = env_tmp.observation_space
    act_space  = env_tmp.action_space
    env_tmp.close()

    move_dirs   = _MOVE_DIRS[model_name]
    custom_model = _MODEL_NAME[model_name]

    cfg = (
        PPOConfig()
        .environment("lunar_mesh_v1")
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda aid, *a, **kw: "shared_policy",
        )
        .training(model={
            "custom_model":              custom_model,
            "_disable_preprocessor_api": True,
            "custom_model_config":       {"move_dirs": move_dirs},
        })
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )
    algo = cfg.build()
    if checkpoint_path and checkpoint_path.lower() != "none":
        algo.restore(checkpoint_path)
    return algo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Navigation variant comparison")
    parser.add_argument("--models", nargs="+",
                        default=["lozano", "lozano_conn_nav",
                                 "lozano_res_nav", "lozano_waypoint_nav"],
                        choices=list(_ENV_CLS),
                        help="Nav variants to compare (in order)")
    parser.add_argument("--checkpoints", nargs="+", default=[],
                        help="Checkpoint paths (one per model; 'none' for heuristic)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per (variant × seed)")
    parser.add_argument("--steps",    type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42, 99, 137],
                        help="Eval seeds")
    parser.add_argument("--maps",     default=MAPS_PATH,
                        help="Precomputed radio maps .npy")
    parser.add_argument("--db",       default="examples/nav_comparison.db",
                        help="SQLite output path")
    parser.add_argument("--plot",     default="examples/nav_comparison.png",
                        help="Output plot path (set to none to skip)")
    args = parser.parse_args()

    # Pad checkpoints list with None if shorter than models list
    ckpts = list(args.checkpoints)
    while len(ckpts) < len(args.models):
        ckpts.append(None)

    # Register models
    ModelCatalog.register_custom_model("mlp_gat_lozano_model", TorchMLPGATLozanoModel)
    ModelCatalog.register_custom_model(
        "mlp_gat_lozano_waypoint_model", TorchMLPGATLozanoWaypointModel
    )
    register_env("lunar_mesh_v1", lambda cfg: make_env(
        cfg.get("model_type", "lozano"), cfg.get("seed", 42), args.maps
    )[0])

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": repo_root}},
        _temp_dir=os.path.expanduser("~/ray_temp"),
    )

    conn = init_db(args.db)
    all_results: dict[str, list[float]] = {m: [] for m in args.models}

    for model_name, ckpt in zip(args.models, ckpts):
        print(f"\n{'='*60}")
        print(f"Variant: {model_name}  checkpoint: {ckpt or 'heuristic-only'}")
        algo = build_algo(model_name, ckpt, args.maps, seed=args.seeds[0])
        policy = algo.get_policy("shared_policy")

        for seed in args.seeds:
            env, raw_env = make_env(model_name, seed, args.maps)
            for ep in range(args.episodes):
                metrics = run_episode(policy, env, raw_env, args.steps)
                insert_episode(
                    conn, model_name, seed, ep,
                    metrics["steps"], metrics["unique_pkts"],
                    metrics["goals_completed"], metrics["total_distance"],
                )
                all_results[model_name].append(metrics["unique_pkts"])
                if (ep + 1) % 5 == 0:
                    print(
                        f"  seed={seed} ep={ep+1}/{args.episodes}  "
                        f"pkts={metrics['unique_pkts']}  "
                        f"goals={metrics['goals_completed']}"
                    )
            env.close()
        algo.stop()

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Variant':<25} {'Mean pkts':>10} {'95% CI':>12} {'N':>5}")
    print("-" * 55)
    for model_name in args.models:
        vals = np.array(all_results[model_name], dtype=float)
        if len(vals) == 0:
            continue
        mean = vals.mean()
        ci   = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"{model_name:<25} {mean:>10.1f} {f'±{ci:.1f}':>12} {len(vals):>5}")

    # ── Plot ───────────────────────────────────────────────────────────
    if args.plot and args.plot.lower() != "none":
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            labels, means, cis = [], [], []
            for model_name in args.models:
                vals = np.array(all_results[model_name], dtype=float)
                if len(vals) == 0:
                    continue
                labels.append(model_name.replace("lozano_", "").replace("_nav", ""))
                means.append(vals.mean())
                cis.append(
                    1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
                    if len(vals) > 1 else 0.0
                )
            xs = range(len(labels))
            ax.bar(xs, means, yerr=cis, capsize=4, alpha=0.8)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel("Unique packets received (mean ± 95% CI)")
            ax.set_title("Navigation variant comparison")
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"\nPlot saved to {args.plot}")
        except ImportError:
            warnings.warn("matplotlib not available — skipping plot")

    conn.close()
    ray.shutdown()


if __name__ == "__main__":
    main()