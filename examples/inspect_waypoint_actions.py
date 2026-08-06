"""
Diagnostic: inspect what nav actions the waypoint_nav policy chooses.

Runs N episodes, records action[0] on every DECISION step (decision_due=1),
and prints a histogram of NULL vs each of the 16 codebook entries.

Usage:
    python examples/inspect_waypoint_actions.py --checkpoint /path/to/ckpt
    python examples/inspect_waypoint_actions.py --checkpoint /path/to/ckpt --episodes 10 --steps 300
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from lunar_mesh_env.marl_env_lozano_waypoint_nav import LunarRoverMeshLozanoWaypointNavEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from models.mlp_gat_lozano_waypoint_model import TorchMLPGATLozanoWaypointModel

DATA_ROOT = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH   = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radio_maps_hm_18.npy")

N_NAV = LunarRoverMeshLozanoWaypointNavEnv.N_NAV  # 17
NUM_AGENTS = 3

# Codebook label: direction × magnitude
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_LABELS = ["NULL"] + [
    f"{d}×{m}" for m in ["R/2", "R"] for d in _COMPASS
]


def make_env(seed: int):
    hm = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(
        maps_path=MAPS_PATH, heightmap=hm, env_width=256, env_height=256,
    )
    raw_env = LunarRoverMeshLozanoWaypointNavEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        packet_mode="rate",
        seed=seed,
    )
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space      = raw_env.action_space(raw_env.possible_agents[0])
    return env, raw_env


def build_algo(checkpoint: str, obs_space, act_space):
    cfg = (
        PPOConfig()
        .environment("lunar_mesh_waypoint_diag_v1")
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
            "custom_model":              "mlp_gat_lozano_waypoint_model",
            "_disable_preprocessor_api": True,
            "custom_model_config":       {"move_dirs": N_NAV},
        })
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )
    algo = cfg.build()
    algo.restore(checkpoint)
    return algo


def run(args):
    ModelCatalog.register_custom_model(
        "mlp_gat_lozano_waypoint_model", TorchMLPGATLozanoWaypointModel
    )

    probe_env, _ = make_env(seed=42)
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    probe_env.close()

    register_env("lunar_mesh_waypoint_diag_v1",
                 lambda cfg: make_env(cfg.get("seed", 42))[0])

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": repo_root}},
        _temp_dir=os.path.expanduser("~/ray_temp"),
    )

    algo = build_algo(args.checkpoint, obs_space, act_space)
    policy = algo.get_policy("shared_policy")

    # Counters
    nav_counts   = np.zeros(N_NAV, dtype=int)   # action[0] on decision steps
    total_decisions = 0
    total_steps  = 0
    # Per-episode: nav_entropy (action dist entropy on decision steps)
    episode_null_fracs = []

    seeds = args.seeds if args.seeds else list(range(args.episodes))

    for ep_idx in range(args.episodes):
        seed = seeds[ep_idx % len(seeds)]
        env, raw_env = make_env(seed=seed)
        obs, _ = env.reset()
        done = False
        step = 0
        ep_decisions = 0
        ep_null = 0

        while not done and step < args.steps:
            actions = {}
            for agent_id, o in obs.items():
                action, _, extra = policy.compute_single_action(
                    o, explore=False, policy_id="shared_policy"
                )
                actions[agent_id] = action

                # Only record on decision steps
                if o.get("decision_due", np.array([0]))[0]:
                    nav_a = int(action[0])
                    nav_counts[nav_a] += 1
                    total_decisions += 1
                    ep_decisions += 1
                    if nav_a == 0:
                        ep_null += 1

            obs, _, terminations, truncations, _ = env.step(actions)
            done = all(terminations.values()) or all(truncations.values()) or not obs
            step += 1

        total_steps += step
        null_frac = ep_null / max(ep_decisions, 1)
        episode_null_fracs.append(null_frac)
        print(f"  ep {ep_idx+1:>3}  seed={seed}  steps={step}  "
              f"decisions={ep_decisions}  NULL%={null_frac*100:.0f}%  "
              f"pkts={raw_env.unique_packets_rcvd}  "
              f"goals={sum(raw_env.agent_goals_completed.values())}")
        env.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Nav action histogram across {total_decisions} decision steps "
          f"({args.episodes} episodes)")
    print(f"{'Action':<12} {'Label':<10} {'Count':>8} {'%':>7}")
    print("-" * 42)
    for i, cnt in enumerate(nav_counts):
        pct = 100.0 * cnt / max(total_decisions, 1)
        marker = " <--" if i == 0 else ""
        print(f"  {i:<10} {_LABELS[i]:<10} {cnt:>8} {pct:>6.1f}%{marker}")

    print(f"\nNULL fraction: {nav_counts[0]/max(total_decisions,1)*100:.1f}%  "
          f"(mean per episode: {np.mean(episode_null_fracs)*100:.1f}%)")
    non_null = nav_counts[1:]
    if non_null.sum() > 0:
        top3 = np.argsort(non_null)[::-1][:3]
        print("Top non-NULL choices:")
        for rank, idx in enumerate(top3):
            print(f"  #{rank+1}: action {idx+1} ({_LABELS[idx+1]})  "
                  f"{non_null[idx]} times ({100*non_null[idx]/max(total_decisions,1):.1f}%)")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--steps",      type=int, default=300)
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42, 99, 137])
    run(parser.parse_args())