"""
Load the latest PPO checkpoint and run a visual evaluation episode.

Usage
-----
    # Auto-detect latest checkpoint
    python examples/eval_ppo_checkpoint.py

    # Point to a specific checkpoint directory
    python examples/eval_ppo_checkpoint.py --checkpoint /path/to/checkpoint_dir

    # Custom output stem and database path
    python examples/eval_ppo_checkpoint.py --out run1 --db results/eval.db

Output: <out>.gif in the current directory, metrics written to SQLite <db>.
"""

import os
import sys
import glob
import argparse
import sqlite3
import datetime

import numpy as np
import torch
import imageio

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from models.action_mask_model import TorchActionMaskModel

DEFAULT_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "radio_maps_hm_18.npy")

# ---------------------------------------------------------------------------
# Mirror the same paths / dims used in train_ppo_rllib.py
# ---------------------------------------------------------------------------
DATA_ROOT   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH     = f"{DATA_ROOT}/hm/hm_18.npy"
_PRETRAINED = os.path.join(_REPO_ROOT, "RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}

RAY_RESULTS_DIR = os.path.expanduser("~/ray_results")
NUM_AGENTS      = 3
MAX_STEPS       = 250
DEFAULT_DB      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_metrics_real_maps_3.db")


# ---------------------------------------------------------------------------
# Env creator (same as training)
# ---------------------------------------------------------------------------

def _build_radio_model(hm, maps_path=None, dummy_mode=True):
    """Return a lookup model if maps_path is given, otherwise RadioMapModelNN."""
    if maps_path:
        return RadioMapModelLookup(maps_path=maps_path, heightmap=hm,
                                   env_width=256, env_height=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RadioMapModelNN(model_paths=MODEL_PATHS, heightmap=hm,
                           env_width=256, env_height=256,
                           dummy_mode=dummy_mode, device=device)


def env_creator(config):
    hm = np.load(config.get("hm_path", HM_PATH))
    radio_model = _build_radio_model(
        hm,
        maps_path=config.get("maps_path", None),
        dummy_mode=config.get("dummy_mode", True),
    )

    raw_env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
        render_mode="rgb_array",
        seed=19,
    )
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space = raw_env.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_checkpoint(results_dir: str) -> str:
    pattern = os.path.join(results_dir, "**", "checkpoint_*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {results_dir}.\n"
            "Pass --checkpoint explicitly or check RAY_RESULTS_DIR."
        )
    return max(candidates, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Migrate older DBs that predate the run_name column
    existing = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    if "episodes" in {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
        if "run_name" not in existing:
            conn.execute("ALTER TABLE episodes ADD COLUMN run_name TEXT")
            conn.commit()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            checkpoint    TEXT    NOT NULL,
            num_agents    INTEGER NOT NULL,
            total_steps   INTEGER,
            dummy_mode    INTEGER NOT NULL,
            seed          INTEGER,
            run_name      TEXT
        );

        CREATE TABLE IF NOT EXISTS step_metrics (
            episode_id            INTEGER NOT NULL REFERENCES episodes(id),
            step                  INTEGER NOT NULL,
            agent_id              TEXT    NOT NULL,
            -- reward
            reward                REAL,
            cumulative_reward     REAL,
            -- navigation
            pos_x                 REAL,
            pos_y                 REAL,
            goal_x                REAL,
            goal_y                REAL,
            dist_to_goal          REAL,
            total_distance        REAL,
            goals_completed       INTEGER,
            mission_done          INTEGER,
            -- energy
            energy                REAL,
            -- communication
            datarate_mbps         REAL,
            bs_connected          INTEGER,
            num_neighbors         INTEGER,
            -- DTN buffer
            num_packets           INTEGER,
            num_packets_generated INTEGER,
            buffer_usage          REAL
        );

        CREATE TABLE IF NOT EXISTS env_step_metrics (
            episode_id            INTEGER NOT NULL REFERENCES episodes(id),
            step                  INTEGER NOT NULL,
            sim_time              INTEGER,
            bs_packets_received   INTEGER,
            avg_datarate_mbps     REAL,
            bs_link_ratio         REAL,
            num_active_agents     INTEGER
        );
    """)
    conn.commit()
    return conn


def collect_agent_metrics(env: LunarRoverMeshEnv, agent_id: str,
                          step: int, reward: float, cumulative: float,
                          episode_id: int) -> tuple:
    agent = env.agent_map[agent_id]
    dtn = agent.payload_manager.get_state()
    dist_to_goal = float(np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2))
    return (
        episode_id,
        step,
        agent_id,
        reward,
        cumulative,
        float(agent.x),
        float(agent.y),
        float(agent.goal_x),
        float(agent.goal_y),
        dist_to_goal,
        float(agent.total_distance),
        int(env.agent_goals_completed.get(agent_id, 0)),
        int(env.mission_done.get(agent_id, False)),
        float(agent.energy),
        float(agent.current_datarate),
        int(agent.bs_connected),
        int(len(agent.neighbors)),
        int(dtn["num_packets"]),
        int(dtn["num_packets_generated"]),
        float(dtn["payload_size"] / max(dtn["buffer_size"], 1)),
    )


def collect_env_metrics(env: LunarRoverMeshEnv, step: int, episode_id: int) -> tuple:
    active = env.agents
    avg_dr = float(np.mean([env.agent_map[a].current_datarate for a in active])) if active else 0.0
    bs_links = sum(1 for a in active if env.agent_map[a].bs_connected)
    bs_ratio = bs_links / len(active) if active else 0.0
    return (
        episode_id,
        step,
        int(env.sim_time),
        int(env.base_station.num_packets_received),
        avg_dr,
        bs_ratio,
        len(active),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--out", type=str, default="eval_ppo")
    parser.add_argument("--db", type=str, default=DEFAULT_DB,
                        help="SQLite database path for metrics.")
    parser.add_argument("--name", type=str, default=None,
                        help="Human-readable label for this run (shown in plots).")
    parser.add_argument("--lookup-maps", type=str, default=None, metavar="PATH",
                        help="Use precomputed radio maps (from precompute_radio_maps.py). "
                             "Fastest and matches lookup-trained policies. "
                             f"Defaults to {DEFAULT_MAPS} if that file exists.")
    parser.add_argument("--dummy-mode", action="store_true", default=True,
                        help="Use analytic radio model. Ignored when --lookup-maps is set.")
    parser.add_argument("--no-dummy-mode", dest="dummy_mode", action="store_false")
    args = parser.parse_args()

    # Auto-detect lookup maps if not specified but default file exists
    maps_path = args.lookup_maps
    if maps_path is None and os.path.exists(DEFAULT_MAPS):
        maps_path = DEFAULT_MAPS
        print(f"Using precomputed radio maps: {maps_path}")

    checkpoint = args.checkpoint or find_latest_checkpoint(RAY_RESULTS_DIR)
    print(f"Loading checkpoint: {checkpoint}")

    ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
    register_env("lunar_mesh_v1", env_creator)
    ray.init(ignore_reinit_error=True, num_gpus=0)

    probe = env_creator({"hm_path": HM_PATH, "num_agents": NUM_AGENTS,
                         "dummy_mode": args.dummy_mode, "maps_path": maps_path})
    obs_space = probe.observation_space
    act_space = probe.action_space
    probe.close()

    algo = (
        PPOConfig()
        .environment("lunar_mesh_v1",
                     env_config={"num_agents": NUM_AGENTS, "hm_path": HM_PATH,
                                 "maps_path": maps_path, "dummy_mode": args.dummy_mode})
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            model={"custom_model": "action_mask_model",
                   "_disable_preprocessor_api": True},
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
        .build()
    )
    algo.restore(checkpoint)
    policy = algo.get_policy("shared_policy")
    print("Policy restored.")

    # ── Build raw env for eval ────────────────────────────────────────────
    hm = np.load(HM_PATH)
    radio_model = _build_radio_model(hm, maps_path=maps_path, dummy_mode=args.dummy_mode)

    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        render_mode="rgb_array",
        seed=199,
    )

    # ── Init DB ───────────────────────────────────────────────────────────
    conn = init_db(args.db)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO episodes (timestamp, checkpoint, num_agents, dummy_mode, seed, run_name) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), checkpoint, NUM_AGENTS,
         int(args.dummy_mode), 19, args.name),
    )
    conn.commit()
    episode_id = cur.lastrowid
    print(f"Episode ID {episode_id} → {args.db}")

    # ── Eval loop ─────────────────────────────────────────────────────────
    obs, _ = env.reset()
    frames = []
    total_rewards = {aid: 0.0 for aid in env.possible_agents}
    agent_rows = []
    env_rows   = []

    print(f"Running {args.steps} steps...")
    for step in range(args.steps):
        actions = {}
        for agent_id in env.agents:
            action, _, _ = policy.compute_single_action(
                obs[agent_id],
                policy_id="shared_policy",
                explore=False,
            )
            actions[agent_id] = action
            if step == 0:
                mask = obs[agent_id]["action_mask"]
                print(f"  [{agent_id}] move mask: {mask[:9]}  action: {action}")

        obs, rewards, terms, truncs, _ = env.step(actions)

        for aid, r in rewards.items():
            total_rewards[aid] += r
            agent_rows.append(
                collect_agent_metrics(env, aid, step, r, total_rewards[aid], episode_id)
            )

        env_rows.append(collect_env_metrics(env, step, episode_id))

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if step % 20 == 0:
            reward_str = "  ".join(
                f"{aid}: {total_rewards[aid]:.1f}" for aid in env.possible_agents
            )
            print(f"  step {step:>4}  |  cumulative rewards: {reward_str}"
                  f"  |  BS pkts: {env.base_station.num_packets_received}")

        if all(terms.values()) or all(truncs.values()):
            print(f"Episode finished at step {step}.")
            break

    # ── Write metrics to DB ───────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO step_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        agent_rows,
    )
    conn.executemany(
        "INSERT INTO env_step_metrics VALUES (?,?,?,?,?,?,?)",
        env_rows,
    )
    cur.execute("UPDATE episodes SET total_steps=? WHERE id=?", (step + 1, episode_id))
    conn.commit()
    conn.close()
    print(f"Metrics saved → {args.db}  (episode_id={episode_id})")

    env.close()
    algo.stop()
    ray.shutdown()

    # ── Save video ────────────────────────────────────────────────────────
    if frames:
        gif_path = f"{args.out}.gif"
        imageio.mimsave(gif_path, frames, fps=4)
        print(f"Saved {gif_path}")

    print("\nFinal cumulative rewards:")
    for aid, r in total_rewards.items():
        print(f"  {aid}: {r:.2f}")


if __name__ == "__main__":
    main()
