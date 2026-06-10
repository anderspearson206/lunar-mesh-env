"""
Load a GAT-MARL PPO checkpoint and run an evaluation episode.

Usage
-----
    # Auto-detect latest checkpoint under ~/ray_results
    python examples/eval_ppo_gat.py

    # Specific checkpoint
    python examples/eval_ppo_gat.py --checkpoint /path/to/checkpoint_dir

    # Custom output stem, DB, and run label
    python examples/eval_ppo_gat.py --out run1 --db results/eval_gat.db --name gat_v1

Output: <out>.gif, metrics written to SQLite <db>.
The DB schema is identical to eval_ppo_checkpoint.py so both can be
compared in the same plotting script.
"""

import os
import sys
import glob
import argparse
import sqlite3
import datetime

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from lunar_mesh_env.marl_env_gat import LunarRoverMeshGATEnv
from models.gat_model import TorchGATModel

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
DATA_ROOT       = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH         = f"{DATA_ROOT}/hm/hm_18.npy"
DEFAULT_MAPS    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "radio_maps_hm_18.npy")
RAY_RESULTS_DIR = os.path.expanduser("~/ray_results")
NUM_AGENTS      = 3
MAX_STEPS       = 250
DEFAULT_DB      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "eval_metrics_gat.db")


# ---------------------------------------------------------------------------
# Env factory (must match train_ppo_gat.py)
# ---------------------------------------------------------------------------

def env_creator(config):
    hm          = np.load(config.get("hm_path", HM_PATH))
    maps_path   = config.get("maps_path", DEFAULT_MAPS)
    radio_model = RadioMapModelLookup(
        maps_path=maps_path,
        heightmap=hm,
        env_width=256,
        env_height=256,
    )
    raw_env = LunarRoverMeshGATEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
        render_mode="rgb_array",
        seed=19,
    )
    raw_env.EP_MAX_TIME = config.get("max_episode_steps", MAX_STEPS)
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space      = raw_env.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_checkpoint(results_dir: str, experiment_prefix: str = "lunar_mesh_ppo_gat") -> str:
    pattern = os.path.join(results_dir, f"{experiment_prefix}*", "**", "checkpoint_*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        # Wider fallback
        candidates = glob.glob(os.path.join(results_dir, "**", "checkpoint_*"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {results_dir}.\n"
            "Pass --checkpoint explicitly or check RAY_RESULTS_DIR."
        )
    return max(candidates, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Database (same schema as eval_ppo_checkpoint.py)
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "episodes" in tables and "run_name" not in existing_cols:
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
            reward                REAL,
            cumulative_reward     REAL,
            pos_x                 REAL,
            pos_y                 REAL,
            goal_x                REAL,
            goal_y                REAL,
            dist_to_goal          REAL,
            total_distance        REAL,
            goals_completed       INTEGER,
            mission_done          INTEGER,
            energy                REAL,
            datarate_mbps         REAL,
            bs_connected          INTEGER,
            num_neighbors         INTEGER,
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


def _agent_row(env, agent_id, step, reward, cumulative, episode_id):
    agent = env.agent_map[agent_id]
    dtn   = agent.payload_manager.get_state()
    dist  = float(np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2))
    return (
        episode_id, step, agent_id,
        reward, cumulative,
        float(agent.x), float(agent.y),
        float(agent.goal_x), float(agent.goal_y),
        dist,
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


def _env_row(env, step, episode_id):
    active   = env.agents
    avg_dr   = float(np.mean([env.agent_map[a].current_datarate for a in active])) if active else 0.0
    bs_links = sum(1 for a in active if env.agent_map[a].bs_connected)
    return (
        episode_id, step,
        int(env.sim_time),
        int(env.base_station.num_packets_received),
        avg_dr,
        bs_links / len(active) if active else 0.0,
        len(active),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory. Auto-detects latest if omitted.")
    parser.add_argument("--maps", type=str, default=None, metavar="PATH",
                        help=f"Precomputed radio maps .npy. Defaults to {DEFAULT_MAPS}.")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=199,
                        help="Env seed for evaluation episode.")
    parser.add_argument("--out", type=str, default="eval_gat",
                        help="Output filename stem for the GIF.")
    parser.add_argument("--db", type=str, default=DEFAULT_DB,
                        help="SQLite database path for metrics.")
    parser.add_argument("--name", type=str, default="gat",
                        help="Human-readable run label (used in plots).")
    args = parser.parse_args()

    maps_path = args.maps or (DEFAULT_MAPS if os.path.exists(DEFAULT_MAPS) else None)
    if maps_path is None:
        raise SystemExit(
            "ERROR: no radio maps file found.\n"
            "Run examples/precompute_radio_maps.py first, or pass --maps."
        )
    print(f"Radio maps : {maps_path}")

    checkpoint = args.checkpoint or find_latest_checkpoint(RAY_RESULTS_DIR)
    print(f"Checkpoint : {checkpoint}")

    # ── RLlib setup ──────────────────────────────────────────────────────────
    ModelCatalog.register_custom_model("gat_model", TorchGATModel)
    register_env("lunar_mesh_gat_v1", env_creator)
    ray.init(ignore_reinit_error=True, num_gpus=0)

    probe     = env_creator({"hm_path": HM_PATH, "num_agents": NUM_AGENTS,
                              "maps_path": maps_path})
    obs_space = probe.observation_space
    act_space = probe.action_space
    probe.close()

    algo = (
        PPOConfig()
        .environment("lunar_mesh_gat_v1",
                     env_config={"num_agents": NUM_AGENTS, "hm_path": HM_PATH,
                                 "maps_path": maps_path})
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(model={"custom_model": "gat_model",
                         "_disable_preprocessor_api": True})
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

    # ── Build eval env ───────────────────────────────────────────────────────
    hm          = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(maps_path=maps_path, heightmap=hm,
                                      env_width=256, env_height=256)
    env = LunarRoverMeshGATEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        render_mode="rgb_array",
        seed=args.seed,
    )

    # ── Init DB ──────────────────────────────────────────────────────────────
    conn = init_db(args.db)
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO episodes (timestamp, checkpoint, num_agents, dummy_mode, seed, run_name) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), checkpoint, NUM_AGENTS, 0, args.seed, args.name),
    )
    conn.commit()
    episode_id = cur.lastrowid
    print(f"Episode ID {episode_id} → {args.db}")

    # ── Eval loop ────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    frames         = []
    total_rewards  = {aid: 0.0 for aid in env.possible_agents}
    agent_rows     = []
    env_rows       = []

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
            agent_rows.append(_agent_row(env, aid, step, r, total_rewards[aid], episode_id))
        env_rows.append(_env_row(env, step, episode_id))

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if step % 20 == 0:
            reward_str = "  ".join(f"{aid}: {total_rewards[aid]:.1f}" for aid in env.possible_agents)
            print(f"  step {step:>4}  |  {reward_str}"
                  f"  |  BS pkts: {env.base_station.num_packets_received}"
                  f"  |  BS conn: {sum(env.agent_map[a].bs_connected for a in env.agents)}/{len(env.agents)}")

        if all(terms.values()) or all(truncs.values()):
            print(f"Episode finished at step {step}.")
            break

    # ── Write metrics ────────────────────────────────────────────────────────
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

    # ── Save GIF ─────────────────────────────────────────────────────────────
    if frames and _HAS_IMAGEIO:
        gif_path = f"{args.out}.gif"
        imageio.mimsave(gif_path, frames, fps=4)
        print(f"Saved {gif_path}")
    elif frames and not _HAS_IMAGEIO:
        print("imageio not installed — skipping GIF. Run: pip install imageio")

    print("\nFinal cumulative rewards:")
    for aid, r in total_rewards.items():
        print(f"  {aid}: {r:.2f}")
    print(f"\nPackets delivered to BS: {env.base_station.num_packets_received}"
          f"  (unique: {len(env.base_station.packets_received)})")


if __name__ == "__main__":
    main()