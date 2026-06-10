"""
A* baseline evaluation — rovers navigate purely via A* pathfinding.

Each rover follows a precomputed A* path to its assigned goal.  On arrival a
new goal is drawn and A* is re-run.  No RL policy is involved.

Metrics are written to the same SQLite schema used by eval_ppo_checkpoint.py
so results can be compared directly with plot_eval_metrics.py.

Usage
-----
    python examples/eval_astar_baseline.py
    python examples/eval_astar_baseline.py --steps 500 --episodes 5 --out astar
    python examples/eval_astar_baseline.py --db path/to/eval.db --name "A* baseline"
"""

import argparse
import datetime
import os
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import imageio

from lunar_mesh_env import LunarRoverMeshEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from lunar_mesh_env.pathfinding import a_star_search

DATA_ROOT   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH     = f"{DATA_ROOT}/hm/hm_18.npy"
DEFAULT_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "radio_maps_hm_18.npy")
DEFAULT_DB   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "eval_metrics_real_maps_3.db")

NUM_AGENTS  = 3
MAX_STEPS   = 500
NUM_EPISODES = 1


# ---------------------------------------------------------------------------
# Database (identical schema to eval_ppo_checkpoint.py)
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
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


def collect_agent_metrics(env, agent_id, step, reward, cumulative, episode_id):
    agent = env.agent_map[agent_id]
    dtn = agent.payload_manager.get_state()
    dist_to_goal = float(np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2))
    return (
        episode_id, step, agent_id,
        reward, cumulative,
        float(agent.x), float(agent.y),
        float(agent.goal_x), float(agent.goal_y),
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


def collect_env_metrics(env, step, episode_id):
    active = env.agents
    avg_dr = float(np.mean([env.agent_map[a].current_datarate for a in active])) if active else 0.0
    bs_links = sum(1 for a in active if env.agent_map[a].bs_connected)
    bs_ratio = bs_links / len(active) if active else 0.0
    return (
        episode_id, step,
        int(env.sim_time),
        int(env.base_station.num_packets_received),
        avg_dr, bs_ratio,
        len(active),
    )


# ---------------------------------------------------------------------------
# A* path management
# ---------------------------------------------------------------------------

def compute_path(env, agent_id: str) -> list:
    """Run A* from agent's current position to its goal."""
    agent = env.agent_map[agent_id]
    start = (int(round(agent.x)), int(round(agent.y)))
    goal  = (int(round(agent.goal_x)), int(round(agent.goal_y)))
    path  = a_star_search(env.heightmap, start, goal,
                          max_incline=env.MAX_INCLINE_PER_STEP)
    return path if path is not None else []


def refresh_paths(env, agent_ids):
    """Recompute A* paths for the given agents."""
    for aid in agent_ids:
        env.agent_map[aid].nav_path = compute_path(env, aid)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def astar_action(env, agent_id: str) -> list:
    """Movement via heuristic_move_action; no communication actions."""
    move = env.heuristic_move_action(agent_id)
    num_rovers = len(env.possible_agents)
    comm = [0] * (num_rovers + 1)
    return [move] + comm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=DEFAULT_MAPS,
                        help="Precomputed radio maps .npy")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--out", default="eval_astar",
                        help="Output filename stem for GIF")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help="SQLite database for metrics")
    parser.add_argument("--name", default="A* baseline",
                        help="Run label shown in plots")
    parser.add_argument("--seed", type=int, default=199)
    parser.add_argument("--render", action="store_true",
                        help="Save a GIF of the first episode")
    parser.add_argument("--routing", default="epidemic",
                        choices=["none", "epidemic", "spray_and_wait"],
                        help="DTN routing protocol (default: epidemic)")
    parser.add_argument("--spray-copies", type=int, default=4,
                        help="Initial token count for spray-and-wait (default: 4)")
    args = parser.parse_args()

    hm = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(
        maps_path=args.maps, heightmap=hm, env_width=256, env_height=256
    )

    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        render_mode="rgb_array" if args.render else None,
        routing_protocol=args.routing,
        spray_copies=args.spray_copies,
        seed=args.seed,
    )
    env.EP_MAX_TIME = args.steps

    conn = init_db(args.db)
    cur  = conn.cursor()

    for ep in range(args.episodes):
        cur.execute(
            "INSERT INTO episodes (timestamp, checkpoint, num_agents, dummy_mode, seed, run_name) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.datetime.now().isoformat(), "astar",
             NUM_AGENTS, 0, args.seed + ep, args.name),
        )
        conn.commit()
        episode_id = cur.lastrowid
        print(f"\n=== Episode {ep+1}/{args.episodes}  (id={episode_id}) ===")

        obs, _ = env.reset()

        # Compute initial A* paths for all agents after reset.
        refresh_paths(env, env.agents)

        total_rewards = {aid: 0.0 for aid in env.possible_agents}
        prev_goals    = {aid: (env.agent_map[aid].goal_x, env.agent_map[aid].goal_y)
                         for aid in env.possible_agents}
        agent_rows, env_rows = [], []
        frames = []

        for step in range(args.steps):
            actions = {aid: astar_action(env, aid) for aid in env.agents}
            obs, rewards, terms, truncs, _ = env.step(actions)

            # Re-run A* whenever a goal changes (new goal assigned after arrival).
            for aid in env.agents:
                agent = env.agent_map[aid]
                new_goal = (agent.goal_x, agent.goal_y)
                if new_goal != prev_goals.get(aid):
                    env.agent_map[aid].nav_path = compute_path(env, aid)
                    prev_goals[aid] = new_goal

            for aid, r in rewards.items():
                total_rewards[aid] += r
                agent_rows.append(
                    collect_agent_metrics(env, aid, step, r, total_rewards[aid], episode_id)
                )
            env_rows.append(collect_env_metrics(env, step, episode_id))

            if args.render and ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if step % 50 == 0:
                reward_str = "  ".join(
                    f"{aid}: {total_rewards[aid]:.1f}" for aid in env.possible_agents
                )
                goals_str = "  ".join(
                    f"{aid}: {env.agent_goals_completed.get(aid, 0)}"
                    for aid in env.possible_agents
                )
                print(f"  step {step:>4}  rewards: {reward_str}  goals: {goals_str}")

            if all(terms.values()) or all(truncs.values()):
                print(f"  Episode done at step {step}.")
                break

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

        print(f"  Final rewards: { {aid: f'{r:.1f}' for aid, r in total_rewards.items()} }")
        print(f"  Metrics saved → {args.db}  (episode_id={episode_id})")

        if frames:
            gif_path = f"{args.out}.gif"
            imageio.mimsave(gif_path, frames, fps=4)
            print(f"  Saved {gif_path}")

    conn.close()
    env.close()


if __name__ == "__main__":
    main()