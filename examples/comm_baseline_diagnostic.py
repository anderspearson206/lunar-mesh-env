"""
Comm baseline diagnostic — no Ray/checkpoint needed.

Runs N episodes each with three comm policies on the same env:
  1. no_comm    — comm flags always 0 (lower bound)
  2. random     — comm flags sampled uniformly from {0,1}
  3. epidemic   — send to every reachable peer/BS (upper bound)

Navigation is A* for all three (same as training envs).
Metrics are written to the same SQLite DB used by eval_ppo_checkpoint.py.

Usage:
    python examples/comm_baseline_diagnostic.py
    python examples/comm_baseline_diagnostic.py --episodes 10 --steps 250
    python examples/comm_baseline_diagnostic.py --gif --gif-out /tmp/diag
    python examples/comm_baseline_diagnostic.py --db examples/eval_metrics_real_maps_3.db
"""

import argparse
import datetime
import os
import sqlite3
import sys
import random

import imageio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lunar_mesh_env.marl_env_astar_comm import LunarRoverMeshAStarCommEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup

HM_PATH   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2/hm/hm_18.npy"
MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radio_maps_hm_18.npy")
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_metrics_real_maps_3.db")

EVAL_SEED  = 199   # matches eval_ppo_checkpoint.py
NUM_AGENTS = 3


# ---------------------------------------------------------------------------
# Database helpers (mirrors eval_ppo_checkpoint.py exactly)
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    existing_ep = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    if "episodes" in tables and "run_name" not in existing_ep:
        conn.execute("ALTER TABLE episodes ADD COLUMN run_name TEXT")
        conn.commit()
    existing_env = {r[1] for r in conn.execute("PRAGMA table_info(env_step_metrics)").fetchall()}
    if "env_step_metrics" in tables:
        if "bs_unique_packets" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN bs_unique_packets INTEGER")
        if "bs_duplicate_packets" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN bs_duplicate_packets INTEGER")
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
            bs_unique_packets     INTEGER,
            bs_duplicate_packets  INTEGER,
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
    bs = env.base_station
    return (
        episode_id, step,
        int(env.sim_time),
        int(bs.num_packets_received),
        int(len(bs.packets_received)),
        int(bs.num_duplicates_received),
        avg_dr, bs_ratio,
        len(active),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_env(seed=EVAL_SEED, render_mode=None, routing_bw_limit=False):
    hm = np.load(HM_PATH)
    rm = RadioMapModelLookup(maps_path=MAPS_PATH, heightmap=hm, env_width=256, env_height=256)
    return LunarRoverMeshAStarCommEnv(
        hm_path=HM_PATH, radio_model=rm, num_agents=NUM_AGENTS,
        seed=seed, render_mode=render_mode, routing_bw_limit=routing_bw_limit,
    )


# ---------------------------------------------------------------------------
# Comm policies
# ---------------------------------------------------------------------------

def epidemic_action(env, agent_id, base_action):
    agent = env.agent_map[agent_id]
    action = np.array(base_action)
    others = sorted(
        (env.agent_map[a] for a in env.possible_agents if a != agent_id),
        key=lambda a: a.ue_id,
    )
    for i, peer in enumerate(others):
        action[1 + i] = 1 if peer in agent.neighbors else 0
    action[-1] = 1 if agent.bs_connected else 0
    return action


def no_comm_action(env, agent_id, base_action):
    action = np.array(base_action)
    action[1:] = 0
    return action


def random_comm_action(env, agent_id, base_action):
    action = np.array(base_action)
    action[1:] = np.random.randint(0, 2, size=action[1:].shape)
    return action


# ---------------------------------------------------------------------------
# Run one policy
# ---------------------------------------------------------------------------

def run_policy(policy_name, checkpoint_label, action_fn,
               n_episodes, n_steps, seeds,
               conn, gif_path=None, gif_fps=4, routing_bw_limit=False):
    packets_per_ep = []
    comm_send_when_reachable   = []
    comm_send_when_unreachable = []

    for ep_idx, seed in enumerate(seeds):
        capture = gif_path is not None and ep_idx == 0
        env = make_env(seed, render_mode="rgb_array" if capture else None,
                       routing_bw_limit=routing_bw_limit)
        obs, _ = env.reset()

        # Insert episode row
        cur = conn.execute(
            "INSERT INTO episodes (timestamp, checkpoint, num_agents, dummy_mode, seed, run_name) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.datetime.now().isoformat(), checkpoint_label,
             NUM_AGENTS, 0, seed, policy_name),
        )
        conn.commit()
        episode_id = cur.lastrowid

        total_rewards = {aid: 0.0 for aid in env.possible_agents}
        agent_rows = []
        env_rows   = []
        gif_frames = []
        last_step  = 0

        for step in range(n_steps):
            if not env.agents:
                break
            actions = {}
            for aid in env.agents:
                base = env.action_space(aid).sample()
                actions[aid] = action_fn(env, aid, base)

            # Comm stats
            for aid in env.agents:
                agent = env.agent_map[aid]
                others = sorted(
                    (env.agent_map[a] for a in env.possible_agents if a != aid),
                    key=lambda a: a.ue_id,
                )
                for i, peer in enumerate(others):
                    reachable = peer in agent.neighbors
                    sent = int(actions[aid][1 + i])
                    (comm_send_when_reachable if reachable else comm_send_when_unreachable).append(sent)
                bs_reachable = agent.bs_connected
                (comm_send_when_reachable if bs_reachable else comm_send_when_unreachable).append(
                    int(actions[aid][-1])
                )

            obs, rewards, terms, truncs, _ = env.step(actions)

            for aid, r in rewards.items():
                total_rewards[aid] += r
                agent_rows.append(collect_agent_metrics(env, aid, step, r, total_rewards[aid], episode_id))

            env_rows.append(collect_env_metrics(env, step, episode_id))
            last_step = step

            if capture:
                frame = env.render()
                if frame is not None:
                    gif_frames.append(frame)

            if all(terms.values()) or all(truncs.values()):
                break

        packets_per_ep.append(env.base_station.num_packets_received)

        # Write metrics
        conn.executemany(
            "INSERT INTO step_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            agent_rows,
        )
        conn.executemany(
            "INSERT INTO env_step_metrics VALUES (?,?,?,?,?,?,?,?,?)",
            env_rows,
        )
        conn.execute("UPDATE episodes SET total_steps=? WHERE id=?", (last_step + 1, episode_id))
        conn.commit()

        env.close()

        if capture and gif_frames:
            imageio.mimsave(gif_path, gif_frames, fps=gif_fps)

    mean_pkt = np.mean(packets_per_ep)
    std_pkt  = np.std(packets_per_ep)
    send_rate_r = np.mean(comm_send_when_reachable)   if comm_send_when_reachable   else 0.0
    send_rate_u = np.mean(comm_send_when_unreachable) if comm_send_when_unreachable else 0.0

    print(f"\n{'─'*55}")
    print(f"  {policy_name}")
    print(f"{'─'*55}")
    print(f"  BS unique packets  : {mean_pkt:.1f} ± {std_pkt:.1f}  (over {n_episodes} episodes)")
    print(f"  Send rate | reachable peer  : {send_rate_r:.3f}")
    print(f"  Send rate | unreachable peer: {send_rate_u:.3f}")
    if gif_path and gif_frames:
        print(f"  GIF saved → {gif_path}  ({len(gif_frames)} frames)")

    return mean_pkt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps",    type=int, default=250)
    parser.add_argument("--db",       type=str, default=DEFAULT_DB)
    parser.add_argument("--gif",      action="store_true",
                        help="Save a GIF for the first episode of each policy")
    parser.add_argument("--gif-out",  default="diagnostic",
                        help="Output path prefix for GIFs (default: 'diagnostic')")
    parser.add_argument("--fps",      type=int, default=4)
    parser.add_argument("--routing-bw-limit", action="store_true", default=False,
                        help="Constrain epidemic to actual link rate (Vahdat 802.11 model)")
    args = parser.parse_args()

    seeds = [EVAL_SEED + i for i in range(args.episodes)]

    print(f"\nRunning {args.episodes} episodes × {args.steps} steps each  (seeds {seeds[0]}..{seeds[-1]})")
    print("Navigation: A* (same as training)  |  Comm policy varies")
    print(f"Writing metrics → {args.db}")

    conn = init_db(args.db)

    def gif(name):
        return f"{args.gif_out}_{name}.gif" if args.gif else None

    policies = [
        ("no_comm",   "baseline/no_comm",   no_comm_action,     gif("no_comm")),
        ("random",    "baseline/random",    random_comm_action, gif("random")),
        ("epidemic",  "baseline/epidemic",  epidemic_action,    gif("epidemic")),
    ]

    results = {}
    for name, label, fn, gpath in policies:
        results[name] = run_policy(
            name, label, fn,
            args.episodes, args.steps, seeds,
            conn, gif_path=gpath, gif_fps=args.fps,
            routing_bw_limit=args.routing_bw_limit,
        )

    conn.close()

    print(f"\n{'='*55}")
    print(f"  Summary")
    print(f"{'='*55}")
    for name, pkts in results.items():
        print(f"  {name:<10}: {pkts:.1f} pkts")
    print(f"\n  Metrics written → {args.db}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()