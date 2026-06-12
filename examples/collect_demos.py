"""
Collect expert demonstrations for SL pretraining (AlphaStar pipeline, phase 1).

Expert policy:
    movement: A* path (lunar_mesh_env/pathfinding.py) followed via
              env.heuristic_move_action (pure pursuit + validity fallback)
    comms:    send to BS when connected and buffer non-empty;
              relay to a peer when the peer is BS-connected and we are not.

Storage is compact: spatial tensors (screen/minimap) are NOT stored — the SL
dataloader reconstructs them from positions via the precomputed radio lookup
and the heightmap (see lunar_mesh_env/spatial_features.build_spatial_obs).

Output: sharded .npz files in results/demos/ with per-sample arrays:
    self_xy(2) goal_xy(2) bs_xy(2) other_xys(2*(N-1)) move_history(4) float32
    action_mask(17) int8   action(5) int8   eps_id int32   t int16

Usage:
    python examples/collect_demos.py --episodes 5      # smoke test
    python examples/collect_demos.py --episodes 1000   # full dataset
"""

import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env import LunarRoverMeshEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from lunar_mesh_env.pathfinding import a_star_search

DATA_ROOT    = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH      = f"{DATA_ROOT}/hm/hm_18.npy"
DEFAULT_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "radio_maps_hm_18.npy")
DEFAULT_OUT  = os.path.join(_REPO_ROOT, "results", "demos")

NUM_AGENTS        = 3
EPISODE_STEPS     = 250
EPISODES_PER_SHARD = 50


# ---------------------------------------------------------------------------
# A* path management (same pattern as eval_astar_baseline.py)
# ---------------------------------------------------------------------------

def compute_path(env, agent_id):
    agent = env.agent_map[agent_id]
    # positions/goals can be drawn up to 256.0 exactly -> clip before rounding
    hi_x, hi_y = int(env.width) - 1, int(env.height) - 1
    start = (int(round(min(agent.x, hi_x))), int(round(min(agent.y, hi_y))))
    goal  = (int(round(min(agent.goal_x, hi_x))), int(round(min(agent.goal_y, hi_y))))
    path  = a_star_search(env.heightmap, start, goal,
                          max_incline=env.MAX_INCLINE_PER_STEP)
    return path if path is not None else []


# ---------------------------------------------------------------------------
# Expert action
# ---------------------------------------------------------------------------

def expert_action(env, agent_id, action_mask):
    """A* movement + heuristic comms, guaranteed consistent with action_mask."""
    agent = env.agent_map[agent_id]
    num_rovers = len(env.possible_agents)
    action = np.zeros(1 + num_rovers + 1, dtype=np.int8)

    move = env.heuristic_move_action(agent_id)
    if action_mask[move] == 0:        # defensive; heuristic checks validity itself
        move = 0
    action[0] = move

    buffer_nonempty = agent.payload_manager.get_state()["num_packets"] > 0
    if buffer_nonempty:
        if agent.bs_connected and action_mask[9 + 2 * num_rovers + 1] == 1:
            # deliver directly to the base station
            action[1 + num_rovers] = 1
        else:
            # one-hop relay: send to a BS-connected peer if reachable
            for j, peer_id in enumerate(env.possible_agents):
                if peer_id == agent_id:
                    continue
                send_bit = 9 + 2 * j + 1
                if action_mask[send_bit] == 1 and env.agent_map[peer_id].bs_connected:
                    action[1 + j] = 1
                    break
    return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=DEFAULT_MAPS)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--start-ep", type=int, default=0,
                        help="Resume collection from this episode index "
                             "(must be a multiple of the shard size)")
    args = parser.parse_args()
    assert args.start_ep % EPISODES_PER_SHARD == 0

    os.makedirs(args.out, exist_ok=True)

    hm = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(
        maps_path=args.maps, heightmap=hm, env_width=256, env_height=256)
    env = LunarRoverMeshEnv(
        hm_path=HM_PATH, radio_model=radio_model,
        num_agents=NUM_AGENTS, seed=args.base_seed)
    env.EP_MAX_TIME = args.steps

    num_rovers = len(env.possible_agents)
    shard, shard_idx = [], args.start_ep // EPISODES_PER_SHARD
    t_start = time.time()
    total_samples = 0

    def flush_shard():
        nonlocal shard, shard_idx
        if not shard:
            return
        keys = shard[0].keys()
        arrays = {k: np.stack([s[k] for s in shard]) for k in keys}
        path = os.path.join(args.out, f"shard_{shard_idx:04d}.npz")
        np.savez_compressed(path, **arrays)
        print(f"  wrote {path}  ({len(shard)} samples)")
        shard, shard_idx = [], shard_idx + 1

    for ep in range(args.start_ep, args.episodes):
        # Vary start/goal/BS placement: reset() draws from RandomState(env.seed + i).
        env.seed = args.base_seed + ep
        obs, _ = env.reset()

        for aid in env.agents:
            env.agent_map[aid].nav_path = compute_path(env, aid)
        prev_goals = {aid: (env.agent_map[aid].goal_x, env.agent_map[aid].goal_y)
                      for aid in env.agents}

        bs_xy = np.array([env.base_station.x, env.base_station.y], dtype=np.float32)

        for t in range(args.steps):
            actions = {}
            for aid in env.agents:
                agent = env.agent_map[aid]
                mask = obs[aid]["action_mask"]
                act = expert_action(env, aid, mask)
                actions[aid] = act

                other_xys = np.array(
                    [[a.x, a.y] for oid, a in env.agent_map.items() if oid != aid],
                    dtype=np.float32).reshape(-1)
                shard.append({
                    "self_xy":      np.array([agent.x, agent.y], dtype=np.float32),
                    "goal_xy":      np.array([agent.goal_x, agent.goal_y], dtype=np.float32),
                    "bs_xy":        bs_xy,
                    "other_xys":    other_xys,
                    "move_history": obs[aid]["move_history"].astype(np.float32),
                    "action_mask":  mask.astype(np.int8),
                    "action":       act,
                    "eps_id":       np.int32(ep),
                    "t":            np.int16(t),
                })
                total_samples += 1

            obs, rewards, terms, truncs, _ = env.step(actions)

            # Recompute A* when a goal changes (new goal after arrival).
            for aid in env.agents:
                agent = env.agent_map[aid]
                new_goal = (agent.goal_x, agent.goal_y)
                if new_goal != prev_goals.get(aid):
                    agent.nav_path = compute_path(env, aid)
                    prev_goals[aid] = new_goal

            if not env.agents:
                break

        if (ep + 1) % EPISODES_PER_SHARD == 0:
            flush_shard()
        if (ep + 1) % 10 == 0:
            rate = (ep + 1) / (time.time() - t_start)
            print(f"episode {ep + 1}/{args.episodes}  "
                  f"samples={total_samples}  ({rate:.1f} eps/s)")

    flush_shard()
    print(f"\nDone: {total_samples} samples in {args.out} "
          f"({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
