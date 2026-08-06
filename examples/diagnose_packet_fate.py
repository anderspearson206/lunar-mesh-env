"""
Diagnose the fate of every packet over one A* episode.

Prints a breakdown: generated / delivered-unique / expired / still-in-buffer,
per agent and in total.  Runs A* navigation with epidemic routing so we see
the best-case delivery the current scenario allows.

Usage
-----
    cd /home/paolo/Documents/lunar-mesh-env
    conda run -n lunar_mesh python examples/diagnose_packet_fate.py
    conda run -n lunar_mesh python examples/diagnose_packet_fate.py --routing none --steps 500
"""

import argparse
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lunar_mesh_env import LunarRoverMeshEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from lunar_mesh_env.pathfinding import a_star_search

DATA_ROOT = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH   = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH = os.path.join(os.path.dirname(__file__), "radio_maps_hm_18.npy")
NUM_AGENTS = 3


def compute_path(env, agent_id):
    agent = env.agent_map[agent_id]
    path = a_star_search(
        env.heightmap,
        (int(agent.x), int(agent.y)),
        (int(agent.goal_x), int(agent.goal_y)),
        max_incline=env.MAX_INCLINE_PER_STEP,
    )
    return path if path else []


def astar_action(env, agent_id):
    move = env.heuristic_move_action(agent_id)
    num_rovers = len(env.possible_agents)
    return [move] + [0] * (num_rovers + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--routing", default="epidemic",
                        choices=["none", "epidemic", "spray_and_wait"])
    parser.add_argument("--seed", type=int, default=199)
    parser.add_argument("--ttl", type=int, default=None,
                        help="Packet TTL in steps (default: env default of 50)")
    args = parser.parse_args()

    hm = np.load(HM_PATH)
    radio_model = RadioMapModelLookup(
        maps_path=MAPS_PATH, heightmap=hm, env_width=256, env_height=256
    )
    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        routing_protocol=args.routing,
        seed=args.seed,
    )
    if args.ttl is not None:
        env.PACKET_TTL = args.ttl
    env.EP_MAX_TIME = args.steps

    obs, _ = env.reset()
    for aid in env.agents:
        env.agent_map[aid].nav_path = compute_path(env, aid)

    prev_goals = {aid: (env.agent_map[aid].goal_x, env.agent_map[aid].goal_y)
                  for aid in env.possible_agents}

    bs_connected_steps = {aid: 0 for aid in env.possible_agents}

    for step in range(args.steps):
        actions = {aid: astar_action(env, aid) for aid in env.agents}
        obs, rewards, terms, truncs, _ = env.step(actions)

        for aid in env.agents:
            if env.agent_map[aid].bs_connected:
                bs_connected_steps[aid] += 1
            agent = env.agent_map[aid]
            new_goal = (agent.goal_x, agent.goal_y)
            if new_goal != prev_goals.get(aid):
                agent.nav_path = compute_path(env, aid)
                prev_goals[aid] = new_goal

        if all(terms.values()) or all(truncs.values()):
            break

    # ── Tally ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Routing: {args.routing}   Steps: {step+1}   Seed: {args.seed}   TTL: {env.PACKET_TTL}")
    print(f"{'='*60}")

    total_gen = total_exp = total_buf = 0
    print(f"\n{'Agent':<12} {'Generated':>10} {'Expired':>10} {'In buffer':>10} {'BS-conn steps':>14}")
    print("-" * 60)
    for aid in env.possible_agents:
        pm = env.agent_map[aid].payload_manager
        gen = pm.num_packets_generated
        exp = pm.num_expired
        buf = len(pm.buffer)
        conn = bs_connected_steps[aid]
        print(f"  {aid:<10} {gen:>10} {exp:>10} {buf:>10} {conn:>14}")
        total_gen += gen
        total_exp += exp
        total_buf += buf

    bs = env.base_station
    unique_delivered = len(bs.packets_received)
    total_rcvd = bs.num_packets_received
    duplicates = bs.num_duplicates_received

    print("-" * 60)
    print(f"  {'TOTAL':<10} {total_gen:>10} {total_exp:>10} {total_buf:>10}")

    print(f"\n  BS received (total incl. duplicates): {total_rcvd}")
    print(f"  BS unique packets:                    {unique_delivered}")
    print(f"  BS duplicates:                        {duplicates}")

    accounted = unique_delivered + total_exp + total_buf
    print(f"\n  Packet fate (unique IDs):")
    print(f"    Delivered to BS : {unique_delivered:>6}  ({100*unique_delivered/max(total_gen,1):.1f}%)")
    print(f"    Expired (TTL)   : {total_exp:>6}  ({100*total_exp/max(total_gen,1):.1f}%)")
    print(f"    Still in buffer : {total_buf:>6}  ({100*total_buf/max(total_gen,1):.1f}%)")
    print(f"    Accounted for   : {accounted:>6}  / {total_gen} generated")

    if accounted != total_gen:
        print(f"    *** Gap {total_gen - accounted} — likely cleaned-up acked packets not yet expired ***")

    print(f"\n  Avg BS-connected steps per rover: "
          f"{sum(bs_connected_steps.values())/NUM_AGENTS:.1f} / {step+1}")
    print()

    env.close()


if __name__ == "__main__":
    main()