# run_mpc_simulation.py

import argparse
import numpy as np
import imageio
import torch
import sys
import os
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN
from lunar_mesh_env.mpc_planner import MPPIPlanner


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_comm_actions(agent_id, env):
    """
    Greedy epidemic communication: send to all connected neighbors and BS.
    This is reactive (not planned) — MPC only plans movement.
    """
    agent = env.agent_map[agent_id]
    num_rovers = len(env.possible_agents)
    comm_actions = [0] * (num_rovers + 1)

    if len(agent.payload_manager.buffer) > 0:
        self_index = None
        for i, target_id in enumerate(env.possible_agents):
            if target_id == agent_id:
                self_index = i
                continue

            target_agent = env.agent_map[target_id]
            if target_agent in agent.neighbors:
                next_packet = agent.payload_manager.buffer[0]
                if target_agent.id not in next_packet.touched:
                    comm_actions[i] = 1

        # Send to self to preserve buffer (epidemic style)
        if any(c == 1 for c in comm_actions) and self_index is not None:
            comm_actions[self_index] = 1

    # Always offload to BS when connected
    if agent.bs_connected:
        comm_actions[-1] = 1

    return comm_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Draw MPPI rollouts on the simulation view')
    args = parser.parse_args()

    SEED = 1886
    set_seed(SEED)

    # DATA_ROOT = '/mnt/2ndSSD/rm_raw_for_network'
    DATA_ROOT = '/home/paolo/Documents/NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy'

    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }

    print(f"Loading Neural Network Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device}")

    try:
        global_hm = np.load(HM_PATH)
        print("Terrain loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Could not find {HM_PATH}. Using flat terrain.")
        global_hm = np.zeros((256, 256))

    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=global_hm,
        env_width=256,
        env_height=256,
        num_inference_steps=4,
        dummy_mode=False,
        device=device
    )

    RADIO_BIAS = 0.0
    NUM_AGENTS = 1

    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        render_mode="rgb_array",
        radio_bias=RADIO_BIAS,
        seed=SEED
    )

    # Create one MPPI planner per rover (shared radio model + heightmap)
    planners = {}
    for agent_id in env.possible_agents:
        planners[agent_id] = MPPIPlanner(
            radio_model=radio_model,
            heightmap=env.heightmap,
            env_width=env.width,
            env_height=env.height,
            horizon=15,
            num_samples=128,
            temperature=1.0,
            noise_sigma=0.8,
            max_dist_per_step=env.MAX_DIST_PER_STEP,
            max_incline=env.MAX_INCLINE_PER_STEP,
            min_dbm=env.MIN_DBM_THRESHOLD,
            w_comm=0.0,
            w_bs=0.25,
            w_goal=1.0,
            w_energy=0.0,
            agent=env.agent_map[agent_id],
            dt_s=0.1,
            bs_decay_rate=0.5,
        )

    if args.debug:
        env.debug_planners = planners

    obs, info = env.reset()
    frames = []

    print(f"\n--- Starting MPC Simulation ---")
    print(f"Agents: {env.possible_agents}")
    print(f"MPPI: horizon=15, samples=128, sigma=0.8")

    SIM_STEPS = 300
    start = time.time()

    for step in range(SIM_STEPS):
        actions = {}

        for agent_id in env.agents:
            agent = env.agent_map[agent_id]

            # Gather other rover positions
            other_positions = [
                (env.agent_map[aid].x, env.agent_map[aid].y)
                for aid in env.agents if aid != agent_id
            ]

            # MPPI plans the movement action
            move_action = planners[agent_id].plan(
                agent_pos=(agent.x, agent.y),
                goal_pos=(agent.goal_x, agent.goal_y),
                other_positions=other_positions,
                bs_pos=(env.base_station.x, env.base_station.y),
            )

            # Greedy communication
            comm_actions = get_comm_actions(agent_id, env)

            actions[agent_id] = [move_action] + comm_actions

        obs, rewards, terms, truncs, infos = env.step(actions)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if step % 10 == 0:
            avg_rew = sum(rewards.values()) / max(len(rewards), 1)
            elapsed = time.time() - start
            bs_unique = len(env.base_station.packets_received)
            print(
                f"Step {step:4d} | Avg Reward: {avg_rew:7.2f} | "
                f"BS Pkts: {bs_unique:4d} | "
                f"Time: {elapsed:.1f}s"
            )

        if not env.agents:
            print("All agents terminated.")
            break

    env.close()
    total_time = time.time() - start

    # --- Dashboard Summary ---
    bs_state = env.base_station.get_state()
    bs_unique = len(env.base_station.packets_received)
    packets_generated = sum(
        env.agent_map[aid].payload_manager.get_state().get('num_packets_generated', 0)
        for aid in env.possible_agents if aid in env.agent_map
    )

    # Grab weights from first planner (all planners share the same config)
    sample_planner = next(iter(planners.values()))

    print(f"\n{'='*60}")
    print(f"  Simulation completed in {total_time:.2f}s ({SIM_STEPS} steps)")
    print(f"{'='*60}")
    print(f"\n  MPPI WEIGHTS")
    print(f"    w_comm        : {sample_planner.w_comm}")
    print(f"    w_bs          : {sample_planner.w_bs}")
    print(f"    w_goal        : {sample_planner.w_goal}")
    print(f"    w_energy      : {sample_planner.w_energy}")
    print(f"    bs_decay_rate : {sample_planner.bs_decay_rate}")
    print(f"\n  BASE STATION")
    print(f"    Unique Packets Received : {bs_unique}")
    print(f"    Total Packets Received  : {bs_state['num_packets_received']}")
    print(f"    Duplicates Received     : {bs_state['num_duplicates_received']}")
    print(f"    Packets Generated       : {packets_generated}")

    print(f"\n  {'Agent':<8} {'Buf%':>6} {'Stored':>8} {'Gen':>6} {'Goals':>8} {'Energy':>8} {'Dist (m)':>10}")
    print(f"  {'-'*56}")
    for i, agent_id in enumerate(env.possible_agents):
        if agent_id in env.agent_map:
            agent = env.agent_map[agent_id]
            dtn = agent.payload_manager.get_state()
            buf_pct = (dtn['payload_size'] / dtn['buffer_size']) * 100
            goals_done = env.agent_goals_completed.get(agent_id, 0)
            label = chr(ord('A') + i)
            print(f"  {label:<8} {buf_pct:>5.1f}% {dtn.get('num_packets', 0):>8} "
                  f"{dtn.get('num_packets_generated', 0):>6} {goals_done:>3}/{env.num_goals:<4} "
                  f"{agent.energy:>8.0f} {agent.total_distance:>10.1f}")
    print()

    # --- Save to results.md ---
    results_path = os.path.join(os.path.dirname(__file__), 'results_bs_decay_rate.md')
    with open(results_path, 'a') as f:
        f.write(f"## Run: seed={SEED}, agents={NUM_AGENTS}, steps={SIM_STEPS}, radio_bias={RADIO_BIAS}\n\n")
        f.write(f"### MPPI Weights\n")
        f.write(f"| w_comm | w_bs | w_goal | w_energy | bs_decay_rate |\n")
        f.write(f"|--------|------|--------|----------|---------------|\n")
        f.write(f"| {sample_planner.w_comm} | {sample_planner.w_bs} | {sample_planner.w_goal} | {sample_planner.w_energy} | {sample_planner.bs_decay_rate} |\n\n")
        f.write(f"### Base Station\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Unique Packets Received | {bs_unique} |\n")
        f.write(f"| Total Packets Received | {bs_state['num_packets_received']} |\n")
        f.write(f"| Duplicates Received | {bs_state['num_duplicates_received']} |\n")
        f.write(f"| Packets Generated | {packets_generated} |\n\n")
        f.write(f"### Agents\n")
        f.write(f"| Agent | Buf% | Stored | Gen | Goals | Energy | Dist (m) |\n")
        f.write(f"|-------|------|--------|-----|-------|--------|----------|\n")
        for i, agent_id in enumerate(env.possible_agents):
            if agent_id in env.agent_map:
                agent = env.agent_map[agent_id]
                dtn = agent.payload_manager.get_state()
                buf_pct = (dtn['payload_size'] / dtn['buffer_size']) * 100
                goals_done = env.agent_goals_completed.get(agent_id, 0)
                label = chr(ord('A') + i)
                f.write(f"| {label} | {buf_pct:.1f}% | {dtn.get('num_packets', 0)} | "
                        f"{dtn.get('num_packets_generated', 0)} | {goals_done}/{env.num_goals} | "
                        f"{agent.energy:.0f} | {agent.total_distance:.1f} |\n")
        f.write(f"\nTime: {total_time:.2f}s\n\n---\n\n")
    print(f"Results appended to {results_path}")

    if frames:
        out_name = f'marl_mpc_{SEED}_{sample_planner.w_comm}_{sample_planner.w_bs}_{sample_planner.w_goal}_{sample_planner.w_energy}_{sample_planner.bs_decay_rate}_rss_200.gif'
        print(f"Saving replay to {out_name} ({len(frames)} frames)...")
        imageio.mimsave(out_name, frames, fps=10)
        imageio.mimwrite(
            f'marl_mpc_{SEED}_{RADIO_BIAS}.mp4',
            frames, fps=10,
            output_params=['-vcodec', 'libx264']
        )
        print("Done!")


if __name__ == "__main__":
    main()

# The core issue is that w_bs rewards a position (being near BS) while w_goal rewards progress (moving toward goal). A position reward always wins over a progress reward at equilibrium because the rover earns it every step for doing nothing.

# A few ways to fix this:

# Make BS a progress reward too — reward getting closer to BS, not absolute throughput
# Only reward BS when the agent has packets — no incentive to stay if the buffer is empty
# Decay the BS reward over time — diminishing returns for staying in the same spot
