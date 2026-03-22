# run_optuna_study.py
# Optuna hyperparameter optimization for MPPI cost weights.

import argparse
import json
import numpy as np
import optuna
import os
import random
import sys
import time
import torch

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
    """Greedy epidemic communication: send to all connected neighbors and BS."""
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
        if any(c == 1 for c in comm_actions) and self_index is not None:
            comm_actions[self_index] = 1

    if agent.bs_connected:
        comm_actions[-1] = 1

    return comm_actions


def run_simulation(seed, w_comm, w_bs, w_goal, w_energy, bs_decay_rate,
                   sim_steps, num_agents, radio_model, hm_path, trial=None):
    """
    Run a single simulation with the given weights and return metrics.

    Args:
        trial: optional Optuna trial for intermediate reporting/pruning
    Returns:
        dict with goals_completed, total_goals, unique_pkts, pkts_generated,
             total_distance, energy_remaining
    """
    set_seed(seed)

    env = LunarRoverMeshEnv(
        hm_path=hm_path,
        radio_model=radio_model,
        num_agents=num_agents,
        render_mode=None,  # no rendering for speed
        radio_bias=0.0,
        seed=seed,
    )

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
            w_comm=w_comm,
            w_bs=w_bs,
            w_goal=w_goal,
            w_energy=w_energy,
            agent=env.agent_map[agent_id],
            dt_s=0.1,
            bs_decay_rate=bs_decay_rate,
        )

    obs, info = env.reset()

    for step in range(sim_steps):
        actions = {}
        for agent_id in env.agents:
            agent = env.agent_map[agent_id]
            other_positions = [
                (env.agent_map[aid].x, env.agent_map[aid].y)
                for aid in env.agents if aid != agent_id
            ]
            move_action = planners[agent_id].plan(
                agent_pos=(agent.x, agent.y),
                goal_pos=(agent.goal_x, agent.goal_y),
                other_positions=other_positions,
                bs_pos=(env.base_station.x, env.base_station.y),
            )
            comm_actions = get_comm_actions(agent_id, env)
            actions[agent_id] = [move_action] + comm_actions

        obs, rewards, terms, truncs, infos = env.step(actions)

        # Report intermediate progress for pruning
        if trial is not None and step % 50 == 0 and step > 0:
            goals_so_far = sum(env.agent_goals_completed.values())
            trial.report(goals_so_far, step)
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

        if not env.agents:
            break

    env.close()

    goals_completed = sum(env.agent_goals_completed.values())
    total_goals = env.num_goals * num_agents
    unique_pkts = len(env.base_station.packets_received)
    pkts_generated = sum(
        env.agent_map[aid].payload_manager.get_state().get('num_packets_generated', 0)
        for aid in env.possible_agents if aid in env.agent_map
    )
    total_distance = sum(
        env.agent_map[aid].total_distance
        for aid in env.possible_agents if aid in env.agent_map
    )
    energy_remaining = sum(
        env.agent_map[aid].energy
        for aid in env.possible_agents if aid in env.agent_map
    )

    return {
        'goals_completed': goals_completed,
        'total_goals': total_goals,
        'unique_pkts': unique_pkts,
        'pkts_generated': pkts_generated,
        'total_distance': total_distance,
        'energy_remaining': energy_remaining,
    }


def make_objective(mode, seeds, sim_steps, num_agents, radio_model, hm_path):
    """Create an Optuna objective function closure."""

    def objective(trial):
        w_comm = trial.suggest_float("w_comm", 0.0, 2.0)
        w_bs = trial.suggest_float("w_bs", 0.0, 1.0)
        w_goal = trial.suggest_float("w_goal", 0.5, 5.0)
        w_energy = trial.suggest_float("w_energy", 0.0, 1.0)
        bs_decay_rate = trial.suggest_float("bs_decay_rate", 0.3, 1.0)

        goal_ratios = []
        delivery_ratios = []

        for seed in seeds:
            metrics = run_simulation(
                seed=seed,
                w_comm=w_comm, w_bs=w_bs, w_goal=w_goal,
                w_energy=w_energy, bs_decay_rate=bs_decay_rate,
                sim_steps=sim_steps, num_agents=num_agents,
                radio_model=radio_model, hm_path=hm_path,
                trial=trial if len(seeds) == 1 else None,
            )
            goal_ratios.append(metrics['goals_completed'] / metrics['total_goals'])
            gen = max(metrics['pkts_generated'], 1)
            delivery_ratios.append(metrics['unique_pkts'] / gen)

        avg_goal = sum(goal_ratios) / len(goal_ratios)
        avg_delivery = sum(delivery_ratios) / len(delivery_ratios)

        # Log extra metrics as trial attributes
        trial.set_user_attr("avg_goal_ratio", avg_goal)
        trial.set_user_attr("avg_delivery_ratio", avg_delivery)

        if mode == 'goals':
            return avg_goal
        elif mode == 'throughput':
            return avg_delivery
        else:  # both
            return avg_goal, avg_delivery

    return objective


def print_results(study, mode):
    """Print study results summary."""
    print(f"\n{'='*60}")
    print(f"  Optuna Study Complete")
    print(f"{'='*60}")
    print(f"  Trials completed: {len(study.trials)}")

    if mode == 'both':
        pareto = study.best_trials
        print(f"\n  Pareto-optimal trials: {len(pareto)}")
        print(f"\n  {'Trial':>6} {'Goals':>8} {'Delivery':>10} | {'w_comm':>7} {'w_bs':>7} {'w_goal':>7} {'w_energy':>8} {'decay':>7}")
        print(f"  {'-'*75}")
        for t in sorted(pareto, key=lambda t: t.values[0], reverse=True):
            p = t.params
            print(f"  {t.number:>6} {t.values[0]:>8.3f} {t.values[1]:>10.3f} | "
                  f"{p['w_comm']:>7.3f} {p['w_bs']:>7.3f} {p['w_goal']:>7.3f} "
                  f"{p['w_energy']:>8.3f} {p['bs_decay_rate']:>7.3f}")
        best_params = [
            {"trial": t.number, "values": list(t.values), "params": t.params}
            for t in pareto
        ]
    else:
        best = study.best_trial
        print(f"\n  Best trial: #{best.number}")
        print(f"  Value: {best.value:.4f}")
        print(f"\n  Best params:")
        for k, v in best.params.items():
            print(f"    {k:<15}: {v:.4f}")
        goal_r = best.user_attrs.get('avg_goal_ratio', 'N/A')
        deliv_r = best.user_attrs.get('avg_delivery_ratio', 'N/A')
        print(f"\n  Goal ratio:     {goal_r}")
        print(f"  Delivery ratio: {deliv_r}")
        best_params = {"trial": best.number, "value": best.value, "params": best.params}

    # Save best params
    out_path = os.path.join(os.path.dirname(__file__), 'optuna_best_params.json')
    with open(out_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Best params saved to {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Optuna optimization for MPPI cost weights")
    parser.add_argument('--objective', choices=['goals', 'throughput', 'both'],
                        default='both', help='Optimization objective')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--n-seeds', type=int, default=3, help='Seeds per trial for averaging')
    parser.add_argument('--sim-steps', type=int, default=300, help='Simulation steps per run')
    parser.add_argument('--num-agents', type=int, default=1, help='Number of rovers')
    parser.add_argument('--study-name', default='mppi_weights', help='Optuna study name')
    parser.add_argument('--storage', default=None,
                        help='Optuna storage URL (default: sqlite in examples/)')
    args = parser.parse_args()

    if args.storage is None:
        db_path = os.path.join(os.path.dirname(__file__), 'optuna_study.db')
        args.storage = f"sqlite:///{db_path}"

    seeds = list(range(args.n_seeds))  # [0, 1, 2, ...]

    # --- Load models once ---
    DATA_ROOT = '/home/paolo/Documents/NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
    HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy'
    MODEL_PATHS = {
        'k2_model': '../RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '../RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '../RadioLunaDiff/pretrained_models_network/diffusion'
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading radio model on {device}...")

    try:
        global_hm = np.load(HM_PATH)
    except FileNotFoundError:
        print(f"Warning: {HM_PATH} not found. Using flat terrain.")
        global_hm = np.zeros((256, 256))

    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=global_hm,
        env_width=256,
        env_height=256,
        num_inference_steps=4,
        dummy_mode=False,
        device=device,
    )
    print("Radio model loaded.\n")

    # --- Create study ---
    if args.objective == 'both':
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            directions=['maximize', 'maximize'],
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction='maximize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )

    objective_fn = make_objective(
        mode=args.objective,
        seeds=seeds,
        sim_steps=args.sim_steps,
        num_agents=args.num_agents,
        radio_model=radio_model,
        hm_path=HM_PATH,
    )

    print(f"Starting Optuna study: {args.study_name}")
    print(f"  Objective : {args.objective}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Seeds/trial: {args.n_seeds} {seeds}")
    print(f"  Sim steps : {args.sim_steps}")
    print(f"  Agents    : {args.num_agents}")
    print(f"  Storage   : {args.storage}")
    est_minutes = args.n_trials * args.n_seeds * args.sim_steps / 300 * 2.2 / 60
    print(f"  Est. time : ~{est_minutes:.0f} min\n")

    start = time.time()
    study.optimize(objective_fn, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"\nOptimization completed in {elapsed / 60:.1f} min")
    print_results(study, args.objective)


if __name__ == "__main__":
    main()
