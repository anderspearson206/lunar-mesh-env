"""
DDQN training with Lozano-style single-edge comm + rate-fill forwarding.

Implements the GAT-DDQN algorithm from:
  "Learning Decentralized Routing Policies via Graph Attention-based
  Multi-Agent Reinforcement Learning in Lunar Delay-Tolerant Networks"
  — Federico Lozano-Cuadra

Same env and model as train_ppo_lozano.py.  Train both and compare
bs_unique_packets in eval_ppo_checkpoint.py --model lozano.

Key DQN settings vs PPO:
  - Off-policy with replay buffer (50K transitions)
  - Double-Q targets (double_q=True) — matches Lozano's DDQN
  - Epsilon-greedy exploration: 1.0 → 0.02 over 200K steps
  - n_step=3 for multi-step TD targets
  - Target network updated every 500 env steps

Usage:
    python examples/train_dqn_lozano.py
    python examples/train_dqn_lozano.py --name my_dqn_run --iterations 200
"""

import argparse
import os
import sys
import numpy as np

import wandb
import ray
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lunar_mesh_env.marl_env_lozano import LunarRoverMeshLozanoEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from models.mlp_gat_lozano_model import TorchMLPGATLozanoModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT  = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH    = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "radio_maps_hm_18.npy")

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
NUM_AGENTS         = 3
LR                 = 1e-4
NUM_ENV_RUNNERS    = 4
TOTAL_ITERATIONS   = 200
CHECKPOINT_FREQ    = 50
EXPERIMENT_NAME    = "lunar_mesh_ddqn_lozano"
WANDB_PROJECT      = "lunar-mesh-rl"

# DQN-specific
REPLAY_BUFFER_SIZE       = 50_000
TRAIN_BATCH_SIZE         = 32
TARGET_UPDATE_FREQ       = 500      # env steps between target-network syncs
N_STEP                   = 3
EPSILON_TIMESTEPS        = 200_000  # steps for epsilon 1.0 → 0.02

# ---------------------------------------------------------------------------
# Environment factory (same as train_ppo_lozano.py)
# ---------------------------------------------------------------------------

def env_creator(config):
    hm = np.load(config.get("hm_path", HM_PATH))
    radio_model = RadioMapModelLookup(
        maps_path=config.get("maps_path", MAPS_PATH),
        heightmap=hm,
        env_width=256,
        env_height=256,
    )
    raw_env = LunarRoverMeshLozanoEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
        packet_mode='rate',
        action_type='discrete',
        seed=19,
    )
    raw_env.EP_MAX_TIME = config.get("max_episode_steps", 250)
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space      = raw_env.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RewardLoggerCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm=None, result, **_):
        it  = result.get("training_iteration", 0)
        rew = result.get("episode_reward_mean")
        if rew is None or (isinstance(rew, float) and np.isnan(rew)):
            rew = result.get("env_runners", {}).get("episode_reward_mean")
        env_runners = result.get("env_runners", {})
        eps = (result.get("episodes_this_iter")
               or env_runners.get("num_episodes")
               or env_runners.get("num_episodes_lifetime"))
        print(f"[iter {it:>4}]  episode_reward_mean={rew}  episodes={eps}")
        if wandb.run is not None and rew is not None and not np.isnan(float(rew)):
            wandb.log({"episode_reward_mean": rew,
                       "episodes_this_iter":  eps or 0}, step=it)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
ModelCatalog.register_custom_model("mlp_gat_lozano_model", TorchMLPGATLozanoModel)
register_env("lunar_mesh_lozano_v1", env_creator)

# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

def build_config(obs_space, act_space, maps_path: str):
    return (
        DQNConfig()
        .environment(
            "lunar_mesh_lozano_v1",
            env_config={
                "num_agents":        NUM_AGENTS,
                "hm_path":           HM_PATH,
                "maps_path":         maps_path,
                "max_episode_steps": 250,
            },
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            model={
                "custom_model":              "mlp_gat_lozano_model",
                "_disable_preprocessor_api": True,
            },
            hiddens=[],                             # model outputs Q-values directly; no extra DQN head
            double_q=True,                          # DDQN — matches Lozano's paper
            target_network_update_freq=TARGET_UPDATE_FREQ,
            replay_buffer_config={
                "type":     "MultiAgentPrioritizedReplayBuffer",
                "capacity": REPLAY_BUFFER_SIZE,
            },
            train_batch_size=TRAIN_BATCH_SIZE,
            n_step=N_STEP,
            lr=LR,
            gamma=0.99,
            grad_clip=1.0,
            epsilon=[[0, 1.0], [EPSILON_TIMESTEPS, 0.02]],
        )
        .callbacks(RewardLoggerCallback)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(
            num_env_runners=NUM_ENV_RUNNERS,
            num_gpus_per_env_runner=0,
            rollout_fragment_length=4,
            sample_timeout_s=120,
        )
        .resources(num_gpus=0)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=MAPS_PATH,
                        help="Path to precomputed radio maps .npy")
    parser.add_argument("--name", default=EXPERIMENT_NAME,
                        help="Run name for W&B and Ray experiment dir")
    parser.add_argument("--restore", default=None, metavar="CHECKPOINT_DIR",
                        help="Warm-start from an existing checkpoint directory")
    parser.add_argument("--iterations", type=int, default=TOTAL_ITERATIONS,
                        help=f"Number of training iterations (default: {TOTAL_ITERATIONS})")
    args = parser.parse_args()
    run_name = args.name

    if not os.path.exists(args.maps):
        print(f"ERROR: maps file not found: {args.maps}")
        print("Run examples/precompute_radio_maps.py first.")
        raise SystemExit(1)

    probe_env = env_creator({"hm_path": HM_PATH, "num_agents": NUM_AGENTS,
                              "max_episode_steps": 500, "maps_path": args.maps})
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    print(f"Observation space : {obs_space}")
    print(f"Action space      : {act_space}")
    probe_env.close()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ray_temp  = os.path.expanduser("~/ray_temp")
    os.makedirs(ray_temp, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": repo_root}},
        _temp_dir=ray_temp,
        _system_config={"memory_usage_threshold": 0.99},
    )

    config = build_config(obs_space, act_space, args.maps)
    param_space = config.to_dict()
    if args.restore:
        param_space["restore"] = os.path.expanduser(args.restore)
        print(f"Warm-starting from: {param_space['restore']}")

    reporter = CLIReporter(
        metric_columns={
            "training_iteration":               "iter",
            "episode_reward_mean":              "rew_mean",
            "episodes_this_iter":               "eps",
            "policy_reward_mean/shared_policy": "policy_rew",
            "info/learner/shared_policy/mean_td_error": "td_err",
        },
        max_progress_rows=5,
        print_intermediate_tables=True,
    )

    tuner = tune.Tuner(
        "DQN",
        run_config=tune.RunConfig(
            name=run_name,
            stop={"training_iteration": args.iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=CHECKPOINT_FREQ,
                checkpoint_at_end=True,
            ),
            callbacks=[
                WandbLoggerCallback(
                    project=WANDB_PROJECT,
                    name=run_name,
                    log_config=False,
                )
            ],
            progress_reporter=reporter,
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"\nBest checkpoint: {best.checkpoint}")
    ray.shutdown()
