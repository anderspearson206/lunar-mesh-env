"""
PPO + GAT-MARL baseline for LunarRoverMeshEnv.

Implements the architecture from:
  Lozano-Cuadra et al., arXiv:2510.20436
  "Learning Decentralized Routing Policies via Graph Attention-based
   Multi-Agent Reinforcement Learning in Lunar Delay-Tolerant Networks"

Key differences from train_ppo_lookup.py:
  - Observation: graph-structured (node features + adjacency) instead of
    flat scalars + 256×256 radio/terrain maps.
  - Model: 2-layer GAT (8 heads → 512 → 64) with centralized value head.
  - Same PPO algorithm and hyperparameters otherwise (fair comparison).

Node features (7-dim, NODE_FEAT_DIM):
  [is_self, is_lander, bs_connected, buffer_fill,
   dist_to_lander_norm, x_norm, y_norm]

Node ordering in the subgraph (always):
  0          : observing agent (self)
  1 .. N-2   : other rovers, sorted by ue_id
  N-1        : base station (lander)

Usage:
    python examples/train_ppo_gat.py
    python examples/train_ppo_gat.py --maps /path/to/radio_maps_hm_18.npy
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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from lunar_mesh_env.marl_env_gat import LunarRoverMeshGATEnv
from models.gat_model import TorchGATModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT  = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH    = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "radio_maps_hm_18.npy")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_AGENTS       = 3
TRAIN_BATCH_SIZE = 4000
LR               = 5e-5
NUM_ENV_RUNNERS  = 6
TOTAL_ITERATIONS = 50
CHECKPOINT_FREQ  = 50
EXPERIMENT_NAME  = "lunar_mesh_ppo_gat"
WANDB_PROJECT    = "lunar-mesh-rl"


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def env_creator(config):
    hm          = np.load(config.get("hm_path", HM_PATH))
    maps_path   = config.get("maps_path", MAPS_PATH)
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
            wandb.log({"episode_reward_mean": rew, "episodes_this_iter": eps or 0}, step=it)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
ModelCatalog.register_custom_model("gat_model", TorchGATModel)
register_env("lunar_mesh_gat_v1", env_creator)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def build_config(obs_space, act_space, maps_path: str):
    return (
        PPOConfig()
        .environment(
            "lunar_mesh_gat_v1",
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
                "custom_model":             "gat_model",
                "_disable_preprocessor_api": True,
            },
            train_batch_size=TRAIN_BATCH_SIZE,
            lr=LR,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.05,
            num_sgd_iter=10,
            grad_clip=1.0,
            minibatch_size=512,
        )
        .callbacks(RewardLoggerCallback)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(
            num_env_runners=NUM_ENV_RUNNERS,
            num_gpus_per_env_runner=0,
            rollout_fragment_length="auto",
            sample_timeout_s=120,
        )
        .resources(num_gpus=0)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=MAPS_PATH)
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--restore", default=None, metavar="CHECKPOINT_DIR")
    parser.add_argument("--iterations", type=int, default=TOTAL_ITERATIONS)
    args     = parser.parse_args()
    run_name = args.name

    if not os.path.exists(args.maps):
        print(f"ERROR: maps file not found: {args.maps}")
        print("Run examples/precompute_radio_maps.py first.")
        raise SystemExit(1)

    probe_env = env_creator({
        "hm_path": HM_PATH, "num_agents": NUM_AGENTS,
        "max_episode_steps": 500, "maps_path": args.maps,
    })
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

    config      = build_config(obs_space, act_space, args.maps)
    param_space = config.to_dict()
    if args.restore:
        param_space["restore"] = os.path.expanduser(args.restore)
        print(f"Warm-starting from: {param_space['restore']}")

    reporter = CLIReporter(
        metric_columns={
            "training_iteration":               "iter",
            "episode_reward_mean":              "rew_mean",
            "episode_reward_min":               "rew_min",
            "episode_reward_max":               "rew_max",
            "episodes_this_iter":               "eps",
            "policy_reward_mean/shared_policy": "policy_rew",
        },
        max_progress_rows=5,
        print_intermediate_tables=True,
    )

    tuner = tune.Tuner(
        "PPO",
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
    best    = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"\nBest checkpoint: {best.checkpoint}")
    ray.shutdown()