"""
PPO training for LunarRoverMeshEnv using RLlib.

Uses a shared policy across all agents (parameter sharing) with
an action-mask model to handle invalid actions.

Usage:
    python examples/train_ppo_rllib.py

Adjust DATA_ROOT and MODEL_PATHS to point to your data.
"""

import os
import sys
import torch
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

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN
from models.action_mask_model import TorchActionMaskModel

# ---------------------------------------------------------------------------
# Paths – edit these to match your setup
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"  # <-- EDIT THIS
HM_PATH = f"{DATA_ROOT}/hm/hm_18.npy"

_PRETRAINED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}

    

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
NUM_AGENTS = 3
TRAIN_BATCH_SIZE = 4000
LR = 5e-5
NUM_ENV_RUNNERS = 1
TOTAL_ITERATIONS = 200
CHECKPOINT_FREQ = 20
EXPERIMENT_NAME = "lunar_mesh_ppo"
WANDB_PROJECT   = "lunar-mesh-rl"

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def env_creator(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env_creator] device={device}  cuda_available={torch.cuda.is_available()}  dummy={config.get('dummy_mode', False)}")
    hm = np.load(config.get("hm_path", HM_PATH))

    use_dummy = config.get("dummy_mode", False)
    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=hm,
        env_width=256,
        env_height=256,
        dummy_mode=use_dummy,
        device=device,
    )

    if not use_dummy:
        if hasattr(radio_model, "k2_model"):
            radio_model.k2_model.to(device)
        if hasattr(radio_model, "pmnet_model"):
            radio_model.pmnet_model.to(device)

    raw_env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
        seed=19,
    )
    raw_env.EP_MAX_TIME = config.get("max_episode_steps", 500)

    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space = raw_env.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Callbacks – explicit W&B logging that works regardless of metric path
# ---------------------------------------------------------------------------

class RewardLoggerCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm=None, result, **_):
        it = result.get("training_iteration", 0)

        # Dig reward out of wherever Ray 2.x put it
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
                       "episodes_this_iter": eps or 0}, step=it)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
register_env("lunar_mesh_v1", env_creator)

# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

def build_config(obs_space, act_space):
    return (
        PPOConfig()
        .environment(
            "lunar_mesh_v1",
            env_config={"num_agents": NUM_AGENTS, "hm_path": HM_PATH,
                        "max_episode_steps": 500, "dummy_mode": False},
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            model={
                "custom_model": "action_mask_model",
                "_disable_preprocessor_api": True,
            },
            train_batch_size=TRAIN_BATCH_SIZE,
            lr=LR,
            # PPO-specific knobs
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.05,         # raised: prevents premature collapse to idle
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
            num_gpus_per_env_runner=1,      # give the GPU to the worker for radio-NN inference
            rollout_fragment_length=200,
            sample_timeout_s=3600,
        )
        .resources(num_gpus=0)              # SGD runs on CPU; policy net is tiny (4→256→9)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Probe observation/action spaces before launching Ray workers
    probe_env = env_creator({"hm_path": HM_PATH, "num_agents": NUM_AGENTS, "max_episode_steps": 500, "dummy_mode": False})
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    print(f"Observation space : {obs_space}")
    print(f"Action space      : {act_space}")
    probe_env.close()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": repo_root}},
    )

    config = build_config(obs_space, act_space)

    reporter = CLIReporter(
        metric_columns={
            "training_iteration":   "iter",
            "episode_reward_mean":  "rew_mean",
            "episode_reward_min":   "rew_min",
            "episode_reward_max":   "rew_max",
            "episodes_this_iter":   "eps",
            "policy_reward_mean/shared_policy": "policy_rew",
        },
        max_progress_rows=5,
        print_intermediate_tables=True,
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            name=EXPERIMENT_NAME,
            stop={"training_iteration": TOTAL_ITERATIONS},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=CHECKPOINT_FREQ,
                checkpoint_at_end=True,
            ),
            callbacks=[
                WandbLoggerCallback(
                    project=WANDB_PROJECT,
                    name=EXPERIMENT_NAME,
                    log_config=False,
                )
            ],
            progress_reporter=reporter,
        ),
        param_space=config.to_dict(),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"\nBest checkpoint: {best.checkpoint}")

    ray.shutdown()
