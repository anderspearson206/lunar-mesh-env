"""
AlphaStar-style PPO fine-tuning for LunarRoverMeshEnv.

Pipeline (see also collect_demos.py and train_sl_policy.py):
    1. collect_demos.py      -> results/demos/*.npz       (A* + comm heuristic)
    2. train_sl_policy.py    -> results/sl_policy.pt      (supervised pretraining)
    3. train_ppo_alphastar.py --sl-weights results/sl_policy.pt
                              -> PPO fine-tuning with a KL(pi || pi_SL) anchor

The policy is TorchAlphaStarModel: CNN encoders over screen/minimap views
(built by SpatialObsWrapper) + autoregressive masked action heads
(TorchARMaskedDistribution). Radio maps are served from the precomputed
lookup array, as in train_ppo_lookup.py.

Usage:
    python examples/train_ppo_alphastar.py --sl-weights results/sl_policy.pt
    python examples/train_ppo_alphastar.py            # from scratch, no anchor
"""

import argparse
import os
import sys
import numpy as np
import torch

import ray
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lunar_mesh_env import LunarRoverMeshEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from lunar_mesh_env.spatial_features import SpatialObsWrapper
from models.alphastar_model import TorchAlphaStarModel
from models.ar_action_dist import TorchARMaskedDistribution

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH   = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "radio_maps_hm_18.npy")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_AGENTS       = 3
TRAIN_BATCH_SIZE = 4000
LR               = 1e-4     # SL-initialized; higher than the 5e-5 scratch LR
ENTROPY_COEFF    = 0.01     # SL init replaces exploration bootstrapping
KL_COEFF_SL      = 0.05
NUM_ENV_RUNNERS  = 6
TOTAL_ITERATIONS = 100
CHECKPOINT_FREQ  = 25
EXPERIMENT_NAME  = "lunar_mesh_ppo_alphastar"
WANDB_PROJECT    = "lunar-mesh-rl"

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

class RandomLayoutWrapper(SpatialObsWrapper):
    """Draw a fresh env seed before each reset.

    LunarRoverMeshEnv.reset derives agent spawns, goal sequences, and BS
    placement from env.seed, so a fixed seed trains on a single layout every
    episode. The spatial CNN policy memorizes that layout and fails to
    generalize (the old 8-input MLP could not, so this never mattered before).
    """

    def __init__(self, raw_env, base_seed):
        super().__init__(raw_env)
        self._layout_rng = np.random.RandomState(base_seed)

    def reset(self, seed=None, options=None):
        self.env.seed = int(self._layout_rng.randint(0, 1_000_000))
        return super().reset(seed=seed, options=options)


def env_creator(config):
    hm = np.load(config.get("hm_path", HM_PATH))
    radio_model = RadioMapModelLookup(
        maps_path=config.get("maps_path", MAPS_PATH),
        heightmap=hm,
        env_width=256,
        env_height=256,
    )
    raw_env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
        seed=19,
    )
    raw_env.EP_MAX_TIME = config.get("max_episode_steps", 250)

    if config.get("randomize_layout", True):
        # different stream per rollout worker
        worker_index = getattr(config, "worker_index", 0)
        wrapped = RandomLayoutWrapper(raw_env, base_seed=10_000 + worker_index)
    else:
        wrapped = SpatialObsWrapper(raw_env)
    env = ParallelPettingZooEnv(wrapped)
    env.observation_space = wrapped.observation_space(raw_env.possible_agents[0])
    env.action_space      = wrapped.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RewardLoggerCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm=None, result, **_):
        it = result.get("training_iteration", 0)

        rew = result.get("episode_reward_mean")
        if rew is None or (isinstance(rew, float) and np.isnan(rew)):
            rew = result.get("env_runners", {}).get("episode_reward_mean")

        env_runners = result.get("env_runners", {})
        eps = (result.get("episodes_this_iter")
               or env_runners.get("num_episodes")
               or env_runners.get("num_episodes_lifetime"))

        sl_kl = (result.get("info", {}).get("learner", {})
                 .get("shared_policy", {}).get("model", {}).get("sl_kl"))
        sl_kl_str = f"  sl_kl={sl_kl:.4f}" if sl_kl is not None else ""
        print(f"[iter {it:>4}]  episode_reward_mean={rew}  episodes={eps}{sl_kl_str}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
ModelCatalog.register_custom_model("alphastar_model", TorchAlphaStarModel)
ModelCatalog.register_custom_action_dist("ar_masked_dist", TorchARMaskedDistribution)
register_env("lunar_mesh_alphastar_v1", env_creator)

# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

def build_config(obs_space, act_space, args):
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("WARNING: CUDA not available — learner runs on CPU.")

    custom_model_config = {
        "sl_weights":  args.sl_weights,
        "kl_coeff_sl": args.kl_coeff_sl if args.sl_weights else 0.0,
    }

    return (
        PPOConfig()
        .environment(
            "lunar_mesh_alphastar_v1",
            env_config={
                "num_agents":        NUM_AGENTS,
                "hm_path":           HM_PATH,
                "maps_path":         args.maps,
                "max_episode_steps": 250,
                "randomize_layout":  not args.fixed_layout,
            },
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            model={
                "custom_model":              "alphastar_model",
                "custom_action_dist":        "ar_masked_dist",
                "custom_model_config":       custom_model_config,
                "_disable_preprocessor_api": True,
            },
            train_batch_size=args.batch_size,
            lr=LR,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=ENTROPY_COEFF,
            num_sgd_iter=6,
            grad_clip=1.0,
            minibatch_size=256,
            # The AR dist's kl() conditions on a freshly sampled chain while
            # prev_action_dist is rebuilt with current head weights, so the
            # adaptive-KL signal is stale. Rely on ratio clipping instead
            # (ACTION_LOGP stored at sampling time is exact).
            kl_coeff=0.0,
        )
        .callbacks(RewardLoggerCallback)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
        )
        .env_runners(
            num_env_runners=args.runners,
            num_gpus_per_env_runner=0,
            rollout_fragment_length="auto",
            sample_timeout_s=180,
        )
        .resources(num_gpus=1 if use_gpu else 0)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=MAPS_PATH,
                        help="Precomputed radio maps .npy")
    parser.add_argument("--sl-weights", default=None,
                        help="SL-pretrained weights from train_sl_policy.py "
                             "(results/sl_policy.pt). Omit to train from scratch.")
    parser.add_argument("--kl-coeff-sl", type=float, default=KL_COEFF_SL,
                        help="Coefficient of the KL(pi || pi_SL) anchor "
                             f"(default: {KL_COEFF_SL}; only used with --sl-weights)")
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--restore", default=None, metavar="CHECKPOINT_DIR")
    parser.add_argument("--iterations", type=int, default=TOTAL_ITERATIONS)
    parser.add_argument("--runners", type=int, default=NUM_ENV_RUNNERS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--fixed-layout", action="store_true",
                        help="Train on the single seed-19 layout (old behavior) "
                             "instead of randomizing start/goal/BS per episode")
    args = parser.parse_args()

    if not os.path.exists(args.maps):
        print(f"ERROR: maps file not found: {args.maps}")
        print("Run examples/precompute_radio_maps.py first.")
        raise SystemExit(1)
    if args.sl_weights and not os.path.exists(args.sl_weights):
        print(f"ERROR: SL weights not found: {args.sl_weights}")
        raise SystemExit(1)
    if args.sl_weights:
        args.sl_weights = os.path.abspath(args.sl_weights)

    probe_env = env_creator({"hm_path": HM_PATH, "num_agents": NUM_AGENTS,
                             "max_episode_steps": 250, "maps_path": args.maps})
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
        # See train_ppo_lookup.py: mmap'd radio maps double-count in RSS.
        _system_config={"memory_usage_threshold": 0.99},
    )

    config = build_config(obs_space, act_space, args)

    param_space = config.to_dict()
    if args.restore:
        param_space["restore"] = os.path.expanduser(args.restore)
        print(f"Warm-starting from: {param_space['restore']}")

    reporter = CLIReporter(
        metric_columns={
            "training_iteration":  "iter",
            "episode_reward_mean": "rew_mean",
            "episode_reward_min":  "rew_min",
            "episode_reward_max":  "rew_max",
            "episodes_this_iter":  "eps",
        },
        max_progress_rows=5,
        print_intermediate_tables=True,
    )

    run_callbacks = []
    if not args.no_wandb:
        from ray.air.integrations.wandb import WandbLoggerCallback
        run_callbacks.append(WandbLoggerCallback(
            project=WANDB_PROJECT, name=args.name, log_config=False))

    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            name=args.name,
            stop={"training_iteration": args.iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
            ),
            callbacks=run_callbacks,
            progress_reporter=reporter,
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"\nBest checkpoint: {best.checkpoint}")

    ray.shutdown()
