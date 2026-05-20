"""
Quick sanity check: run 1 RLlib PPO iteration with real radio NN and print
whether episode_reward_mean is NaN or a real number.

Run: python examples/test_rllib_reward.py
Expected output (if working): [OK] episode_reward_mean = <number>
"""

import os, sys, torch, numpy as np

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN
from models.action_mask_model import TorchActionMaskModel

DATA_ROOT   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH     = f"{DATA_ROOT}/hm/hm_18.npy"
_PRETRAINED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}
NUM_AGENTS = 3
EP_MAX     = 100   # short episode so we always get completed episodes


def env_creator(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env_creator] device={device}")
    hm = np.load(HM_PATH)
    rm = RadioMapModelNN(MODEL_PATHS, hm, 256, 256,
                         dummy_mode=config.get("dummy_mode", False),
                         device=device)
    raw = LunarRoverMeshEnv(hm_path=HM_PATH, radio_model=rm,
                             num_agents=NUM_AGENTS, seed=19)
    raw.EP_MAX_TIME = EP_MAX
    env = ParallelPettingZooEnv(raw)
    env.observation_space = raw.observation_spaces[raw.possible_agents[0]]
    env.action_space      = raw.action_space(raw.possible_agents[0])
    return env


ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
register_env("test_env", env_creator)

probe = env_creator({"dummy_mode": False})
obs_space = probe.observation_space
act_space = probe.action_space
probe.close()

ray.init(ignore_reinit_error=True,
         runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath(
             os.path.join(os.path.dirname(__file__), ".."))}})

algo = (
    PPOConfig()
    .environment("test_env", env_config={"dummy_mode": False})
    .framework("torch")
    .api_stack(enable_rl_module_and_learner=False,
               enable_env_runner_and_connector_v2=False)
    .training(model={"custom_model": "action_mask_model",
                     "_disable_preprocessor_api": True},
              train_batch_size=300)   # small: 3× EP_MAX steps
    .multi_agent(policies={"shared_policy": (None, obs_space, act_space, {})},
                 policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy")
    .env_runners(num_env_runners=1,
                 num_gpus_per_env_runner=1,
                 rollout_fragment_length=100,
                 sample_timeout_s=3600)
    .resources(num_gpus=0)
    .build()
)

print("Running 1 training iteration...")
result = algo.train()

rew = result.get("episode_reward_mean")
if rew is None or (isinstance(rew, float) and np.isnan(rew)):
    rew = result.get("env_runners", {}).get("episode_reward_mean")

if rew is None or (isinstance(rew, float) and np.isnan(float(rew))):
    print(f"[FAIL] episode_reward_mean = {rew}  (still NaN)")
else:
    print(f"[OK]   episode_reward_mean = {rew:.2f}")

algo.stop()
ray.shutdown()
