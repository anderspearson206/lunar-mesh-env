import os
import ray
import torch
import numpy as np
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig


from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN
from models.action_mask_model import TorchActionMaskModel 


ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
MODEL_PATHS = {
        'k2_model': '/home/patorrad/Documents/lunar-mesh-env/RadioLunaDiff/pretrained_models_network/k2unet/best_k2_model.pth',
        'pmnet_model': '/home/patorrad/Documents/lunar-mesh-env/RadioLunaDiff/pretrained_models_network/pmnet/best_pm_model.pt',
        'diffusion_model': '/home/patorrad/Documents/lunar-mesh-env/RadioLunaDiff/pretrained_models_network/diffusion'
    }

def env_creator(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hm = np.load(config.get("hm_path", ""))
    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS, 
        heightmap=hm,
        env_width=256,
        env_height=256,
        dummy_mode=False,
        device=device
    )
    
    if hasattr(radio_model, 'k2_model'):
        radio_model.k2_model.to(device)
    if hasattr(radio_model, 'pmnet_model'):
        radio_model.pmnet_model.to(device)

    raw_env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", ""), 
        radio_model=radio_model,
        num_agents=config.get("num_agents", 3)
    )
    env = ParallelPettingZooEnv(raw_env)

    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space = raw_env.action_space(raw_env.possible_agents[0])
    return env

register_env("lunar_mesh_v1", env_creator)



DATA_ROOT = '/mnt/2ndSSD/rm_raw_for_network'

HM_PATH = f'{DATA_ROOT}/hm/hm_18.npy' 

test_env = env_creator({"hm_path": HM_PATH, "num_agents": 3})
obs_space = test_env.observation_space
print(f"Observation Space: {obs_space}")
act_space = test_env.action_space
print(f"Action Space: {act_space}")
test_env.close()

config = (
    PPOConfig()
    .environment("lunar_mesh_v1", env_config={"num_agents": 3, 
                                              "hm_path": HM_PATH})
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    .training(
        model={"custom_model": "action_mask_model", 
               "_disable_preprocessor_api": True,
               },
        train_batch_size=2000,
        lr=5e-5,
    )
    .multi_agent(
        policies={
            "shared_policy": (None, obs_space, act_space, {})
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .env_runners(num_env_runners=2, 
                 num_gpus_per_env_runner=0.2) 
    .resources(num_gpus=0.5 if torch.cuda.is_available() else 0)
)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            name="lunar_mesh_experiment",
            stop={"training_iteration": 100},
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=10)
        ),
        param_space=config.to_dict(),
    )
    
    results = tuner.fit()
    ray.shutdown()