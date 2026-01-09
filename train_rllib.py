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

def env_creator(config):
    # use dummy mode for testing
    dummy_radio = RadioMapModelNN(
        model_paths={}, 
        heightmap=np.zeros((256, 256)),
        env_width=256,
        env_height=256,
        dummy_mode=True
    )

    env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", ""), 
        radio_model=dummy_radio,
        num_agents=config.get("num_agents", 3)
    )
    return ParallelPettingZooEnv(env)

register_env("lunar_mesh_v1", env_creator)
DATA_ROOT = '../NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'

HM_PATH = r'C:\Users\Anders\Documents\NASA_DCGR_NETWORKING\radio_data_2\radio_data_2\hm\hm_18.npy'

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
        model={"custom_model": "action_mask_model"},
        train_batch_size=2000,
        lr=5e-5,
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .env_runners(num_env_runners=2) 
    .resources(num_gpus=1 if torch.cuda.is_available() else 0)
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