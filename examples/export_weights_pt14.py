"""
Export PPO policy weights in a format loadable by PyTorch 1.4.

PyTorch >= 1.6 defaults to a zip-based checkpoint format that PyTorch 1.4
cannot read. This script loads an RLlib checkpoint, extracts the model
state_dict, and re-saves it using the legacy pickle format.

Usage
-----
    python examples/export_weights_pt14.py
    python examples/export_weights_pt14.py --checkpoint /path/to/checkpoint_dir
    python examples/export_weights_pt14.py --out policy_weights.pth
"""

import argparse
import os
import sys

import numpy as np
import torch

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env import LunarRoverMeshEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from models.action_mask_model import TorchActionMaskModel

DATA_ROOT  = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH    = f"{DATA_ROOT}/hm/hm_18.npy"
MAPS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radio_maps_hm_18.npy")
NUM_AGENTS = 3

DEFAULT_CHECKPOINT = (
    "/home/paolo/ray_results/spedup_3agents_goal+ill001_comm1"
    "/PPO_lunar_mesh_lookup_v1_2b004_00000_0_2026-06-06_15-29-23"
    "/checkpoint_000000"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "policy_weights_pt14.pth"
)


def env_creator(config):
    hm = np.load(config.get("hm_path", HM_PATH))
    radio_model = RadioMapModelLookup(
        maps_path=config.get("maps_path", MAPS_PATH),
        heightmap=hm, env_width=256, env_height=256,
    )
    raw_env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
        seed=19,
    )
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space      = raw_env.action_space(raw_env.possible_agents[0])
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to RLlib checkpoint directory")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output .pth file path")
    args = parser.parse_args()

    ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
    register_env("lunar_mesh_lookup_v1", env_creator)
    ray.init(ignore_reinit_error=True, num_gpus=0)

    probe = env_creator({"hm_path": HM_PATH, "maps_path": MAPS_PATH,
                         "num_agents": NUM_AGENTS})
    obs_space = probe.observation_space
    act_space = probe.action_space
    probe.close()

    algo = (
        PPOConfig()
        .environment("lunar_mesh_lookup_v1",
                     env_config={"hm_path": HM_PATH, "maps_path": MAPS_PATH,
                                 "num_agents": NUM_AGENTS})
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .training(model={"custom_model": "action_mask_model",
                         "_disable_preprocessor_api": True})
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
        )
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
        .build()
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    algo.restore(args.checkpoint)
    policy = algo.get_policy("shared_policy")
    state_dict = policy.model.state_dict()

    print(f"Model keys: {list(state_dict.keys())}")
    print(f"Input layer shape: {state_dict['internal_model.0.weight'].shape}")

    # Save in legacy format readable by PyTorch 1.4.
    # _use_new_zipfile_serialization=False forces the old pickle-based format
    # (PyTorch >= 1.6 defaults to zip; PyTorch 1.4 cannot open zip files).
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state_dict, args.out, _use_new_zipfile_serialization=False)
    print(f"Saved → {args.out}  ({os.path.getsize(args.out) / 1024:.1f} KB)")

    # Verify it round-trips correctly.
    loaded = torch.load(args.out, map_location="cpu")
    for k in state_dict:
        assert torch.equal(state_dict[k], loaded[k]), f"Mismatch on key {k}"
    print("Round-trip verification passed.")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()