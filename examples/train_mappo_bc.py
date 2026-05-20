"""
BC pre-training + MAPPO fine-tuning for LunarRoverMeshEnv.

Pipeline
--------
Phase 1 – Expert data   : roll out A* navigation episodes, write SampleBatches.
Phase 2 – BC pre-train  : imitate A* movement to warm-start the shared actor.
Phase 3 – MAPPO fine-tune: centralized critic sees global state (all agent
           positions, goals, energy); decentralised actors see local obs only.

Usage
-----
    # Full pipeline
    python examples/train_mappo_bc.py

    # Skip data collection (already done)
    python examples/train_mappo_bc.py --skip-expert

    # Skip BC, jump straight to MAPPO (no warm-start)
    python examples/train_mappo_bc.py --skip-bc

    # Resume from a specific BC checkpoint
    python examples/train_mappo_bc.py --skip-expert --bc-checkpoint /path/to/ckpt
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.offline.json_writer import JsonWriter
import gymnasium as gym

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH     = f"{DATA_ROOT}/hm/hm_18.npy"
_PRETRAINED = os.path.join(_REPO_ROOT, "RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}

EXPERT_DATA_DIR  = "/tmp/lunar_mesh_expert"
BC_WEIGHTS_PATH  = "/tmp/lunar_mesh_bc_weights.pt"   # actor weights saved after BC

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_AGENTS          = 3
NUM_EXPERT_EPISODES = 200
BC_ITERATIONS       = 50
MAPPO_ITERATIONS    = 200
TRAIN_BATCH_SIZE    = 4000
CHECKPOINT_FREQ     = 20

# EM illumination reward (MAPPO fine-tuning only)
# Reward = delta_coverage * EM_REWARD_SCALE, shared equally across agents.
# Coverage = fraction of 256×256 cells where at least one agent's signal ≥ threshold.
CONNECT_THRESHOLD_DBM = -82.0   # matches the connectivity threshold in marl_env.py
EM_REWARD_SCALE       = 5.0     # tune relative to navigation rewards (~2–100 range)

# ---------------------------------------------------------------------------
# Obs / action layout  (all dims assume NUM_AGENTS = 3)
# ---------------------------------------------------------------------------
# Scalar obs keys (excludes terrain, radio_map, action_mask)
_SCALAR_KEYS = [
    "energy",                    # (1,)
    "position",                  # (2,)
    "buffer_usage",              # (1,)
    "num_packets",               # (1,)
    "other_agent_vectors",       # (NUM_AGENTS+1, 2)
    "other_agent_connectivity",  # (NUM_AGENTS+1,)
    "goal_vector",               # (2,)
]
SCALAR_DIM = 1 + 2 + 1 + 1 + (NUM_AGENTS + 1) * 2 + (NUM_AGENTS + 1) + 2  # = 19
MASK_DIM   = 9 + 2 * (NUM_AGENTS + 1)                                        # = 17
# Centralized critic input: pos(2) + goal(2) + energy(1) per agent
GLOBAL_DIM = NUM_AGENTS * 5                                                  # = 15
# Flat MAPPO obs layout: [action_mask | scalars | global_state]
MAPPO_OBS_DIM = MASK_DIM + SCALAR_DIM + GLOBAL_DIM                          # = 51


def _compute_em_coverage(raw_env: LunarRoverMeshEnv) -> float:
    """
    Fraction of the 256×256 map covered by at least one agent's EM footprint.
    Uses generate_map_batch (cached by int position) so repeated calls at the
    same location are free after the first inference.
    """
    positions = [
        (raw_env.agent_map[aid].x, raw_env.agent_map[aid].y)
        for aid in raw_env.possible_agents
    ]
    maps = raw_env.radio_model.generate_map_batch(positions, "5.8")
    # maps: (N_agents, 256, 256) dBm — take the best signal per cell
    union_map = np.max(maps, axis=0)
    return float((union_map >= CONNECT_THRESHOLD_DBM).mean())


def _flatten_scalars(obs: dict) -> np.ndarray:
    """Flatten local scalar obs keys into a 1-D float32 array."""
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).flatten() for k in _SCALAR_KEYS]
    )


def _build_global_state(raw_env: LunarRoverMeshEnv) -> np.ndarray:
    """Compact global state for the centralized critic: all agents' pos/goal/energy."""
    parts = []
    for aid in raw_env.possible_agents:
        a = raw_env.agent_map[aid]
        parts += [
            a.x / raw_env.width,
            a.y / raw_env.height,
            (a.goal_x - a.x) / raw_env.width,
            (a.goal_y - a.y) / raw_env.height,
            a.energy / raw_env.START_ENERGY,
        ]
    return np.array(parts, dtype=np.float32)


# ---------------------------------------------------------------------------
# A* expert policy helpers
# ---------------------------------------------------------------------------

def _nav_path_to_move(agent) -> int:
    """Return the discrete move action (0-8) that follows the agent's nav_path."""
    if not agent.nav_path or len(agent.nav_path) < 2:
        return 0  # idle

    nx, ny = agent.nav_path[1]
    dx = nx - agent.x
    dy = ny - agent.y
    adx, ady = abs(dx), abs(dy)

    if adx < 0.5 and ady < 0.5:
        return 0
    if adx < ady * 0.4:
        return 1 if dy > 0 else 2   # N / S
    if ady < adx * 0.4:
        return 4 if dx > 0 else 3   # E / W
    if dx > 0 and dy > 0: return 5  # NE
    if dx < 0 and dy > 0: return 6  # NW
    if dx > 0 and dy < 0: return 7  # SE
    return 8                        # SW


def _expert_action(agent_id: str, raw_env: LunarRoverMeshEnv) -> np.ndarray:
    """A* move action + send to BS if connected; skip peer comm during BC."""
    agent = raw_env.agent_map[agent_id]
    move  = _nav_path_to_move(agent)
    comm  = np.zeros(NUM_AGENTS + 1, dtype=np.int64)
    comm[-1] = 1 if agent.bs_connected else 0   # last slot = BS
    return np.array([move, *comm], dtype=np.int64)


# ---------------------------------------------------------------------------
# Phase 1: Expert trajectory collection
# ---------------------------------------------------------------------------

def collect_expert_data():
    """
    Roll out A* navigation episodes and write per-agent SampleBatches to
    EXPERT_DATA_DIR for offline BC training.

    Uses dummy_mode=True for the radio model so no GPU/VRAM is needed here.
    The BC policy learns terrain-only navigation; EM awareness is added in MAPPO.
    """
    os.makedirs(EXPERT_DATA_DIR, exist_ok=True)
    writer = JsonWriter(EXPERT_DATA_DIR)

    hm = np.load(HM_PATH)
    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=hm,
        env_width=256,
        env_height=256,
        dummy_mode=True,            # skip loading radio NNs
        device=torch.device("cpu"),
    )

    for ep in range(NUM_EXPERT_EPISODES):
        raw_env = LunarRoverMeshEnv(
            hm_path=HM_PATH,
            radio_model=radio_model,
            num_agents=NUM_AGENTS,
            seed=ep,
        )
        obs, _ = raw_env.reset()

        buffers = {aid: defaultdict(list) for aid in raw_env.possible_agents}
        active  = list(raw_env.agents)
        t = 0

        while active:
            actions = {aid: _expert_action(aid, raw_env) for aid in active}
            next_obs, rewards, terms, truncs, _ = raw_env.step(actions)

            for aid in list(active):
                done = terms.get(aid, False) or truncs.get(aid, False)
                buffers[aid][SampleBatch.OBS].append(_flatten_scalars(obs[aid]))
                buffers[aid][SampleBatch.ACTIONS].append(actions[aid])
                buffers[aid][SampleBatch.REWARDS].append(float(rewards.get(aid, 0.0)))
                buffers[aid][SampleBatch.DONES].append(done)
                buffers[aid][SampleBatch.NEXT_OBS].append(
                    _flatten_scalars(next_obs.get(aid, obs[aid]))
                )
                buffers[aid]["eps_id"].append(ep * NUM_AGENTS
                                              + raw_env.possible_agents.index(aid))
                buffers[aid]["t"].append(t)
                if done:
                    active.remove(aid)

            obs    = next_obs
            active = [a for a in active if a in raw_env.agents]
            t     += 1

        for aid, buf in buffers.items():
            if buf[SampleBatch.OBS]:
                writer.write(SampleBatch({k: np.array(v) for k, v in buf.items()}))

        raw_env.close()
        if (ep + 1) % 20 == 0:
            print(f"  Expert episodes: {ep + 1}/{NUM_EXPERT_EPISODES}")

    print(f"Expert data written to {EXPERT_DATA_DIR}")


# ---------------------------------------------------------------------------
# Shared actor model  (used for BC; weights transferred to MAPPO actor)
# ---------------------------------------------------------------------------

class BCActorModel(TorchModelV2, nn.Module):
    """
    Simple MLP on compact scalar obs.
    Trained offline with BC, then its weights seed the MAPPO actor.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_dim = model_config.get("custom_model_config", {}).get("scalar_dim", SCALAR_DIM)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head  = nn.Linear(256, 1)
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float()
        # Guard: if preprocessor appended extra dims, trim to scalar features
        if x.shape[-1] > SCALAR_DIM:
            x = x[:, :SCALAR_DIM]
        feat = self.trunk(x)
        self._value = self.value_head(feat).squeeze(1)
        return self.policy_head(feat), state

    def value_function(self):
        return self._value


# ---------------------------------------------------------------------------
# MAPPO model  (decentralised actor + centralized critic)
# ---------------------------------------------------------------------------

class MAPPOModel(TorchModelV2, nn.Module):
    """
    Flat obs layout expected from MAPPOEnvWrapper:
        [action_mask (MASK_DIM) | scalar_features (SCALAR_DIM) | global_state (GLOBAL_DIM)]

    Actor  – reads scalar_features only (local, decentralised).
    Critic – reads global_state (centralized during training).

    BC weights are injected into actor_trunk / actor_policy_head via a
    DefaultCallbacks hook before MAPPO training starts.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg        = model_config.get("custom_model_config", {})
        scalar_dim = cfg.get("scalar_dim", SCALAR_DIM)
        global_dim = cfg.get("global_dim", GLOBAL_DIM)
        self._mask_dim   = cfg.get("mask_dim", MASK_DIM)
        self._scalar_dim = scalar_dim

        # Decentralised actor
        self.actor_trunk = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.actor_policy_head = nn.Linear(256, num_outputs)

        # Centralized critic
        self.critic_trunk = nn.Sequential(
            nn.Linear(global_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
        )
        self.critic_value_head = nn.Linear(128, 1)

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        flat = input_dict["obs_flat"].float()

        mask        = flat[:, :self._mask_dim]
        scalar_feat = flat[:, self._mask_dim: self._mask_dim + self._scalar_dim]
        global_feat = flat[:, self._mask_dim + self._scalar_dim:]

        # Actor (decentralised)
        logits = self.actor_policy_head(self.actor_trunk(scalar_feat))
        inf_mask = torch.clamp(torch.log(mask), min=-1e9)

        # Centralized critic
        self._value = self.critic_value_head(
            self.critic_trunk(global_feat)
        ).squeeze(1)

        return logits + inf_mask, state

    def value_function(self):
        return self._value


# ---------------------------------------------------------------------------
# Callback: inject BC actor weights into MAPPO model at init
# ---------------------------------------------------------------------------

class BCInitCallback(DefaultCallbacks):
    """Load saved BC actor weights into the MAPPO actor trunk on algorithm init."""

    def on_algorithm_init(self, algorithm, **kwargs):
        if not os.path.exists(BC_WEIGHTS_PATH):
            return
        bc_weights = torch.load(BC_WEIGHTS_PATH, map_location="cpu")
        policy = algorithm.get_policy("shared_policy")
        model  = policy.model

        # Map BCActorModel keys → MAPPOModel actor keys
        mapping = {
            "trunk.0.weight":     "actor_trunk.0.weight",
            "trunk.0.bias":       "actor_trunk.0.bias",
            "trunk.2.weight":     "actor_trunk.2.weight",
            "trunk.2.bias":       "actor_trunk.2.bias",
            "policy_head.weight": "actor_policy_head.weight",
            "policy_head.bias":   "actor_policy_head.bias",
        }
        own_state = model.state_dict()
        n_loaded  = 0
        for bc_key, mappo_key in mapping.items():
            if bc_key in bc_weights and mappo_key in own_state:
                own_state[mappo_key].copy_(bc_weights[bc_key])
                n_loaded += 1
        model.load_state_dict(own_state)
        print(f"[BCInitCallback] Loaded {n_loaded} BC actor tensors into MAPPO model.")


# ---------------------------------------------------------------------------
# MAPPO env wrapper: augments obs with global state
# ---------------------------------------------------------------------------

class MAPPOEnvWrapper(ParallelPettingZooEnv):
    """
    Flattens each agent's obs and appends a compact global state vector.
    Output per agent: [action_mask | scalars | global_state]  (float32, Box)

    Also injects a shared EM-illumination reward: agents are jointly rewarded
    for increasing the fraction of the terrain covered by their EM footprints.
    """

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_coverage = _compute_em_coverage(self.env)
        return self._augment(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = super().step(actions)

        coverage     = _compute_em_coverage(self.env)
        em_reward    = (coverage - self._prev_coverage) * EM_REWARD_SCALE
        self._prev_coverage = coverage

        for aid in rew:
            rew[aid] += em_reward   # shared cooperative signal

        return self._augment(obs), rew, term, trunc, info

    def _augment(self, obs_dict: dict) -> dict:
        gs = _build_global_state(self.env)   # self.env = LunarRoverMeshEnv
        result = {}
        for aid, agent_obs in obs_dict.items():
            mask    = np.asarray(agent_obs["action_mask"], dtype=np.float32)
            scalars = _flatten_scalars(agent_obs)
            result[aid] = np.concatenate([mask, scalars, gs])
        return result


def mappo_env_creator(config):
    hm = np.load(config.get("hm_path", HM_PATH))
    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=hm,
        env_width=256,
        env_height=256,
        dummy_mode=False,
        device=torch.device("cpu"),   # workers use CPU; GPU reserved for trainer
    )
    raw_env = LunarRoverMeshEnv(
        hm_path=config.get("hm_path", HM_PATH),
        radio_model=radio_model,
        num_agents=config.get("num_agents", NUM_AGENTS),
    )
    env = MAPPOEnvWrapper(raw_env)
    env.observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(MAPPO_OBS_DIM,), dtype=np.float32
    )
    env.action_space = raw_env.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Phase 2: BC pre-training
# ---------------------------------------------------------------------------

def bc_pretrain() -> str:
    """
    Train BCActorModel on expert JSON data. Returns path to BC weights file.
    Uses compact scalar obs only – keeps offline data small and training fast.
    """
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(SCALAR_DIM,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([9] + [2] * (NUM_AGENTS + 1))

    ModelCatalog.register_custom_model("bc_actor", BCActorModel)

    config = (
        BCConfig()
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            input_=EXPERT_DATA_DIR,
            model={
                "custom_model": "bc_actor",
                "custom_model_config": {"scalar_dim": SCALAR_DIM},
            },
            train_batch_size=512,
            lr=1e-3,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda aid, *args, **kwargs: "shared_policy",
        )
    )

    ray.init(ignore_reinit_error=True)
    tuner = tune.Tuner(
        "BC",
        run_config=tune.RunConfig(
            name="lunar_mesh_bc",
            stop={"training_iteration": BC_ITERATIONS},
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
        ),
        param_space=config.to_dict(),
    )
    results = tuner.fit()

    # Extract and save actor weights so the MAPPO callback can load them
    best = results.get_best_result(metric="training_iteration", mode="max")
    bc_algo = config.build()
    bc_algo.restore(best.checkpoint)
    bc_weights = bc_algo.get_policy("shared_policy").model.state_dict()
    torch.save(bc_weights, BC_WEIGHTS_PATH)
    bc_algo.stop()

    ray.shutdown()
    print(f"BC actor weights saved to {BC_WEIGHTS_PATH}")
    return BC_WEIGHTS_PATH


# ---------------------------------------------------------------------------
# Phase 3: MAPPO fine-tuning
# ---------------------------------------------------------------------------

def mappo_finetune(use_bc_init: bool = True):
    """
    Fine-tune with MAPPO.

    Centralized critic: sees all agents' positions, goals, and energy levels.
    Decentralized actor: sees only local obs (as at execution time).

    If use_bc_init=True and BC_WEIGHTS_PATH exists, the actor is warm-started
    from BC weights via BCInitCallback.
    """
    ModelCatalog.register_custom_model("mappo_model", MAPPOModel)
    register_env("lunar_mesh_mappo", mappo_env_creator)

    probe     = mappo_env_creator({"hm_path": HM_PATH, "num_agents": NUM_AGENTS})
    obs_space = probe.observation_space
    act_space = probe.action_space
    probe.close()

    callbacks = BCInitCallback if use_bc_init else DefaultCallbacks

    config = (
        PPOConfig()
        .environment(
            "lunar_mesh_mappo",
            env_config={"num_agents": NUM_AGENTS, "hm_path": HM_PATH},
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            model={
                "custom_model": "mappo_model",
                "custom_model_config": {
                    "scalar_dim": SCALAR_DIM,
                    "global_dim": GLOBAL_DIM,
                    "mask_dim":   MASK_DIM,
                },
                "_disable_preprocessor_api": True,
            },
            train_batch_size=TRAIN_BATCH_SIZE,
            lr=5e-5,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            num_sgd_iter=10,
            minibatch_size=512,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda aid, *args, **kwargs: "shared_policy",
        )
        .callbacks(callbacks)
        .env_runners(num_env_runners=1, num_gpus_per_env_runner=0)
        .resources(num_gpus=0.5 if torch.cuda.is_available() else 0)
    )

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": _REPO_ROOT}},
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            name="lunar_mesh_mappo",
            stop={"training_iteration": MAPPO_ITERATIONS},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=CHECKPOINT_FREQ,
                checkpoint_at_end=True,
            ),
        ),
        param_space=config.to_dict(),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"\nBest MAPPO checkpoint: {best.checkpoint}")
    ray.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-expert", action="store_true",
        help="Skip Phase 1 (expert data already in EXPERT_DATA_DIR)",
    )
    parser.add_argument(
        "--skip-bc", action="store_true",
        help="Skip Phase 2 BC pre-training",
    )
    parser.add_argument(
        "--bc-checkpoint", type=str, default=None,
        help="Path to existing BC checkpoint (implies --skip-bc)",
    )
    args = parser.parse_args()

    if not args.skip_expert:
        print("=== Phase 1: Collecting A* expert trajectories ===")
        collect_expert_data()

    skip_bc = args.skip_bc or (args.bc_checkpoint is not None)
    if not skip_bc:
        print("\n=== Phase 2: BC pre-training ===")
        bc_pretrain()

    use_bc = os.path.exists(BC_WEIGHTS_PATH)
    print(f"\n=== Phase 3: MAPPO fine-tuning (BC init: {use_bc}) ===")
    mappo_finetune(use_bc_init=use_bc)
