"""
Test the BC-pretrained policy on LunarRoverMeshEnv.

Loads weights from BC_WEIGHTS_PATH (no Ray required) and rolls out episodes,
reporting per-episode and aggregate metrics.

Usage
-----
    python examples/test_bc_policy.py
    python examples/test_bc_policy.py --weights /tmp/lunar_mesh_bc_weights.pt --episodes 20
    python examples/test_bc_policy.py --render --episodes 3
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import imageio

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN

# ---------------------------------------------------------------------------
# Paths / hyper-params  (mirror train_mappo_bc.py)
# ---------------------------------------------------------------------------
DATA_ROOT   = '/home/paolo/Documents/NASA_DCGR_NETWORKING/radio_data_2/radio_data_2'
HM_PATH     = f"{DATA_ROOT}/hm/hm_18.npy"
_PRETRAINED = os.path.join(_REPO_ROOT, "RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}
DEFAULT_WEIGHTS = "/tmp/lunar_mesh_bc_weights.pt"

NUM_AGENTS = 3
SCALAR_DIM = 19   # 1+2+1+1+(NUM_AGENTS+1)*2+(NUM_AGENTS+1)+2

_SCALAR_KEYS = [
    "energy",
    "position",
    "buffer_usage",
    "num_packets",
    "other_agent_vectors",
    "other_agent_connectivity",
    "goal_vector",
]

# MultiDiscrete layout: [move(9), comm_agent0(2), ..., comm_bs(2)]
_MOVE_CLASSES = 9
_COMM_CLASSES = 2


# ---------------------------------------------------------------------------
# Model (must match BCActorModel in train_mappo_bc.py)
# ---------------------------------------------------------------------------

class BCActorModel(nn.Module):
    def __init__(self, scalar_dim: int, num_outputs: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        feat = self.trunk(x)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_scalars(obs: dict) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[k], dtype=np.float32).flatten() for k in _SCALAR_KEYS]
    )


def _action_from_logits(logits: torch.Tensor, num_agents: int) -> np.ndarray:
    """
    Split logits into one move head + (num_agents+1) comm heads and sample.
    Returns int64 array of shape [1 + num_agents + 1].
    """
    logits = logits.squeeze(0)
    offset = 0

    move_logits = logits[offset: offset + _MOVE_CLASSES]
    move = int(torch.distributions.Categorical(logits=move_logits).sample())
    offset += _MOVE_CLASSES

    comm = []
    for _ in range(num_agents + 1):
        c_logits = logits[offset: offset + _COMM_CLASSES]
        comm.append(int(torch.distributions.Categorical(logits=c_logits).sample()))
        offset += _COMM_CLASSES

    return np.array([move, *comm], dtype=np.int64)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(env: LunarRoverMeshEnv, model: BCActorModel,
                device: torch.device, seed: int, max_steps: int = 300) -> tuple[dict, list]:
    obs, _ = env.reset(seed=seed)
    active = list(env.agents)

    metrics = defaultdict(float)
    metrics["steps"] = 0
    frames = []

    while active:
        actions = {}
        for aid in active:
            scalars = _flatten_scalars(obs[aid])
            if scalars.shape[0] > SCALAR_DIM:
                scalars = scalars[:SCALAR_DIM]
            x = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(x)
            actions[aid] = _action_from_logits(logits, len(env.possible_agents))

        obs, rewards, terms, truncs, info = env.step(actions)

        for aid in active:
            metrics["total_reward"] += float(rewards.get(aid, 0.0))

        active = [a for a in active if a in env.agents and
                  not terms.get(a, False) and not truncs.get(a, False)]
        metrics["steps"] += 1
        if metrics["steps"] >= max_steps:
            break

        frame = env.render()
        if frame is not None:
            frames.append(frame)

    # Pull final stats from the raw env
    goals_done = sum(
        env.agent_goals_completed.get(aid, 0) for aid in env.possible_agents
    )
    total_goals = env.num_goals * len(env.possible_agents)
    metrics["goal_ratio"]      = goals_done / max(total_goals, 1)
    metrics["goals_completed"] = goals_done
    metrics["total_goals"]     = total_goals

    total_pkts = sum(
        getattr(env.agent_map[aid], "packets_delivered", 0)
        for aid in env.possible_agents
    )
    generated  = sum(
        getattr(env.agent_map[aid], "packets_generated", 0)
        for aid in env.possible_agents
    )
    metrics["packets_delivered"]  = total_pkts
    metrics["packets_generated"]  = generated
    metrics["delivery_ratio"]     = total_pkts / max(generated, 1)

    return dict(metrics), frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default=DEFAULT_WEIGHTS,
                        help="Path to BC actor weights (.pt)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Base random seed (incremented per episode)")
    parser.add_argument("--steps",    type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--render",   action="store_true",
                        help="Save a GIF per episode to bc_policy_ep<N>.gif")
    parser.add_argument("--dummy-radio", action="store_true",
                        help="Use dummy radio model (faster, no GPU needed)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"ERROR: BC weights not found at {args.weights}")
        print("Run train_mappo_bc.py first to generate them.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Weights: {args.weights}")
    print(f"Episodes: {args.episodes}, Steps: {args.steps}, Agents: {NUM_AGENTS}\n")

    # --- Build model ---
    num_outputs = _MOVE_CLASSES + _COMM_CLASSES * (NUM_AGENTS + 1)
    model = BCActorModel(SCALAR_DIM, num_outputs).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded BC model  (params: {sum(p.numel() for p in model.parameters()):,})\n")

    # --- Build environment ---
    hm = np.load(HM_PATH)
    radio_model = RadioMapModelNN(
        model_paths=MODEL_PATHS,
        heightmap=hm,
        env_width=256,
        env_height=256,
        dummy_mode=args.dummy_radio,
        device=torch.device("cpu"),
    )
    env = LunarRoverMeshEnv(
        hm_path=HM_PATH,
        radio_model=radio_model,
        num_agents=NUM_AGENTS,
        render_mode="rgb_array" if args.render else None,
    )

    # --- Evaluation loop ---
    all_metrics = []
    for ep in range(args.episodes):
        m, frames = run_episode(env, model, device, seed=args.seed + ep, max_steps=args.steps)
        all_metrics.append(m)
        print(
            f"Ep {ep+1:3d}/{args.episodes} | "
            f"Goals {m['goals_completed']:.0f}/{m['total_goals']:.0f} "
            f"({m['goal_ratio']*100:.0f}%) | "
            f"Pkts {m['packets_delivered']:.0f}/{m['packets_generated']:.0f} "
            f"({m['delivery_ratio']*100:.0f}%) | "
            f"Reward {m['total_reward']:7.1f} | "
            f"Steps {m['steps']:.0f}"
        )
        if args.render and frames:
            gif_path = f"bc_policy_ep{ep+1}.gif"
            print(f"  Saving GIF ({len(frames)} frames) -> {gif_path}")
            imageio.mimsave(gif_path, frames, fps=10)
            try:
                mp4_path = gif_path.replace(".gif", ".mp4")
                imageio.mimwrite(mp4_path, frames, fps=10, output_params=["-vcodec", "libx264"])
            except Exception:
                pass

    env.close()

    # --- Aggregate summary ---
    def avg(key):
        return float(np.mean([m[key] for m in all_metrics]))

    print("\n--- Summary ---")
    print(f"  Goal ratio      : {avg('goal_ratio')*100:.1f}%")
    print(f"  Delivery ratio  : {avg('delivery_ratio')*100:.1f}%")
    print(f"  Avg reward      : {avg('total_reward'):.1f}")
    print(f"  Avg steps       : {avg('steps'):.0f}")
    print(f"  Avg pkts deliv. : {avg('packets_delivered'):.1f}")


if __name__ == "__main__":
    main()
