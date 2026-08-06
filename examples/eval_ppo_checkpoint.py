"""
Load the latest PPO checkpoint and run a visual evaluation episode.

Usage
-----
    # Auto-detect latest checkpoint
    python examples/eval_ppo_checkpoint.py

    # Point to a specific checkpoint directory
    python examples/eval_ppo_checkpoint.py --checkpoint /path/to/checkpoint_dir

    # Custom output stem and database path
    python examples/eval_ppo_checkpoint.py --out run1 --db results/eval.db

Output: <out>.gif in the current directory, metrics written to SQLite <db>.
"""

import os
import sys
import glob
import argparse
import sqlite3
import datetime

import numpy as np
import torch
import imageio

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lunar_mesh_env import LunarRoverMeshEnv, RadioMapModelNN
from lunar_mesh_env.marl_env_mlp_gat import LunarRoverMeshMLPGATEnv
from lunar_mesh_env.marl_env_radio import LunarRoverMeshRadioEnv
from lunar_mesh_env.marl_env_shared_reward import LunarRoverMeshSharedRewardEnv
from lunar_mesh_env.marl_env_forward_reward import LunarRoverMeshForwardRewardEnv
from lunar_mesh_env.marl_env_eps_comm import LunarRoverMeshEpsCommEnv
from lunar_mesh_env.marl_env_touched_reward import LunarRoverMeshTouchedRewardEnv
from lunar_mesh_env.marl_env_touched_eps_comm import LunarRoverMeshTouchedEpsCommEnv
from lunar_mesh_env.marl_env_astar_comm import LunarRoverMeshAStarCommEnv
from lunar_mesh_env.marl_env_astar_eps_comm import LunarRoverMeshAStarEpsCommEnv
from lunar_mesh_env.marl_env_astar_forward_reward import LunarRoverMeshAStarForwardRewardEnv
from lunar_mesh_env.marl_env_lozano import LunarRoverMeshLozanoEnv
from lunar_mesh_env.marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv
from lunar_mesh_env.marl_env_lozano_rl_nav_radio import LunarRoverMeshLozanoRLNavRadioEnv
from lunar_mesh_env.marl_env_lozano_rl_nav_cnn import LunarRoverMeshLozanoRLNavCNNEnv
from lunar_mesh_env.marl_env_lozano_rl_nav_crop import LunarRoverMeshLozanoRLNavCropEnv
from lunar_mesh_env.marl_env_lozano_res_nav import LunarRoverMeshLozanoResNavEnv
from lunar_mesh_env.marl_env_lozano_waypoint_nav import LunarRoverMeshLozanoWaypointNavEnv
from lunar_mesh_env.marl_env_lozano_conn_nav import LunarRoverMeshLozanoConnNavEnv
from lunar_mesh_env.radio_model_lookup import RadioMapModelLookup
from models.action_mask_model import TorchActionMaskModel
from models.mlp_gat_model import TorchMLPGATModel
from models.mlp_gat_radio_model import TorchMLPGATRadioModel
from models.mlp_gat_lozano_model import TorchMLPGATLozanoModel
from models.mlp_gat_lozano_waypoint_model import TorchMLPGATLozanoWaypointModel
from models.mlp_gat_lozano_radio_model import TorchMLPGATLozanoRadioModel
from models.mlp_gat_lozano_cnn_model import TorchMLPGATLozanoCNNModel
from models.mlp_gat_lozano_crop_model import TorchMLPGATLozanoCropModel

_RADIO_MODES    = {"radio_rssi": "rssi", "radio_crop": "crop", "radio_full": "full"}
_MLPGAT_SUBENVS = {
    "shared_reward":    LunarRoverMeshSharedRewardEnv,
    "forward_reward":   LunarRoverMeshForwardRewardEnv,
    "eps_comm":         LunarRoverMeshEpsCommEnv,
    "touched_reward":   LunarRoverMeshTouchedRewardEnv,
    "touched_eps_comm": LunarRoverMeshTouchedEpsCommEnv,
    "astar_comm":       LunarRoverMeshAStarCommEnv,
    "astar_eps_comm":          LunarRoverMeshAStarEpsCommEnv,
    "astar_forward_reward":    LunarRoverMeshAStarForwardRewardEnv,
}

_LOZANO_SUBENVS = {
    "lozano":                LunarRoverMeshLozanoEnv,
    "lozano_rl_nav":         LunarRoverMeshLozanoRLNavEnv,
    "lozano_rl_nav_radio":   LunarRoverMeshLozanoRLNavRadioEnv,
    "lozano_rl_nav_cnn":     LunarRoverMeshLozanoRLNavCNNEnv,
    "lozano_rl_nav_crop":    LunarRoverMeshLozanoRLNavCropEnv,
    "lozano_res_nav":        LunarRoverMeshLozanoResNavEnv,
    "lozano_waypoint_nav":   LunarRoverMeshLozanoWaypointNavEnv,
    "lozano_conn_nav":       LunarRoverMeshLozanoConnNavEnv,
}

DEFAULT_MAPS = "/home/paolo/Documents/lunar-mesh-env/DATA_MAPS/radio_maps_hm_15.npy" #os.path.join(os.path.dirname(os.path.abspath(__file__)),
                #            "radio_maps_hm_18.npy")

# ---------------------------------------------------------------------------
# Mirror the same paths / dims used in train_ppo_rllib.py
# ---------------------------------------------------------------------------
DATA_ROOT   = "/home/paolo/Documents/lunar-mesh-env/DATA/radio_data_2/radio_data_2"
HM_PATH     = f"{DATA_ROOT}/hm/hm_15.npy"
_PRETRAINED = os.path.join(_REPO_ROOT, "RadioLunaDiff/pretrained_models_network")
MODEL_PATHS = {
    "k2_model":        os.path.join(_PRETRAINED, "k2unet/best_k2_model.pth"),
    "pmnet_model":     os.path.join(_PRETRAINED, "pmnet/best_pm_model.pt"),
    "diffusion_model": os.path.join(_PRETRAINED, "diffusion"),
}

RAY_RESULTS_DIR = os.path.expanduser("~/ray_results")
NUM_AGENTS      = 3
MAX_STEPS       = 750
EVAL_SEED       = 199
DEFAULT_DB      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_metrics_real_maps_3.db")


# ---------------------------------------------------------------------------
# Env creator (same as training)
# ---------------------------------------------------------------------------

def _build_radio_model(hm, maps_path=None, dummy_mode=True):
    """Return a lookup model if maps_path is given, otherwise RadioMapModelNN."""
    if maps_path:
        return RadioMapModelLookup(maps_path=maps_path, heightmap=hm,
                                   env_width=256, env_height=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RadioMapModelNN(model_paths=MODEL_PATHS, heightmap=hm,
                           env_width=256, env_height=256,
                           dummy_mode=dummy_mode, device=device)


def env_creator(config):
    hm = np.load(config.get("hm_path", HM_PATH))
    radio_model = _build_radio_model(
        hm,
        maps_path=config.get("maps_path", None),
        dummy_mode=config.get("dummy_mode", True),
    )

    model_type = config.get("model_type", "mlp")
    if model_type in _RADIO_MODES:
        raw_env = LunarRoverMeshRadioEnv(
            hm_path=config.get("hm_path", HM_PATH),
            radio_model=radio_model,
            num_agents=config.get("num_agents", NUM_AGENTS),
            radio_obs=_RADIO_MODES[model_type],
            render_mode="rgb_array",
            seed=19,
        )
    elif model_type in _LOZANO_SUBENVS:
        env_cls = _LOZANO_SUBENVS[model_type]
        kwargs = dict(
            hm_path=config.get("hm_path", HM_PATH),
            radio_model=radio_model,
            num_agents=config.get("num_agents", NUM_AGENTS),
            packet_mode='rate',
            render_mode="rgb_array",
            seed=19,
        )
        if model_type == "lozano":
            kwargs["action_type"] = config.get("action_type", "multidiscrete")
        raw_env = env_cls(**kwargs)
        if model_type in ("lozano_rl_nav", "lozano_rl_nav_radio", "lozano_rl_nav_cnn", "lozano_rl_nav_crop"):
            raw_env.eps_nav = 0.0
    elif model_type in _MLPGAT_SUBENVS:
        env_cls = _MLPGAT_SUBENVS[model_type]
        raw_env = env_cls(
            hm_path=config.get("hm_path", HM_PATH),
            radio_model=radio_model,
            num_agents=config.get("num_agents", NUM_AGENTS),
            render_mode="rgb_array",
            seed=19,
        )
    elif model_type == "mlp_gat":
        raw_env = LunarRoverMeshMLPGATEnv(
            hm_path=config.get("hm_path", HM_PATH),
            radio_model=radio_model,
            num_agents=config.get("num_agents", NUM_AGENTS),
            render_mode="rgb_array",
            seed=19,
        )
    else:
        raw_env = LunarRoverMeshEnv(
            hm_path=config.get("hm_path", HM_PATH),
            radio_model=radio_model,
            num_agents=config.get("num_agents", NUM_AGENTS),
            render_mode="rgb_array",
            seed=19,
        )
    env = ParallelPettingZooEnv(raw_env)
    env.observation_space = raw_env.observation_spaces[raw_env.possible_agents[0]]
    env.action_space = raw_env.action_space(raw_env.possible_agents[0])
    return env


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_checkpoint(results_dir: str) -> str:
    pattern = os.path.join(results_dir, "**", "checkpoint_*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {results_dir}.\n"
            "Pass --checkpoint explicitly or check RAY_RESULTS_DIR."
        )
    return max(candidates, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    # Migrate episodes table
    existing_ep = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    if "episodes" in tables and "run_name" not in existing_ep:
        conn.execute("ALTER TABLE episodes ADD COLUMN run_name TEXT")
        conn.commit()
    # Migrate env_step_metrics table
    existing_env = {r[1] for r in conn.execute("PRAGMA table_info(env_step_metrics)").fetchall()}
    if "env_step_metrics" in tables:
        if "bs_unique_packets" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN bs_unique_packets INTEGER")
        if "bs_duplicate_packets" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN bs_duplicate_packets INTEGER")
        if "covered_pixels" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN covered_pixels INTEGER")
        if "bw_utilization" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN bw_utilization REAL")
        if "bw_util_connected" not in existing_env:
            conn.execute("ALTER TABLE env_step_metrics ADD COLUMN bw_util_connected REAL")
        conn.commit()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            checkpoint    TEXT    NOT NULL,
            num_agents    INTEGER NOT NULL,
            total_steps   INTEGER,
            dummy_mode    INTEGER NOT NULL,
            seed          INTEGER,
            run_name      TEXT
        );

        CREATE TABLE IF NOT EXISTS step_metrics (
            episode_id            INTEGER NOT NULL REFERENCES episodes(id),
            step                  INTEGER NOT NULL,
            agent_id              TEXT    NOT NULL,
            -- reward
            reward                REAL,
            cumulative_reward     REAL,
            -- navigation
            pos_x                 REAL,
            pos_y                 REAL,
            goal_x                REAL,
            goal_y                REAL,
            dist_to_goal          REAL,
            total_distance        REAL,
            goals_completed       INTEGER,
            mission_done          INTEGER,
            -- energy
            energy                REAL,
            -- communication
            datarate_mbps         REAL,
            bs_connected          INTEGER,
            num_neighbors         INTEGER,
            -- DTN buffer
            num_packets           INTEGER,
            num_packets_generated INTEGER,
            buffer_usage          REAL
        );

        CREATE TABLE IF NOT EXISTS env_step_metrics (
            episode_id            INTEGER NOT NULL REFERENCES episodes(id),
            step                  INTEGER NOT NULL,
            sim_time              INTEGER,
            bs_packets_received   INTEGER,
            bs_unique_packets     INTEGER,
            bs_duplicate_packets  INTEGER,
            avg_datarate_mbps     REAL,
            bs_link_ratio         REAL,
            num_active_agents     INTEGER,
            covered_pixels        INTEGER,
            bw_utilization        REAL,
            bw_util_connected     REAL
        );
    """)
    conn.commit()
    return conn


def collect_agent_metrics(env: LunarRoverMeshEnv, agent_id: str,
                          step: int, reward: float, cumulative: float,
                          episode_id: int) -> tuple:
    agent = env.agent_map[agent_id]
    dtn = agent.payload_manager.get_state()
    dist_to_goal = float(np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2))
    return (
        episode_id,
        step,
        agent_id,
        reward,
        cumulative,
        float(agent.x),
        float(agent.y),
        float(agent.goal_x),
        float(agent.goal_y),
        dist_to_goal,
        float(agent.total_distance),
        int(env.agent_goals_completed.get(agent_id, 0)),
        int(env.mission_done.get(agent_id, False)),
        float(agent.energy),
        float(agent.current_datarate),
        int(agent.bs_connected),
        int(len(agent.neighbors)),
        int(dtn["num_packets"]),
        int(dtn["num_packets_generated"]),
        float(dtn["payload_size"] / max(dtn["buffer_size"], 1)),
    )


def collect_env_metrics(env: LunarRoverMeshEnv, step: int, episode_id: int,
                        prev_bs_packets: int = 0) -> tuple:
    from lunar_mesh_env.marl_env import PACKET_SIZE_BITS
    active = env.agents
    bs = env.base_station
    # current_datarate is only set by the old route-finder, not by A*/Lozano env.
    # Compute actual link rate from the radio model instead.
    dr_vals = []
    bs_capacity_bits = 0.0
    for a in active:
        agent = env.agent_map[a]
        if agent.bs_connected:
            rate = env.radio_model.get_throughput_pos(agent.x, agent.y, bs.x, bs.y)
            dr_vals.append(rate)
            bs_capacity_bits += rate * 1e6 * env.STEP_LENGTH
        elif agent.neighbors:
            dr_vals.append(max(
                env.radio_model.get_throughput_pos(agent.x, agent.y, n.x, n.y)
                for n in agent.neighbors))
        else:
            dr_vals.append(0.0)
    avg_dr = float(np.mean(dr_vals)) if dr_vals else 0.0
    bs_links = sum(1 for a in active if env.agent_map[a].bs_connected)
    bs_ratio = bs_links / len(active) if active else 0.0

    # Bandwidth utilization: bits delivered to BS this step / total BS link capacity.
    # Uses raw count (includes duplicates) since duplicates consume real bandwidth.
    packets_this_step = bs.num_packets_received - prev_bs_packets
    bits_delivered = packets_this_step * PACKET_SIZE_BITS
    bw_util = min(bits_delivered / bs_capacity_bits, 1.0) if bs_capacity_bits > 0 else 0.0
    # NULL when no BS connection so AVG() in SQL ignores disconnected steps.
    bw_util_connected = min(bits_delivered / bs_capacity_bits, 1.0) if bs_capacity_bits > 0 else None

    # Instantaneous coverage: union of BS map + each rover's current radio map.
    # Uses _cached_radio_map set by _compute_coverage_reward this step.
    covered_pixels = 0
    if hasattr(env, "bs_radio_map"):
        instant_map = env.bs_radio_map.copy() if env.bs_radio_map is not None else None
        for agent in env.agent_map.values():
            rm = getattr(agent, "_cached_radio_map", None)
            if rm is None:
                rm = env.radio_model.generate_map((agent.x, agent.y), '5.8')
            if rm is not None:
                if instant_map is None:
                    instant_map = rm.copy()
                else:
                    np.maximum(instant_map, rm, out=instant_map)
        if instant_map is not None:
            covered_pixels = int((instant_map >= env.MIN_DBM_THRESHOLD).sum())
    return (
        episode_id,
        step,
        int(env.sim_time),
        int(bs.num_packets_received),
        int(len(bs.packets_received)),
        int(bs.num_duplicates_received),
        avg_dr,
        bs_ratio,
        len(active),
        covered_pixels,
        float(bw_util),
        bw_util_connected,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--out", type=str, default="eval_ppo")
    parser.add_argument("--db", type=str, default=DEFAULT_DB,
                        help="SQLite database path for metrics.")
    parser.add_argument("--name", type=str, default=None,
                        help="Human-readable label for this run (shown in plots).")
    parser.add_argument("--lookup-maps", type=str, default=None, metavar="PATH",
                        help="Use precomputed radio maps (from precompute_radio_maps.py). "
                             "Fastest and matches lookup-trained policies. "
                             f"Defaults to {DEFAULT_MAPS} if that file exists.")
    parser.add_argument("--dummy-mode", action="store_true", default=True,
                        help="Use analytic radio model. Ignored when --lookup-maps is set.")
    parser.add_argument("--no-dummy-mode", dest="dummy_mode", action="store_false")
    parser.add_argument("--model",
                        choices=["mlp", "mlp_gat",
                                 "radio_rssi", "radio_crop", "radio_full",
                                 "shared_reward", "forward_reward", "eps_comm",
                                 "touched_reward", "touched_eps_comm",
                                 "astar_comm", "astar_eps_comm",
                                 "astar_forward_reward",
                                 "lozano", "lozano_rl_nav", "lozano_rl_nav_radio",
                                 "lozano_rl_nav_cnn", "lozano_rl_nav_crop",
                                 "lozano_res_nav", "lozano_waypoint_nav",
                                 "lozano_conn_nav"],
                        default="mlp",
                        help="Policy architecture to evaluate (default: mlp)")
    parser.add_argument("--routing", default="none",
                        choices=["none", "epidemic", "spray_and_wait"],
                        help="Override comm decisions with a routing protocol (default: none)")
    parser.add_argument("--spray-copies", type=int, default=4,
                        help="Max copies for spray_and_wait (default: 4)")
    parser.add_argument("--routing-bw-limit", action="store_true", default=False,
                        help="Constrain epidemic to actual link rate (Vahdat 802.11 model)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes (default: 1)")
    parser.add_argument("--ttl", type=int, default=None,
                        help="Override packet TTL in steps (default: env default of 50)")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo",
                        help="Algorithm that produced the checkpoint (default: ppo)")
    args = parser.parse_args()

    # Auto-detect lookup maps if not specified but default file exists
    maps_path = args.lookup_maps
    if maps_path is None and os.path.exists(DEFAULT_MAPS):
        maps_path = DEFAULT_MAPS
        print(f"Using precomputed radio maps: {maps_path}")

    routing_only = args.routing != "none"

    checkpoint = None if routing_only else (args.checkpoint or find_latest_checkpoint(RAY_RESULTS_DIR))
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
    print(f"Model type: {'routing-only (' + args.routing + ')' if routing_only else args.model}")

    ModelCatalog.register_custom_model("action_mask_model",      TorchActionMaskModel)
    ModelCatalog.register_custom_model("mlp_gat_model",          TorchMLPGATModel)
    ModelCatalog.register_custom_model("mlp_gat_radio_model",    TorchMLPGATRadioModel)
    ModelCatalog.register_custom_model("mlp_gat_lozano_model",         TorchMLPGATLozanoModel)
    ModelCatalog.register_custom_model("mlp_gat_lozano_waypoint_model", TorchMLPGATLozanoWaypointModel)
    ModelCatalog.register_custom_model("mlp_gat_lozano_radio_model",   TorchMLPGATLozanoRadioModel)
    ModelCatalog.register_custom_model("mlp_gat_lozano_cnn_model",     TorchMLPGATLozanoCNNModel)
    ModelCatalog.register_custom_model("mlp_gat_lozano_crop_model",    TorchMLPGATLozanoCropModel)
    register_env("lunar_mesh_v1", env_creator)
    ray.init(ignore_reinit_error=True, num_gpus=0)

    lozano_action_type = "discrete" if args.algo == "dqn" else "multidiscrete"
    env_config = {"num_agents": NUM_AGENTS, "hm_path": HM_PATH,
                  "maps_path": maps_path, "dummy_mode": args.dummy_mode,
                  "model_type": args.model,
                  "action_type": lozano_action_type}

    if args.model in _RADIO_MODES:
        custom_model      = "mlp_gat_radio_model"
        extra_model_cfg   = {"custom_model_config": {"radio_mode": _RADIO_MODES[args.model]}}
    elif args.model in _LOZANO_SUBENVS:
        if args.model == "lozano_rl_nav_crop":
            custom_model    = "mlp_gat_lozano_crop_model"
            extra_model_cfg = {"custom_model_config": {"move_dirs": 9}}
        elif args.model == "lozano_rl_nav_cnn":
            custom_model    = "mlp_gat_lozano_cnn_model"
            extra_model_cfg = {"custom_model_config": {"move_dirs": 9}}
        elif args.model == "lozano_rl_nav_radio":
            custom_model    = "mlp_gat_lozano_radio_model"
            extra_model_cfg = {"custom_model_config": {"move_dirs": 9}}
        elif args.model in ("lozano_rl_nav", "lozano_res_nav"):
            custom_model    = "mlp_gat_lozano_model"
            extra_model_cfg = {"custom_model_config": {"move_dirs": 9}}
        elif args.model == "lozano_waypoint_nav":
            from lunar_mesh_env.marl_env_lozano_waypoint_nav import (
                LunarRoverMeshLozanoWaypointNavEnv as _WNE,
            )
            custom_model    = "mlp_gat_lozano_waypoint_model"
            extra_model_cfg = {"custom_model_config": {"move_dirs": _WNE.N_NAV}}
        else:
            custom_model    = "mlp_gat_lozano_model"
            extra_model_cfg = {}
    elif args.model in ("mlp_gat", *_MLPGAT_SUBENVS):
        custom_model      = "mlp_gat_model"
        extra_model_cfg   = {}
    else:
        custom_model      = "action_mask_model"
        extra_model_cfg   = {}

    policy = None
    if not routing_only:
        probe = env_creator(env_config)
        obs_space = probe.observation_space
        act_space = probe.action_space
        probe.close()

        base_cfg = (
            (DQNConfig() if args.algo == "dqn" else PPOConfig())
            .environment("lunar_mesh_v1", env_config=env_config)
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .multi_agent(
                policies={"shared_policy": (None, obs_space, act_space, {})},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .env_runners(num_env_runners=0)
            .resources(num_gpus=0)
        )
        model_cfg = {"custom_model": custom_model,
                     "_disable_preprocessor_api": True,
                     **extra_model_cfg}
        if args.algo == "dqn":
            base_cfg = base_cfg.training(
                model=model_cfg,
                hiddens=[],
                replay_buffer_config={"type": "MultiAgentPrioritizedReplayBuffer"},
            )
        else:
            base_cfg = base_cfg.training(model=model_cfg)
        algo = base_cfg.build()
        try:
            algo.restore(checkpoint)
        except (ValueError, RuntimeError) as e:
            if "optimizer" in str(e).lower() or "parameter group" in str(e).lower():
                # Optimizer state mismatch — weights-only fallback (safe for eval)
                import pickle
                policy_pkl = os.path.join(
                    checkpoint, "policies", "shared_policy", "policy_state.pkl"
                )
                with open(policy_pkl, "rb") as f:
                    policy_state = pickle.load(f)
                algo.get_policy("shared_policy").set_weights(policy_state["weights"])
                print(f"Warning: optimizer state skipped ({e})")
            else:
                raise
        policy = algo.get_policy("shared_policy")
        print("Policy restored.")
    else:
        print(f"Routing-only mode ({args.routing}) — no checkpoint needed.")

    # ── Init DB ───────────────────────────────────────────────────────────
    conn = init_db(args.db)

    # ── Multi-episode eval loop ───────────────────────────────────────────
    hm = np.load(HM_PATH)
    radio_model = _build_radio_model(hm, maps_path=maps_path, dummy_mode=args.dummy_mode)

    _routing_kwargs = dict(
        routing_protocol=args.routing,
        spray_copies=args.spray_copies,
        routing_bw_limit=args.routing_bw_limit,
    )

    seeds = [EVAL_SEED + i for i in range(args.episodes)]
    print(f"Running {args.episodes} episode(s), seeds {seeds[0]}..{seeds[-1]}")

    packets_per_ep = []

    def _make_env(seed):
        # Routing-only baselines use A* navigation so only the routing protocol matters.
        # packet_mode='rate' matches lozano env (TTL=5000); boolean mode (TTL=50) expires
        # packets before agents reach the BS.
        if routing_only:
            return LunarRoverMeshAStarCommEnv(
                hm_path=HM_PATH, radio_model=radio_model, num_agents=NUM_AGENTS,
                packet_mode='rate', render_mode="rgb_array", seed=seed, **_routing_kwargs,
            )
        if args.model in _RADIO_MODES:
            return LunarRoverMeshRadioEnv(
                hm_path=HM_PATH, radio_model=radio_model, num_agents=NUM_AGENTS,
                radio_obs=_RADIO_MODES[args.model], render_mode="rgb_array",
                seed=seed, **_routing_kwargs,
            )
        elif args.model in _LOZANO_SUBENVS:
            env_cls = _LOZANO_SUBENVS[args.model]
            kw = dict(hm_path=HM_PATH, radio_model=radio_model, num_agents=NUM_AGENTS,
                      packet_mode='rate', render_mode="rgb_array",
                      seed=seed, **_routing_kwargs)
            if args.model == "lozano":
                kw["action_type"] = lozano_action_type
            env = env_cls(**kw)
            if args.model in ("lozano_rl_nav", "lozano_rl_nav_radio", "lozano_rl_nav_cnn", "lozano_rl_nav_crop", "lozano_res_nav"):
                env.eps_nav = 0.0
            return env
        elif args.model in _MLPGAT_SUBENVS:
            env_cls = _MLPGAT_SUBENVS[args.model]
            return env_cls(
                hm_path=HM_PATH, radio_model=radio_model, num_agents=NUM_AGENTS,
                render_mode="rgb_array", seed=seed, **_routing_kwargs,
            )
        elif args.model == "mlp_gat":
            return LunarRoverMeshMLPGATEnv(
                hm_path=HM_PATH, radio_model=radio_model, num_agents=NUM_AGENTS,
                render_mode="rgb_array", seed=seed, **_routing_kwargs,
            )
        else:
            return LunarRoverMeshEnv(
                hm_path=HM_PATH, radio_model=radio_model, num_agents=NUM_AGENTS,
                render_mode="rgb_array", seed=seed, **_routing_kwargs,
            )

    for ep_idx, seed in enumerate(seeds):
        env = _make_env(seed)
        if args.ttl is not None:
            env.PACKET_TTL = args.ttl

        cur = conn.cursor()
        cur.execute(
            "INSERT INTO episodes (timestamp, checkpoint, num_agents, dummy_mode, seed, run_name) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.datetime.now().isoformat(), checkpoint or args.routing, NUM_AGENTS,
             int(args.dummy_mode), seed, args.name),
        )
        conn.commit()
        episode_id = cur.lastrowid
        print(f"\nEpisode {ep_idx + 1}/{args.episodes}  seed={seed}  episode_id={episode_id}")

        obs, _ = env.reset()
        frames = []
        total_rewards    = {aid: 0.0 for aid in env.possible_agents}
        agent_rows       = []
        env_rows         = []
        prev_bs_packets  = 0

        print(f"Running {args.steps} steps...")
        for step in range(args.steps):
            actions = {}
            for agent_id in env.agents:
                if policy is not None:
                    action, _, _ = policy.compute_single_action(
                        obs[agent_id],
                        policy_id="shared_policy",
                        explore=False,
                    )
                else:
                    # Routing-only: A* handles nav, routing protocol handles comm.
                    action = np.zeros(env.action_space(agent_id).shape, dtype=np.int64)
                actions[agent_id] = action
                if step == 0 and ep_idx == 0:
                    mask = obs[agent_id].get("action_mask", [])
                    print(f"  [{agent_id}] action: {action}  mask[:9]: {mask[:9]}")

            obs, rewards, terms, truncs, _ = env.step(actions)

            for aid, r in rewards.items():
                total_rewards[aid] += r
                agent_rows.append(
                    collect_agent_metrics(env, aid, step, r, total_rewards[aid], episode_id)
                )

            env_rows.append(collect_env_metrics(env, step, episode_id, prev_bs_packets))
            prev_bs_packets = int(env.base_station.num_packets_received)

            if ep_idx == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if step % 20 == 0:
                reward_str = "  ".join(
                    f"{aid}: {total_rewards[aid]:.1f}" for aid in env.possible_agents
                )
                print(f"  step {step:>4}  |  cumulative rewards: {reward_str}"
                      f"  |  BS pkts: {env.base_station.num_packets_received}")

            if all(terms.values()) or all(truncs.values()):
                print(f"Episode finished at step {step}.")
                break

        packets_per_ep.append(env.base_station.num_packets_received)

        conn.executemany(
            "INSERT INTO step_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            agent_rows,
        )
        conn.executemany(
            "INSERT INTO env_step_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            env_rows,
        )
        cur.execute("UPDATE episodes SET total_steps=? WHERE id=?", (step + 1, episode_id))
        conn.commit()
        print(f"  BS unique packets: {env.base_station.num_packets_received}  "
              f"episode_id={episode_id}")

        if ep_idx == 0 and frames:
            gif_path = f"{args.out}.gif"
            imageio.mimsave(gif_path, frames, fps=4)
            print(f"  GIF saved → {gif_path}")

        env.close()

    conn.close()
    if not routing_only:
        algo.stop()
    ray.shutdown()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nMetrics saved → {args.db}")
    if len(packets_per_ep) > 1:
        print(f"BS unique packets: {np.mean(packets_per_ep):.1f} ± {np.std(packets_per_ep):.1f}"
              f"  over {args.episodes} episodes")


if __name__ == "__main__":
    main()
