"""
Nav-only RL env with epidemic communication.

The RL policy controls 9-way movement (action[0]) only.
All packet delivery is handled automatically by epidemic routing —
the comm action (action[1]) is fixed to hold (0) every step.

Inherits from LunarRoverMeshLozanoRLNavEnv to reuse:
  - heuristic_move_action / _compute_move_mask
  - _handle_communication_step (comm=0 → no-op; epidemic fires after)
  - packet generation, DTN buffer, reward structure

Observation space vs parent:
  - action_mask shrunk from (9 + max_nodes,) to (9,)
  - graph_adj, graph_node_features, buffer_usage, other_agent_connectivity dropped
  - move_history re-encoded as (dx,dy) displacement vectors (shape HISTORY_LEN*2)
    so the MLP can directly detect oscillation (zero net displacement pattern)
  - terrain_slopes(8,) added: normalised height delta in each of 8 move directions,
    giving one-step terrain look-ahead beyond the binary move mask
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv

# Action index → unit displacement.  Matches _handle_movement_step direction mapping.
_ACTION_DX = np.array([0, 0, 0, -1, 1, 0.707, -0.707, 0.707, -0.707], dtype=np.float32)
_ACTION_DY = np.array([0, 1, -1, 0, 0, 0.707, 0.707, -0.707, -0.707], dtype=np.float32)


class LunarRoverMeshNavOnlyEpidemicEnv(LunarRoverMeshLozanoRLNavEnv):
    """9-way RL nav; epidemic handles all packet delivery."""

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(9)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        d = dict(base.spaces)
        d["action_mask"] = spaces.Box(0, 1, shape=(9,), dtype=np.int8)
        # move_history re-encoded as (dx,dy) pairs in [-1,1]; shape = HISTORY_LEN*2
        d["move_history"] = spaces.Box(-1, 1, shape=(self.HISTORY_LEN * 2,), dtype=np.float32)
        # terrain_slopes: normalised Δheight in each of 8 move directions (actions 1-8)
        d["terrain_slopes"] = spaces.Box(-5, 5, shape=(8,), dtype=np.float32)
        # goal_dist: explicit scalar distance to goal, normalised by map diagonal
        d["goal_dist"] = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        d.pop("graph_adj", None)
        d.pop("graph_node_features", None)
        d.pop("buffer_usage", None)
        d.pop("other_agent_connectivity", None)
        return spaces.Dict(d)

    def _get_obs(self, agent_id):
        obs = super()._get_obs(agent_id)
        obs["action_mask"] = obs["action_mask"][:9]

        # Convert stored action indices → (dx,dy) displacement vectors.
        idx = self._move_history[agent_id].astype(int)   # shape (HISTORY_LEN,)
        dx  = _ACTION_DX[idx]                             # shape (HISTORY_LEN,)
        dy  = _ACTION_DY[idx]                             # shape (HISTORY_LEN,)
        obs["move_history"] = np.stack([dx, dy], axis=1).flatten()  # (HISTORY_LEN*2,)

        # Terrain slope in each of 8 move directions, normalised by MAX_INCLINE.
        agent = self.agent_map[agent_id]
        scale = self.MAX_DIST_PER_STEP
        nx = np.clip(agent.x + _ACTION_DX[1:] * scale, 0, self.width  - 1).astype(int)
        ny = np.clip(agent.y + _ACTION_DY[1:] * scale, 0, self.height - 1).astype(int)
        curr_z   = self.heightmap[int(agent.y), int(agent.x)]
        target_z = self.heightmap[ny, nx]
        slopes   = (target_z - curr_z) / max(self.MAX_INCLINE_PER_STEP, 1e-6)
        obs["terrain_slopes"] = np.clip(slopes, -5, 5).astype(np.float32)

        # Explicit distance to goal, normalised by map diagonal (~362 px for 256×256).
        agent = self.agent_map[agent_id]
        raw_dist = np.sqrt((agent.goal_x - agent.x)**2 + (agent.goal_y - agent.y)**2)
        obs["goal_dist"] = np.array([raw_dist / 362.0], dtype=np.float32)

        obs.pop("graph_adj", None)
        obs.pop("graph_node_features", None)
        obs.pop("buffer_usage", None)
        obs.pop("other_agent_connectivity", None)
        return obs

    def step(self, actions):
        # Lift scalar nav actions to [nav, 0] expected by parent step()
        full = {aid: np.array([int(a), 0]) for aid, a in actions.items()}
        return super().step(full)
