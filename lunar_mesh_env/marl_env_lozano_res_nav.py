"""
Lozano-style comm env with A*-guided residual navigation.

Navigation: A* provides the base direction each step; the RL policy picks
among the A*-suggested direction ±RES_SECTORS neighbouring sectors (3 choices
when RES_SECTORS=1). The tighter mask keeps navigation sensible from step 0
without any eps_nav curriculum while letting the policy learn radio-aware
micro-detours on top of A*.

Comm action: single Lozano edge — identical to LunarRoverMeshLozanoRLNavEnv.

Compare against:
  train_ppo_lozano.py         — A* nav (fixed), Lozano comm RL
  train_ppo_lozano_rl_nav.py  — full RL nav + comm (needs eps_nav curriculum)
  train_ppo_lozano_res_nav.py — A*-guided RL nav (this) + Lozano comm RL
"""

import numpy as np

from .pathfinding import a_star_search
from .marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv


# Sector encoding matches heuristic_move_action() in marl_env.py.
# Sectors increase CCW from East: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
_ACTION_TO_SECTOR = {1: 2, 2: 6, 3: 4, 4: 0, 5: 1, 6: 3, 7: 7, 8: 5}
_SECTOR_TO_ACTION = {v: k for k, v in _ACTION_TO_SECTOR.items()}


class LunarRoverMeshLozanoResNavEnv(LunarRoverMeshLozanoRLNavEnv):
    """
    Lozano RL comm + A*-biased navigation residuals.

    Action space (same as LunarRoverMeshLozanoRLNavEnv):
      MultiDiscrete([9, max_nodes])

    The nav action mask is tightened to ±RES_SECTORS sectors around the A*
    direction, so the policy makes small radio-aware corrections to A* rather
    than learning global navigation from scratch.
    """

    eps_nav: float = 0.0    # override parent default; no comm-hold forcing ever
    RES_SECTORS: int = 1    # ±sectors around A* direction allowed in nav mask
    STEP_PENALTY: float = -0.15  # per-step time pressure to discourage zig-zag hovering

    # ------------------------------------------------------------------
    # Reset: initialise A* paths for all agents
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        for agent_id in list(self.agent_map):
            self._refresh_path(agent_id)
        return result

    # ------------------------------------------------------------------
    # A* path helpers
    # ------------------------------------------------------------------

    def _astar_threshold(self):
        return (self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)) * 0.9

    def _refresh_path(self, agent_id):
        agent = self.agent_map[agent_id]
        path = a_star_search(
            self.heightmap,
            (int(agent.x), int(agent.y)),
            (int(agent.goal_x), int(agent.goal_y)),
            max_incline=self._astar_threshold(),
        )
        agent.nav_path = path if path else []

    # ------------------------------------------------------------------
    # Observation: replace nav portion of mask with A*-biased mask
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id):
        # Lazily refresh path when depleted (goal reached or A* found no path)
        if not getattr(self.agent_map.get(agent_id), "nav_path", None):
            self._refresh_path(agent_id)

        obs = super()._get_obs(agent_id)               # full mask shape (9 + max_nodes,)
        astar_action = self.heuristic_move_action(agent_id)  # 0-8

        terrain_mask = obs["action_mask"][:9].copy()
        nav_mask = self._biased_nav_mask(terrain_mask, astar_action)

        obs["action_mask"] = np.concatenate(
            [nav_mask, obs["action_mask"][9:]]
        ).astype(np.int8)
        return obs

    def _biased_nav_mask(self, terrain_mask: np.ndarray, astar_action: int) -> np.ndarray:
        """
        Restrict terrain_mask to the A* direction ±RES_SECTORS sectors.
        Falls back to full terrain mask if all adjacent directions are blocked.
        """
        if astar_action == 0:
            # A* is idle (goal reached or path blocked) — allow all terrain-valid moves
            return terrain_mask

        nav_mask = np.zeros(9, dtype=np.int8)
        nav_mask[0] = 1   # idle is always valid

        center = _ACTION_TO_SECTOR[astar_action]
        for offset in range(-self.RES_SECTORS, self.RES_SECTORS + 1):
            action_idx = _SECTOR_TO_ACTION[(center + offset) % 8]
            nav_mask[action_idx] = terrain_mask[action_idx]

        # If terrain blocks all A*-adjacent moves (cliff face etc.), fall back
        if nav_mask[1:].sum() == 0:
            return terrain_mask

        return nav_mask

    # ------------------------------------------------------------------
    # Step: apply per-step time pressure on top of base rewards
    # ------------------------------------------------------------------

    def step(self, actions):
        obs, rewards, terms, truncs, infos = super().step(actions)
        for aid in rewards:
            rewards[aid] += self.STEP_PENALTY
        return obs, rewards, terms, truncs, infos