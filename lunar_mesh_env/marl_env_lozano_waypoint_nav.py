"""
Lozano-style comm env with slow-timescale waypoint-residual navigation.

Navigation: every WAYPOINT_INTERVAL steps (or on relay arrival / new goal), the RL
policy chooses one of N_NAV options:
  - index 0  = NULL → navigate straight to true goal via terrain A*
  - index 1..16 = waypoint offsets from the nominal A* subgoal
                  (8 compass directions × 2 magnitudes {R/2, R})
Between decisions, A* low-level pathing follows the committed nav_path.

This fixes the timescale problem of the per-step residual: "go rim-hugging for
relay coverage" is now a single coherent action rather than 20 repeated micro-
corrections that A* instantly undoes.

Comm action: single Lozano edge — identical to LunarRoverMeshLozanoRLNavEnv.

Floor guarantee: a policy that always emits NULL reproduces terrain-only A*
navigation (same as lozano_res_nav at RES_SECTORS=0).

Observation extras (mandatory per spec — connectivity signal for the nav head):
  - bs_signal       (1,)          normalised dBm at current position
  - waypoint_signals (N_NAV-1,)  normalised dBm at each of the 16 codebook cells
  - decision_due    (1,)          1 if nav action is live this step

Step penalty: small per-moving-step cost; exempt during RELAY_HOLD loitering.
"""

import functools
import numpy as np
from gymnasium import spaces

from .pathfinding import a_star_search
from .marl_env_lozano_rl_nav import LunarRoverMeshLozanoRLNavEnv

# 8 compass directions (N, NE, E, SE, S, SW, W, NW) as unit-vector components
_COMPASS_DIRS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]


class LunarRoverMeshLozanoWaypointNavEnv(LunarRoverMeshLozanoRLNavEnv):
    """
    Lozano RL comm + slow-timescale waypoint-residual navigation.

    Action space:
      MultiDiscrete([N_NAV, max_nodes])
      action[0] = waypoint choice (0=NULL/A*, 1..N_NAV-1=codebook offsets)
      action[1] = Lozano comm edge (unchanged)

    The policy commits a waypoint every WAYPOINT_INTERVAL steps; between
    decisions action[0] is a no-op and is masked to NULL-only.
    """

    eps_nav: float = 0.0          # no curriculum; comm trained from step 0
    WAYPOINT_RADIUS: int = 30     # R cells — codebook magnitudes are R//2 and R
    WAYPOINT_INTERVAL: int = 10   # decision cadence (steps)
    RELAY_HOLD: int = 5           # hold steps at relay after arrival (0 = disabled)
    N_NAV: int = 17               # 1 NULL + 8 dirs × 2 magnitudes = 17
    STEP_PENALTY: float = -0.02   # per moving step (exempt during hold)

    # ------------------------------------------------------------------
    # Init: build offset codebook
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        half_r = max(1, self.WAYPOINT_RADIUS // 2)
        full_r = self.WAYPOINT_RADIUS

        self._codebook: list[tuple[float, float]] = []
        for r in [half_r, full_r]:
            for dx, dy in _COMPASS_DIRS:
                norm = float(np.sqrt(dx ** 2 + dy ** 2))
                self._codebook.append((dx / norm * r, dy / norm * r))

        assert len(self._codebook) == self.N_NAV - 1, \
            f"codebook size mismatch: {len(self._codebook)} != {self.N_NAV - 1}"

        # Per-agent waypoint state (re-initialised in reset)
        self._wp_counter: dict[str, int] = {}
        self._decision_due: dict[str, bool] = {}
        self._relay_target: dict[str, tuple[int, int] | None] = {}
        self._hold_left: dict[str, int] = {}
        # Temp flag: was a decision active at start of this step? (for _get_obs)
        self._this_step_decision: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Action / observation spaces
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        max_nodes = len(self.possible_agents) + 1
        return spaces.MultiDiscrete([self.N_NAV, max_nodes])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        max_nodes = len(self.possible_agents) + 1
        d = dict(base.spaces)
        # Resize action mask: N_NAV nav slots + comm slots
        d["action_mask"] = spaces.Box(
            0, 1, shape=(self.N_NAV + max_nodes,), dtype=np.int8
        )
        # Connectivity signal features (mandatory for useful nav learning)
        d["bs_signal"] = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        d["waypoint_signals"] = spaces.Box(
            0.0, 1.0, shape=(self.N_NAV - 1,), dtype=np.float32
        )
        d["decision_due"] = spaces.Box(0, 1, shape=(1,), dtype=np.int8)
        return spaces.Dict(d)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Agents are now placed; initialise waypoint state and refresh A* paths.
        for aid in list(self.agent_map):
            self._wp_counter[aid] = 0
            self._decision_due[aid] = True   # first step IS a decision step
            self._this_step_decision[aid] = True   # expose decision_due=1 in first obs
            self._relay_target[aid] = None
            self._hold_left[aid] = 0
            self._refresh_path_to_goal(aid)
        # Re-compute obs so the returned dict shows correct nav masks / signals.
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        return obs, info

    # ------------------------------------------------------------------
    # A* helpers
    # ------------------------------------------------------------------

    def _astar_threshold(self) -> float:
        return (self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)) * 0.9

    def _refresh_path_to_goal(self, agent_id: str) -> None:
        agent = self.agent_map[agent_id]
        path = a_star_search(
            self.heightmap,
            (int(agent.x), int(agent.y)),
            (int(agent.goal_x), int(agent.goal_y)),
            max_incline=self._astar_threshold(),
        )
        agent.nav_path = path if path else []

    def _refresh_path_to(self, agent_id: str, target_xy: tuple[int, int]) -> bool:
        """Run A* to target_xy; store on nav_path. Returns True if path found."""
        agent = self.agent_map[agent_id]
        path = a_star_search(
            self.heightmap,
            (int(agent.x), int(agent.y)),
            target_xy,
            max_incline=self._astar_threshold(),
        )
        agent.nav_path = path if path else []
        return path is not None

    # ------------------------------------------------------------------
    # Nominal subgoal (straight-line proxy, O(1) — avoids A* in obs)
    # ------------------------------------------------------------------

    def _nominal_subgoal_fast(self, agent_id: str) -> tuple[float, float]:
        """Point WAYPOINT_RADIUS cells ahead along the straight line to goal."""
        agent = self.agent_map[agent_id]
        dx = agent.goal_x - agent.x
        dy = agent.goal_y - agent.y
        dist = max(float(np.sqrt(dx ** 2 + dy ** 2)), 1e-6)
        r = min(float(self.WAYPOINT_RADIUS), dist)
        return (agent.x + dx / dist * r, agent.y + dy / dist * r)

    # ------------------------------------------------------------------
    # Connectivity signal lookup
    # ------------------------------------------------------------------

    def _signal_at(self, x: float, y: float) -> float:
        """Normalised BS dBm at (x, y): 0.0 = at threshold, 1.0 = threshold+40dB."""
        if self.bs_radio_map is None:
            return 0.0
        px = int(np.clip(x, 0, self.width - 1))
        py = int(np.clip(y, 0, self.height - 1))
        raw = float(self.bs_radio_map[py, px])
        return float(np.clip((raw - self.MIN_DBM_THRESHOLD) / 40.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Waypoint codebook helpers
    # ------------------------------------------------------------------

    def _candidate_xy(self, agent_id: str, offset_idx: int) -> tuple[int, int]:
        """Map codebook offset index (0..N_NAV-2) to candidate (rx, ry)."""
        sx, sy = self._nominal_subgoal_fast(agent_id)
        dx, dy = self._codebook[offset_idx]
        rx = int(np.clip(sx + dx, 0, self.width - 1))
        ry = int(np.clip(sy + dy, 0, self.height - 1))
        return (rx, ry)

    def _quick_valid(self, agent_id: str, rx: int, ry: int) -> bool:
        """Cheap terrain check: in-bounds + rough slope feasibility."""
        if not (0 <= rx < int(self.width) and 0 <= ry < int(self.height)):
            return False
        agent = self.agent_map[agent_id]
        h_diff = (self.heightmap[ry, rx]
                  - self.heightmap[int(agent.y), int(agent.x)])
        return h_diff <= self.MAX_INCLINE_PER_STEP * self.WAYPOINT_RADIUS

    def _resolve_waypoint(
        self, agent_id: str, action_idx: int
    ) -> tuple[int, int] | None:
        """
        Full A* reachability check. Returns relay (rx, ry) or None (fall back to NULL).
        Only called on decision steps (expensive allowed).
        """
        if action_idx == 0:
            return None
        rx, ry = self._candidate_xy(agent_id, action_idx - 1)
        agent = self.agent_map[agent_id]
        path = a_star_search(
            self.heightmap,
            (int(agent.x), int(agent.y)),
            (rx, ry),
            max_incline=self._astar_threshold(),
        )
        return (rx, ry) if path is not None else None

    # ------------------------------------------------------------------
    # Nav action mask
    # ------------------------------------------------------------------

    def _build_nav_mask(self, agent_id: str) -> np.ndarray:
        """
        N_NAV-element mask.
        - Non-decision steps: NULL only (mask[0]=1, rest=0).
        - Decision steps: NULL + quick-valid codebook entries.
        """
        mask = np.zeros(self.N_NAV, dtype=np.int8)
        mask[0] = 1  # NULL always valid
        if not self._decision_due.get(agent_id, False):
            return mask
        for i in range(self.N_NAV - 1):
            rx, ry = self._candidate_xy(agent_id, i)
            if self._quick_valid(agent_id, rx, ry):
                mask[i + 1] = 1
        return mask

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: str) -> dict:
        obs = super()._get_obs(agent_id)       # action_mask shape: (9 + max_nodes,)
        agent = self.agent_map[agent_id]

        # Replace nav portion of action mask (first 9 slots → N_NAV slots)
        nav_mask = self._build_nav_mask(agent_id)
        comm_mask = obs["action_mask"][9:]      # comm portion: (max_nodes,) — unchanged
        obs["action_mask"] = np.concatenate([nav_mask, comm_mask]).astype(np.int8)

        # BS signal at current position
        obs["bs_signal"] = np.array(
            [self._signal_at(agent.x, agent.y)], dtype=np.float32
        )

        # Signal at each of the 16 codebook waypoints (relative to nominal subgoal)
        sx, sy = self._nominal_subgoal_fast(agent_id)
        obs["waypoint_signals"] = np.array(
            [self._signal_at(sx + dx, sy + dy) for (dx, dy) in self._codebook],
            dtype=np.float32,
        )

        # Decision flag: was a nav decision active at the start of this step?
        obs["decision_due"] = np.array(
            [1 if self._this_step_decision.get(agent_id, False) else 0],
            dtype=np.int8,
        )

        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: dict) -> tuple:
        # ── Phase 1: record decision status; resolve waypoints ──────────
        for agent_id, action in actions.items():
            in_hold = self._hold_left.get(agent_id, 0) > 0
            is_due = (not in_hold) and self._decision_due.get(agent_id, False)
            self._this_step_decision[agent_id] = is_due

            if is_due:
                relay = self._resolve_waypoint(agent_id, int(action[0]))
                self._relay_target[agent_id] = relay
                if relay is not None:
                    self._refresh_path_to(agent_id, relay)
                else:
                    self._refresh_path_to_goal(agent_id)
                self._wp_counter[agent_id] = 0
                self._decision_due[agent_id] = False

        # ── Phase 2: snapshot goal positions to detect new goal later ───
        pre_goals = {
            aid: (self.agent_map[aid].goal_x, self.agent_map[aid].goal_y)
            for aid in actions
        }

        # ── Phase 3: override action[0] with heuristic / idle ───────────
        nav_actions: dict = {}
        for agent_id, action in actions.items():
            if self._hold_left.get(agent_id, 0) > 0:
                move = 0  # idle during relay hold
            else:
                move = self.heuristic_move_action(agent_id)
            nav_act = np.array(action, copy=True)
            nav_act[0] = move
            nav_actions[agent_id] = nav_act

        # ── Phase 4: base step (movement + comm + rewards) ──────────────
        obs, rewards, terms, truncs, infos = super().step(nav_actions)

        # ── Phase 5: movement step penalty (exempt when idle / hold) ────
        for agent_id, nav_act in nav_actions.items():
            if agent_id in rewards and nav_act[0] != 0:
                rewards[agent_id] += self.STEP_PENALTY

        # ── Phase 6: post-step waypoint state updates ────────────────────
        for agent_id in list(actions.keys()):
            agent = self.agent_map.get(agent_id)
            if agent is None:
                continue

            # New goal assigned by base class on arrival → reset everything
            new_goal = (agent.goal_x, agent.goal_y)
            if new_goal != pre_goals[agent_id]:
                self._relay_target[agent_id] = None
                self._hold_left[agent_id] = 0
                self._wp_counter[agent_id] = 0
                self._decision_due[agent_id] = True
                self._refresh_path_to_goal(agent_id)
                continue

            # Relay hold countdown
            if self._hold_left.get(agent_id, 0) > 0:
                self._hold_left[agent_id] -= 1
                if self._hold_left[agent_id] == 0:
                    self._relay_target[agent_id] = None
                    self._refresh_path_to_goal(agent_id)
                    self._wp_counter[agent_id] = 0
                    self._decision_due[agent_id] = True
                continue

            # Relay arrival detection
            relay = self._relay_target.get(agent_id)
            if relay is not None:
                rx, ry = relay
                dist = float(np.sqrt((agent.x - rx) ** 2 + (agent.y - ry) ** 2))
                if dist < self.MAX_DIST_PER_STEP * 4.0:
                    if self.RELAY_HOLD > 0:
                        self._hold_left[agent_id] = self.RELAY_HOLD
                        agent.nav_path = []    # idle during hold
                    else:
                        self._relay_target[agent_id] = None
                        self._refresh_path_to_goal(agent_id)
                        self._wp_counter[agent_id] = 0
                        self._decision_due[agent_id] = True
                    continue

            # Decision timer
            self._wp_counter[agent_id] = self._wp_counter.get(agent_id, 0) + 1
            if self._wp_counter[agent_id] >= self.WAYPOINT_INTERVAL:
                self._decision_due[agent_id] = True
                self._wp_counter[agent_id] = 0

            # Lazy path refresh if nav_path depleted (goal nav only)
            if (not agent.nav_path
                    and relay is None
                    and not self.mission_done.get(agent_id, False)):
                self._refresh_path_to_goal(agent_id)

        return obs, rewards, terms, truncs, infos