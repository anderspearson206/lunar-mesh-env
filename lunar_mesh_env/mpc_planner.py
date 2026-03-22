# mpc_planner.py

import numpy as np
from typing import List, Tuple, Optional

from .radio_model_nn import RadioMapModelNN


# 8 direction angles matching the env's action encoding
# Action 0 = idle, Actions 1-8 = N, S, W, E, NE, NW, SE, SW
_ACTION_ANGLES = {
    1: np.pi / 2,        # N  (dy=+1)
    2: -np.pi / 2,       # S  (dy=-1)
    3: np.pi,            # W  (dx=-1)
    4: 0.0,              # E  (dx=+1)
    5: np.pi / 4,        # NE
    6: 3 * np.pi / 4,    # NW
    7: -np.pi / 4,       # SE
    8: -3 * np.pi / 4,   # SW
}

# Reverse: sorted list of (angle, action_id) for discretization
_ANGLE_TO_ACTION = sorted(
    [(angle, act) for act, angle in _ACTION_ANGLES.items()],
    key=lambda x: x[0],
)
_DISC_ANGLES = np.array([a for a, _ in _ANGLE_TO_ACTION])
_DISC_ACTIONS = np.array([a for _, a in _ANGLE_TO_ACTION])


def _angle_to_action(angle: float) -> int:
    """Map a continuous angle (radians) to the nearest discrete action (1-8)."""
    # Wrap angle to [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    diffs = np.abs(_DISC_ANGLES - angle)
    # Handle wrap-around at +/- pi
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    return int(_DISC_ACTIONS[np.argmin(diffs)])


def _action_to_direction(action: int) -> Tuple[float, float]:
    """Return (dx, dy) unit-ish vector for an action, matching the env."""
    if action == 0:
        return 0.0, 0.0
    if action == 1: return 0.0, 1.0
    if action == 2: return 0.0, -1.0
    if action == 3: return -1.0, 0.0
    if action == 4: return 1.0, 0.0
    if action == 5: return 0.707, 0.707
    if action == 6: return -0.707, 0.707
    if action == 7: return 0.707, -0.707
    if action == 8: return -0.707, -0.707
    return 0.0, 0.0


class MPPIPlanner:
    """
    Model Predictive Path Integral (MPPI) planner for lunar rover communication
    optimization. Plans movement trajectories that balance goal navigation with
    mesh network quality using the diffusion radio model as a forward predictor.
    """

    def __init__(
        self,
        radio_model: RadioMapModelNN,
        heightmap: np.ndarray,
        env_width: float = 256.0,
        env_height: float = 256.0,
        horizon: int = 15,
        num_samples: int = 128,
        temperature: float = 1.0,
        noise_sigma: float = 0.8,
        max_dist_per_step: float = 4.0,
        max_incline: float = 1.0,
        min_dbm: float = -82.0,
        w_comm: float = 1.0,
        w_bs: float = 5.0,
        w_goal: float = 2.0,
        w_energy: float = 0.1,
        agent=None,
        dt_s: float = 0.1,
        bs_decay_rate: float = 0.9,
        radio_batch_size: int = 8,
        radio_eval_stride: int = 3,
        radio_pos_quant: int = 4,
    ):
        self.radio_model = radio_model
        self.heightmap = heightmap
        self.env_width = env_width
        self.env_height = env_height
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.max_dist_per_step = max_dist_per_step
        self.max_incline = max_incline
        self.min_dbm = min_dbm

        self.w_comm = w_comm
        self.w_bs = w_bs
        self.w_goal = w_goal
        self.w_energy = w_energy
        self.agent = agent
        self.dt_s = dt_s
        self.bs_decay_rate = bs_decay_rate
        self.radio_batch_size = radio_batch_size
        self.radio_eval_stride = radio_eval_stride
        self.radio_pos_quant = radio_pos_quant

        # Tracks consecutive steps the rover is in BS coverage
        self.bs_dwell_steps = 0

        # Nominal angle sequence (warm-started between calls)
        self.nominal: Optional[np.ndarray] = None
        self._prev_goal: Optional[Tuple[float, float]] = None

        # Debug: last rollout data (populated when plan() is called)
        self.last_rollout: Optional[dict] = None

    def _init_nominal(self, agent_pos: Tuple[float, float],
                      goal_pos: Tuple[float, float]) -> np.ndarray:
        """Initialize nominal sequence pointing toward the goal."""
        dx = goal_pos[0] - agent_pos[0]
        dy = goal_pos[1] - agent_pos[1]
        angle = np.arctan2(dy, dx)
        return np.full(self.horizon, angle, dtype=np.float64)

    def _rollout_positions(
        self, start: np.ndarray, angles: np.ndarray,
        goal_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roll out positions for all samples simultaneously.

        Args:
            start: (2,) starting position
            angles: (N, H) angle sequences
            goal_pos: (2,) if provided, trajectories stop advancing once
                      within max_dist_per_step of the goal

        Returns:
            positions: (N, H+1, 2) trajectory positions (includes start)
            valid: (N, H) boolean mask of slope-valid steps
        """
        N, H = angles.shape
        positions = np.zeros((N, H + 1, 2), dtype=np.float64)
        positions[:, 0, :] = start
        valid = np.ones((N, H), dtype=bool)
        # Track which samples have already reached the goal
        arrived = np.zeros(N, dtype=bool)

        hm = self.heightmap
        hm_h, hm_w = hm.shape[:2]

        for k in range(H):
            dx = np.cos(angles[:, k]) * self.max_dist_per_step
            dy = np.sin(angles[:, k]) * self.max_dist_per_step

            new_x = np.clip(positions[:, k, 0] + dx, 0, self.env_width - 1)
            new_y = np.clip(positions[:, k, 1] + dy, 0, self.env_height - 1)

            # Slope check (vectorized)
            cx = np.clip(positions[:, k, 0].astype(int), 0, hm_w - 1)
            cy = np.clip(positions[:, k, 1].astype(int), 0, hm_h - 1)
            nx = np.clip(new_x.astype(int), 0, hm_w - 1)
            ny = np.clip(new_y.astype(int), 0, hm_h - 1)

            z_curr = hm[cy, cx]
            z_next = hm[ny, nx]
            dz = z_next - z_curr

            slope_ok = dz <= self.max_incline
            valid[:, k] = slope_ok

            # Blocked moves stay in place
            final_x = np.where(slope_ok, new_x, positions[:, k, 0])
            final_y = np.where(slope_ok, new_y, positions[:, k, 1])

            # Clamp samples that already reached the goal
            if goal_pos is not None:
                final_x = np.where(arrived, positions[:, k, 0], final_x)
                final_y = np.where(arrived, positions[:, k, 1], final_y)
                dist_to_goal = np.sqrt(
                    (final_x - goal_pos[0]) ** 2 + (final_y - goal_pos[1]) ** 2
                )
                arrived |= dist_to_goal < self.max_dist_per_step

            positions[:, k + 1, 0] = final_x
            positions[:, k + 1, 1] = final_y

        return positions, valid

    def _batch_radio_lookup(
        self, positions: np.ndarray
    ) -> dict:
        """
        Batch-generate radio maps for unique integer positions in the
        rolled-out trajectories. Only evaluates at strided horizon steps
        to limit the number of unique positions. Chunks inference into
        small sub-batches to avoid GPU OOM.

        Args:
            positions: (N, H+1, 2) array of positions

        Returns:
            maps: dict of {(ix, iy): np.ndarray(256, 256)}
        """
        # Only collect positions at the strided evaluation steps
        # (step 0 is start, step 1..H are the rolled-out steps)
        eval_steps = list(range(1, positions.shape[1], self.radio_eval_stride))
        # Always include the last step
        if eval_steps[-1] != positions.shape[1] - 1:
            eval_steps.append(positions.shape[1] - 1)

        subset = positions[:, eval_steps, :]  # (N, num_eval, 2)
        flat = subset.reshape(-1, 2)
        # Quantize to coarser grid to reduce unique positions
        q = self.radio_pos_quant
        ix = np.clip((flat[:, 0] / q).astype(int) * q, 0, int(self.env_width) - 1)
        iy = np.clip((flat[:, 1] / q).astype(int) * q, 0, int(self.env_height) - 1)

        unique_keys = set(zip(ix.tolist(), iy.tolist()))

        # Filter out positions already cached in the radio model
        uncached = []
        for key in unique_keys:
            cache_key = (key[0], key[1], '5.8')
            if cache_key not in self.radio_model.cache:
                uncached.append(key)

        # Chunked inference to avoid GPU OOM
        bs = self.radio_batch_size
        for i in range(0, len(uncached), bs):
            chunk = uncached[i : i + bs]
            self.radio_model.generate_map_batch(
                [(float(x), float(y)) for x, y in chunk], '5.8'
            )

        # Build lookup from cache
        maps = {}
        for key in unique_keys:
            cache_key = (key[0], key[1], '5.8')
            rm = self.radio_model.cache.get(cache_key)
            if rm is not None:
                maps[key] = rm

        return maps

    def _lookup_signal(
        self, radio_maps: dict, from_pos: np.ndarray, to_pos: np.ndarray
    ) -> np.ndarray:
        """
        Look up signal strength from from_pos to to_pos using pre-generated maps.

        Args:
            radio_maps: {(ix, iy): radio_map} dict
            from_pos: (N,2) transmitter positions
            to_pos: (2,) or (N,2) receiver position(s)

        Returns:
            rss: (N,) signal strengths in dBm
        """
        N = from_pos.shape[0]
        rss = np.full(N, -150.0)

        # Quantize transmitter positions to same grid used in _batch_radio_lookup
        q = self.radio_pos_quant
        from_ix = np.clip((from_pos[:, 0] / q).astype(int) * q, 0, int(self.env_width) - 1)
        from_iy = np.clip((from_pos[:, 1] / q).astype(int) * q, 0, int(self.env_height) - 1)

        # Receiver pixel coords
        if to_pos.ndim == 1:
            to_pos = np.broadcast_to(to_pos, (N, 2))
        r_idx = np.clip((to_pos[:, 1] / self.env_height * 255).astype(int), 0, 255)
        c_idx = np.clip((to_pos[:, 0] / self.env_width * 255).astype(int), 0, 255)

        for i in range(N):
            key = (int(from_ix[i]), int(from_iy[i]))
            rm = radio_maps.get(key)
            if rm is not None:
                rss[i] = rm[r_idx[i], c_idx[i]]

        return rss

    def _throughput_vectorized(self, rss: np.ndarray) -> np.ndarray:
        """Vectorized dBm -> Mbps conversion for 5.8 GHz band."""
        tp = np.zeros_like(rss)
        tp = np.where(rss >= -65, 54.0, tp)
        tp = np.where((rss >= -66) & (rss < -65), 48.0, tp)
        tp = np.where((rss >= -70) & (rss < -66), 36.0, tp)
        tp = np.where((rss >= -74) & (rss < -70), 24.0, tp)
        tp = np.where((rss >= -77) & (rss < -74), 18.0, tp)
        tp = np.where((rss >= -79) & (rss < -77), 12.0, tp)
        tp = np.where((rss >= -81) & (rss < -79), 9.0, tp)
        tp = np.where((rss >= -82) & (rss < -81), 6.0, tp)
        return tp

    def _compute_costs(
        self,
        positions: np.ndarray,
        valid: np.ndarray,
        goal_pos: np.ndarray,
        other_positions: List[np.ndarray],
        bs_pos: np.ndarray,
        radio_maps: dict,
        w_bs_effective: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute MPPI cost for each sample trajectory.

        Args:
            positions: (N, H+1, 2)
            valid: (N, H) slope validity
            goal_pos: (2,)
            other_positions: list of (2,) arrays for other rovers
            bs_pos: (2,)
            radio_maps: pre-built lookup dict

        Returns:
            costs: (N,)
        """
        N = positions.shape[0]
        H = self.horizon
        costs = np.zeros(N)

        # Steps where radio is evaluated (matches _batch_radio_lookup)
        radio_eval_steps = set(range(0, H, self.radio_eval_stride))
        # import pdb; pdb.set_trace()
        if H - 1 not in radio_eval_steps:
            radio_eval_steps.add(H - 1)

        for k in range(H):
            pos_k = positions[:, k + 1, :]       # (N, 2) current step position
            pos_prev = positions[:, k, :]         # (N, 2) previous position

            # --- Terrain infeasibility penalty ---
            costs += np.where(~valid[:, k], 1000.0, 0.0)
            # --- Communication (only at strided steps to match available maps) ---
            if k in radio_eval_steps:
                for other_pos in other_positions:
                    rss = self._lookup_signal(radio_maps, pos_k, np.array(other_pos))
                    tp = self._throughput_vectorized(rss)
                    costs -= self.w_comm * tp

                rss_bs = self._lookup_signal(radio_maps, pos_k, bs_pos)
                tp_bs = self._throughput_vectorized(rss_bs)
                w_bs_eff = w_bs_effective if w_bs_effective is not None else self.w_bs
                costs -= w_bs_eff * tp_bs

            # --- Goal progress ---
            dist_prev = np.sqrt(
                (pos_prev[:, 0] - goal_pos[0]) ** 2
                + (pos_prev[:, 1] - goal_pos[1]) ** 2
            )
            dist_curr = np.sqrt(
                (pos_k[:, 0] - goal_pos[0]) ** 2
                + (pos_k[:, 1] - goal_pos[1]) ** 2
            )
            # Negative delta = getting closer = good → subtract to reduce cost
            costs -= self.w_goal * (dist_prev - dist_curr)

            # --- Energy (uses agent's drivetrain model) ---
            if self.agent is not None:
                for i in range(N):
                    energy_j, _ = self.agent.compute_step_energy_joules(
                        curr_xy=(pos_prev[i, 0], pos_prev[i, 1]),
                        next_xy=(pos_k[i, 0], pos_k[i, 1]),
                        dt_s=self.dt_s,
                        max_incline_per_step_m=self.max_incline,
                    )
                    costs[i] += self.w_energy * energy_j

        # --- Terminal cost: remaining distance to goal ---
        final_pos = positions[:, -1, :]
        final_dist = np.sqrt(
            (final_pos[:, 0] - goal_pos[0]) ** 2
            + (final_pos[:, 1] - goal_pos[1]) ** 2
        )
        costs += self.w_goal * final_dist



        return costs

    def plan(
        self,
        agent_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        other_positions: List[Tuple[float, float]],
        bs_pos: Tuple[float, float],
    ) -> int:
        """
        Run one MPPI planning iteration and return the best discrete action.

        Args:
            agent_pos: (x, y) current rover position
            goal_pos: (x, y) navigation goal
            other_positions: list of (x, y) positions of other rovers
            bs_pos: (x, y) base station position

        Returns:
            action: int in [0, 8], discrete movement action
        """
        agent_pos = np.array(agent_pos, dtype=np.float64)
        goal_pos_arr = np.array(goal_pos, dtype=np.float64)
        bs_pos_arr = np.array(bs_pos, dtype=np.float64)
        other_pos_arr = [np.array(p, dtype=np.float64) for p in other_positions]

        # Check if near goal → idle
        dist_to_goal = np.linalg.norm(goal_pos_arr - agent_pos)
        if dist_to_goal < self.max_dist_per_step:
            return 0

        # Initialize or re-initialize nominal if goal changed
        if (self.nominal is None
                or self._prev_goal is None
                or self._prev_goal != goal_pos):
            self.nominal = self._init_nominal(tuple(agent_pos), goal_pos)
            self._prev_goal = goal_pos

        N = self.num_samples
        H = self.horizon

        # Sample angle perturbations around the nominal
        noise = np.random.randn(N, H) * self.noise_sigma
        angles = self.nominal[np.newaxis, :] + noise  # (N, H)

        # Roll out positions (clamp at goal to prevent overshoot)
        positions, valid = self._rollout_positions(agent_pos, angles, goal_pos=goal_pos_arr)

        # Batch radio map generation for all rolled-out positions
        radio_maps = self._batch_radio_lookup(positions)

        # BS dwell tracking: check if agent is currently in BS coverage
        agent_pos_2d = agent_pos.reshape(1, 2)
        rss_at_agent = self._lookup_signal(radio_maps, agent_pos_2d, bs_pos_arr)
        tp_at_agent = self._throughput_vectorized(rss_at_agent)
        if tp_at_agent[0] > 0:
            self.bs_dwell_steps += 1
        else:
            self.bs_dwell_steps = 0

        w_bs_effective = self.w_bs * (self.bs_decay_rate ** self.bs_dwell_steps)

        # Compute costs
        costs = self._compute_costs(
            positions, valid, goal_pos_arr, other_pos_arr, bs_pos_arr, radio_maps,
            w_bs_effective=w_bs_effective,
        )

        # MPPI weighting
        costs_shifted = costs - np.min(costs)  # numerical stability
        weights = np.exp(-costs_shifted / self.temperature)
        weights_sum = np.sum(weights)
        if weights_sum < 1e-10:
            weights = np.ones(N) / N
        else:
            weights = weights / weights_sum

        # Weighted average to update nominal
        self.nominal = np.sum(weights[:, np.newaxis] * angles, axis=0)

        # Extract first action and discretize
        best_angle = self.nominal[0]
        action = _angle_to_action(best_angle)

        # Validate the chosen action against terrain
        dx, dy = _action_to_direction(action)
        if dx != 0 or dy != 0:
            dist = np.sqrt(dx**2 + dy**2)
            scale = self.max_dist_per_step / dist
            new_x = np.clip(agent_pos[0] + dx * scale, 0, self.env_width - 1)
            new_y = np.clip(agent_pos[1] + dy * scale, 0, self.env_height - 1)

            hm = self.heightmap
            cx = int(np.clip(agent_pos[0], 0, hm.shape[1] - 1))
            cy = int(np.clip(agent_pos[1], 0, hm.shape[0] - 1))
            nx = int(np.clip(new_x, 0, hm.shape[1] - 1))
            ny = int(np.clip(new_y, 0, hm.shape[0] - 1))

            if hm[ny, nx] - hm[cy, cx] > self.max_incline:
                # Fall back to goal-directed heuristic with nearby alternatives
                goal_angle = np.arctan2(
                    goal_pos[1] - agent_pos[1],
                    goal_pos[0] - agent_pos[0]
                )
                for offset in [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]:
                    alt_action = _angle_to_action(goal_angle + offset)
                    adx, ady = _action_to_direction(alt_action)
                    if adx == 0 and ady == 0:
                        continue
                    adist = np.sqrt(adx**2 + ady**2)
                    ascale = self.max_dist_per_step / adist
                    anx = np.clip(agent_pos[0] + adx * ascale, 0, self.env_width - 1)
                    any_ = np.clip(agent_pos[1] + ady * ascale, 0, self.env_height - 1)
                    anxi = int(np.clip(anx, 0, hm.shape[1] - 1))
                    anyi = int(np.clip(any_, 0, hm.shape[0] - 1))
                    if hm[anyi, anxi] - hm[cy, cx] <= self.max_incline:
                        action = alt_action
                        break
                else:
                    action = 0  # idle if all blocked

        # Store rollout data for debug visualization
        dx, dy = _action_to_direction(action)
        self.last_rollout = {
            'positions': positions,   # (N, H+1, 2)
            'costs': costs,           # (N,)
            'weights': weights,       # (N,)
            'agent_pos': agent_pos,   # (2,) start position
            'chosen_action': action,
            'chosen_dx_dy': (dx, dy),
            'nominal_angle': best_angle,
        }

        # Warm-start: shift nominal left for next call
        self.nominal = np.append(self.nominal[1:], self.nominal[-1])

        return action
