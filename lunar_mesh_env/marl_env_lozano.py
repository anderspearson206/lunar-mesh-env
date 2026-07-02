"""
Lozano-style comm env: single-edge selection + rate-fill forwarding.

Two action_type modes:
  'multidiscrete' (default, for PPO):
      MultiDiscrete([1, MAX_NODES])
      action[0] = movement idle slot (A* overrides), action[1] = edge index
  'discrete' (for DQN — RLlib DQN only supports Discrete):
      Discrete(MAX_NODES)
      action = edge index directly (0=hold, 1..N-2=peers, N-1=BS)

Transmission uses send_packets_to_target (rate-fill, sender keeps copies).
Navigation is A* (inherited from LunarRoverMeshAStarCommEnv).
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env_astar_comm import LunarRoverMeshAStarCommEnv
from .marl_env import PACKET_SIZE_BITS, BURST_SIZE_MBITS, BURST_PROBABILITY


class LunarRoverMeshLozanoEnv(LunarRoverMeshAStarCommEnv):

    def __init__(self, *args, action_type='multidiscrete', **kwargs):
        self._action_type = action_type   # must be set before super().__init__ calls observation_space()
        super().__init__(*args, **kwargs)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        max_nodes = len(self.possible_agents) + 1
        if self._action_type == 'discrete':
            return spaces.Discrete(max_nodes)
        # multidiscrete: movement slot (idle only) + edge slot
        return spaces.MultiDiscrete([1, max_nodes])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        spaces_dict = dict(base.spaces)
        max_nodes = len(self.possible_agents) + 1
        if self._action_type == 'discrete':
            # Discrete mode: mask covers only comm edges (no movement slot)
            mask_dim = max_nodes
        else:
            # MultiDiscrete mode: 1 (idle always valid) + MAX_NODES edge options
            mask_dim = 1 + max_nodes
        spaces_dict["action_mask"] = spaces.Box(0, 1, shape=(mask_dim,), dtype=np.int8)
        return spaces.Dict(spaces_dict)

    def _get_obs(self, agent_id):
        obs   = super()._get_obs(agent_id)
        agent = self.agent_map[agent_id]

        non_self  = self._get_non_self_nodes(agent_id)

        comm_mask = np.ones(len(non_self) + 1, dtype=np.int8)  # hold always valid
        for i, target in enumerate(non_self):
            if target is self.base_station:
                comm_mask[i + 1] = 1 if agent.bs_connected else 0
            else:
                rssi = self.radio_model.get_signal_strength(
                    agent.x, agent.y, target.x, target.y)
                comm_mask[i + 1] = 1 if rssi > self.MIN_DBM_THRESHOLD else 0

        if self._action_type == 'discrete':
            obs["action_mask"] = comm_mask
        else:
            move_mask = np.ones(1, dtype=np.int8)
            obs["action_mask"] = np.concatenate([move_mask, comm_mask]).astype(np.int8)
        return obs

    # ------------------------------------------------------------------
    # Helper: ordered list of non-self nodes matching graph node indices
    # ------------------------------------------------------------------

    def _get_non_self_nodes(self, agent_id):
        """Returns [*other_rovers_sorted_by_ue_id, base_station].

        Index i in this list corresponds to graph node (i+1), so
        action[1] = k maps to non_self[k-1].
        """
        others = sorted(
            [self.agent_map[a] for a in self.possible_agents if a != agent_id],
            key=lambda a: a.ue_id,
        )
        return others + [self.base_station]

    # ------------------------------------------------------------------
    # Comm step: single-edge selection replaces binary per-target flags
    # ------------------------------------------------------------------

    def step(self, actions):
        if self._action_type != 'discrete':
            return super().step(actions)
        # Discrete mode: action is a scalar comm edge index; A* still controls movement.
        # Wrap to [move, edge] so the base env's step can process it normally,
        # bypassing AStarCommEnv.step() which would try new_action[0]=move on a scalar.
        from .marl_env_astar_comm import LunarRoverMeshAStarCommEnv
        wrapped = {}
        for aid, action in actions.items():
            move = self.heuristic_move_action(aid)
            wrapped[aid] = np.array([move, int(action)])
        result = super(LunarRoverMeshAStarCommEnv, self).step(wrapped)
        for agent in self.agent_map.values():
            if not agent.nav_path and not self.mission_done.get(agent.id, False):
                self._compute_path(agent)
        return result

    def _handle_communication_step(self, actions, rewards, infos):
        spray_copies = self.spray_copies if self.routing_protocol == 'spray_and_wait' else 1
        active_agents = [
            self.agent_map[aid] for aid in actions
            if self.agent_map[aid].energy > 0
        ]

        # Network state sync (same as base)
        for agent in active_agents:
            if agent.bs_connected:
                agent.network_state["BS_0"].update(self.base_station.packets_received)
        for agent in active_agents:
            for neighbor in agent.neighbors:
                agent.merge_network_state(neighbor.network_state)

        # Buffer cleanup (same as base)
        for agent in active_agents:
            agent.cleanup_buffer()
            agent.drop_expired_packets(self.sim_time)

        # Per-agent comm + packet generation
        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                continue

            # --- Lozano single-edge selection ---
            # action may be scalar (original discrete) or [move, edge] (wrapped by step())
            edge_idx = int(action[1] if np.ndim(action) > 0 else action)
            if edge_idx > 0:
                non_self = self._get_non_self_nodes(agent_id)
                target = non_self[edge_idx - 1]

                # Check reachability
                if target is self.base_station:
                    reachable = agent.bs_connected
                else:
                    reachable = target in agent.neighbors

                if reachable:
                    pre = self.base_station.num_packets_received
                    agent.send_packets_to_target(target, self.STEP_LENGTH, self.sim_time)
                    delivered = self.base_station.num_packets_received - pre
                    if delivered > 0:
                        rewards[agent_id] += delivered * self.REWARD_PACKET_DELIVERY
                    link_color = 'cyan' if target is self.base_station else 'green'
                    self.custom_links[(agent, target)] = link_color

            # --- Packet generation (same as base) ---
            if self.packet_mode == 'boolean':
                if (np.random.rand() < self.PACKET_GEN_PROB
                        and not self.mission_done.get(agent_id, False)):
                    agent.generate_packet(
                        size=10, time_to_live=50,
                        destination="BS_0", time=self.sim_time,
                        spray_copies=spray_copies,
                    )
            else:
                bits_needed = self.TELEMETRY_RATE_MBPS * self.STEP_LENGTH * 1e6
                num_telemetry = int(bits_needed / PACKET_SIZE_BITS)
                for _ in range(num_telemetry):
                    agent.generate_packet(
                        size=PACKET_SIZE_BITS, time_to_live=5000,
                        destination="BS_0", time=self.sim_time,
                        spray_copies=spray_copies,
                    )
                if np.random.rand() < BURST_PROBABILITY:
                    bits_burst = BURST_SIZE_MBITS * 1e6
                    num_burst = int(bits_burst / PACKET_SIZE_BITS)
                    dtn_state = agent.payload_manager.get_state()
                    space_left = dtn_state['buffer_size'] - dtn_state['payload_size']
                    packets_to_gen = min(num_burst, int(space_left / PACKET_SIZE_BITS))
                    for _ in range(packets_to_gen):
                        agent.generate_packet(
                            size=PACKET_SIZE_BITS, time_to_live=5000,
                            destination="BS_0", time=self.sim_time,
                            spray_copies=spray_copies,
                        )

        return actions, rewards, infos
