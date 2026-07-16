"""
Lozano-style comm env with RL-controlled navigation.

Same as LunarRoverMeshLozanoEnv except:
  - Inherits directly from LunarRoverMeshTouchedRewardEnv (no A* override)
  - Action space: MultiDiscrete([9, MAX_NODES]) — RL picks move direction AND comm edge
  - 9-way movement mask from _compute_move_mask (terrain + energy aware)

This is the fair RL-vs-RL comparison:
  train_ppo_lozano.py      — A* nav,  single-edge comm (comm policy only)
  train_ppo_lozano_rl_nav  — RL nav,  single-edge comm (joint nav+comm policy)
"""

import functools
import random
import numpy as np
from gymnasium import spaces

from .marl_env_touched_reward import LunarRoverMeshTouchedRewardEnv
from .marl_env import PACKET_SIZE_BITS, BURST_SIZE_MBITS, BURST_PROBABILITY


class LunarRoverMeshLozanoRLNavEnv(LunarRoverMeshTouchedRewardEnv):
    eps_nav: float = 1.0  # set by NavCurriculumCallback; 1.0=always hold, 0.0=full policy

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        max_nodes = len(self.possible_agents) + 1   # peers + BS
        return spaces.MultiDiscrete([9, max_nodes])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        base = super().observation_space(agent)
        spaces_dict = dict(base.spaces)
        max_nodes = len(self.possible_agents) + 1
        # 9 movement directions + MAX_NODES comm edge options
        spaces_dict["action_mask"] = spaces.Box(0, 1, shape=(9 + max_nodes,), dtype=np.int8)
        return spaces.Dict(spaces_dict)

    def _get_obs(self, agent_id):
        obs   = super()._get_obs(agent_id)
        agent = self.agent_map[agent_id]

        move_mask = self._compute_move_mask(agent)          # (9,)

        non_self  = self._get_non_self_nodes(agent_id)
        comm_mask = np.ones(len(non_self) + 1, dtype=np.int8)  # hold always valid
        for i, target in enumerate(non_self):
            if target is self.base_station:
                comm_mask[i + 1] = 1 if agent.bs_connected else 0
            else:
                rssi = self.radio_model.get_signal_strength(
                    agent.x, agent.y, target.x, target.y)
                comm_mask[i + 1] = 1 if rssi > self.MIN_DBM_THRESHOLD else 0

        obs["action_mask"] = np.concatenate([move_mask, comm_mask]).astype(np.int8)
        return obs

    # ------------------------------------------------------------------
    # Helper: ordered list of non-self nodes matching graph node indices
    # ------------------------------------------------------------------

    def _get_non_self_nodes(self, agent_id):
        """[*other_rovers_sorted_by_ue_id, base_station] — same order as graph nodes 1..N."""
        others = sorted(
            [self.agent_map[a] for a in self.possible_agents if a != agent_id],
            key=lambda a: a.ue_id,
        )
        return others + [self.base_station]

    # ------------------------------------------------------------------
    # Comm step: single-edge selection (action[1] = comm edge index)
    # ------------------------------------------------------------------

    def _handle_communication_step(self, actions, rewards, infos):
        if self.eps_nav > 0.0:
            actions = dict(actions)
            for agent_id, action in list(actions.items()):
                if random.random() < self.eps_nav:
                    action = action.copy()
                    action[1] = 0  # force hold — no comm this step
                    actions[agent_id] = action

        spray_copies = self.spray_copies if self.routing_protocol == 'spray_and_wait' else 1
        active_agents = [
            self.agent_map[aid] for aid in actions
            if self.agent_map[aid].energy > 0
        ]

        for agent in active_agents:
            if agent.bs_connected:
                agent.network_state["BS_0"].update(self.base_station.packets_received)
        for agent in active_agents:
            for neighbor in agent.neighbors:
                agent.merge_network_state(neighbor.network_state)

        for agent in active_agents:
            agent.cleanup_buffer()
            agent.drop_expired_packets(self.sim_time)

        for agent_id, action in actions.items():
            agent = self.agent_map[agent_id]
            if agent.energy <= 0:
                continue

            edge_idx = int(action[1])
            if edge_idx > 0:
                non_self = self._get_non_self_nodes(agent_id)
                target   = non_self[edge_idx - 1]

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

            # Packet generation
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
                    num_burst  = int(bits_burst / PACKET_SIZE_BITS)
                    dtn_state  = agent.payload_manager.get_state()
                    space_left = dtn_state['buffer_size'] - dtn_state['payload_size']
                    for _ in range(min(num_burst, int(space_left / PACKET_SIZE_BITS))):
                        agent.generate_packet(
                            size=PACKET_SIZE_BITS, time_to_live=5000,
                            destination="BS_0", time=self.sim_time,
                            spray_copies=spray_copies,
                        )

        return actions, rewards, infos
