"""
Graph-observation variant of LunarRoverMeshEnv for the MLP-GAT hybrid policy.

Replaces the 256×256 terrain/radio-map observations with a compact local
1-hop subgraph (node feature matrix + adjacency vector).  Scalars used by
the MLP branch (position, goal_vector, move_history) are retained.

Node features (NODE_FEAT_DIM = 7) per node:
  [is_self, is_lander, bs_connected, buffer_fill,
   dist_to_lander_norm, x_norm, y_norm]

Node ordering (always):
  0        : observing agent (self)
  1..N-2   : other rovers sorted by ue_id
  N-1      : base station (lander)
"""

import functools
import numpy as np
from gymnasium import spaces

from .marl_env import LunarRoverMeshEnv

NODE_FEAT_DIM = 7


class LunarRoverMeshMLPGATEnv(LunarRoverMeshEnv):
    """
    Identical to LunarRoverMeshEnv except _get_obs returns graph-structured
    observations instead of full 256×256 radio/terrain maps.
    """

    def _build_graph_obs(self, agent_id):
        """
        Returns:
          node_features : (MAX_NODES, NODE_FEAT_DIM) float32
          adj_1d        : (MAX_NODES,) float32 — 1 if self can reach node j
        Node 0 is always the observing agent.
        """
        agent   = self.agent_map[agent_id]
        others  = sorted(
            [self.agent_map[a] for a in self.possible_agents if a != agent_id],
            key=lambda a: a.ue_id,
        )
        bs_x, bs_y = self.base_station.x, self.base_station.y
        max_dist    = np.sqrt(self.width ** 2 + self.height ** 2)
        MAX_NODES   = len(self.possible_agents) + 1

        node_feats = np.zeros((MAX_NODES, NODE_FEAT_DIM), dtype=np.float32)
        adj        = np.zeros(MAX_NODES, dtype=np.float32)

        def _make_feat(is_self, is_lander, x, y, bs_connected, buf_fill):
            dist = np.sqrt((x - bs_x) ** 2 + (y - bs_y) ** 2) / max_dist
            return [
                float(is_self),
                float(is_lander),
                float(bs_connected),
                float(np.clip(buf_fill, 0.0, 1.0)),
                float(np.clip(dist, 0.0, 1.0)),
                x / self.width,
                y / self.height,
            ]

        # Node 0: self
        dtn = agent.payload_manager.get_state()
        buf = float(dtn["payload_size"]) / max(float(dtn["buffer_size"]), 1.0)
        node_feats[0] = _make_feat(True, False, agent.x, agent.y, agent.bs_connected, buf)
        adj[0] = 1.0  # self-loop

        # Nodes 1..N-2: other rovers sorted by ue_id
        for i, other in enumerate(others):
            dtn = other.payload_manager.get_state()
            buf = float(dtn["payload_size"]) / max(float(dtn["buffer_size"]), 1.0)
            node_feats[i + 1] = _make_feat(False, False, other.x, other.y, other.bs_connected, buf)
            adj[i + 1] = 1.0 if other in agent.neighbors else 0.0

        # Node N-1: base station (lander)
        node_feats[-1] = _make_feat(False, True, bs_x, bs_y, True, 0.0)
        adj[-1] = 1.0 if agent.bs_connected else 0.0

        return node_feats, adj

    def _peer_targets_from_flags(self, agent, agent_id, comm_flags_peer):
        """Graph order: other rovers sorted by ue_id, self excluded."""
        others = sorted(
            (self.agent_map[a] for a in self.possible_agents if a != agent_id),
            key=lambda a: a.ue_id,
        )
        return [others[i] for i, f in enumerate(comm_flags_peer)
                if f == 1 and others[i] in agent.neighbors]

    def _get_obs(self, agent_id):
        agent    = self.agent_map[agent_id]
        obs_dict = agent.get_local_observation(list(self.agent_map.values()))

        # Action mask — comm targets: other rovers (ue_id order, self excluded) then BS.
        # No self-slot: action space is [9, 2*(N-1), 2] = [9,2,2,2] for 3 agents.
        move_mask  = self._compute_move_mask(agent)
        comm_masks = []
        others = sorted(
            (self.agent_map[a] for a in self.possible_agents if a != agent_id),
            key=lambda a: a.ue_id,
        )
        for other in others:
            rssi = self.radio_model.get_signal_strength(
                agent.x, agent.y, other.x, other.y
            )
            comm_masks.append([1, 1] if rssi > self.MIN_DBM_THRESHOLD else [1, 0])
        rssi_bs = self.radio_model.get_signal_strength(
            agent.x, agent.y, self.base_station.x, self.base_station.y
        )
        comm_masks.append([1, 1] if rssi_bs > self.MIN_DBM_THRESHOLD else [1, 0])
        action_mask = np.concatenate(
            [move_mask, np.array(comm_masks).flatten()]
        ).astype(np.int8)

        node_feats, adj = self._build_graph_obs(agent_id)
        move_history    = self._move_history[agent_id] * (256.0 / 8.0)

        return {
            "action_mask":              action_mask,
            "graph_node_features":      node_feats,
            "graph_adj":                adj,
            "goal_vector":              np.clip(obs_dict["goal_vector"], -256.0, 256.0),
            "move_history":             move_history,
            "other_agent_connectivity": np.clip(
                obs_dict["other_agent_connectivity"].astype(np.float32), 0.0, 1.0
            ),
            "position":                 np.clip(obs_dict["position"], 0.0, 256.0),
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num_rovers = len(self.possible_agents)
        # comm targets: (num_rovers-1) peers + BS = num_rovers targets, no self-slot
        mask_dim   = 9 + num_rovers * 2
        MAX_NODES  = num_rovers + 1

        return spaces.Dict({
            "action_mask":              spaces.Box(0, 1,      shape=(mask_dim,),             dtype=np.int8),
            "graph_node_features":      spaces.Box(-np.inf, np.inf,
                                                   shape=(MAX_NODES, NODE_FEAT_DIM),         dtype=np.float32),
            "graph_adj":                spaces.Box(0, 1,      shape=(MAX_NODES,),            dtype=np.float32),
            "goal_vector":              spaces.Box(-256, 256, shape=(2,),                    dtype=np.float32),
            "move_history":             spaces.Box(0, 256,    shape=(self.HISTORY_LEN,),     dtype=np.float32),
            "other_agent_connectivity": spaces.Box(0, 1,      shape=(num_rovers + 1,),       dtype=np.float32),
            "position":                 spaces.Box(0, 256,    shape=(2,),                    dtype=np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, _agent):
        # [movement(9)] + [comm_peer(2)] * (num_rovers-1) + [comm_BS(2)]
        # = num_rovers comm targets total (self excluded)
        num_rovers = len(self.possible_agents)
        return spaces.MultiDiscrete([9] + [2] * num_rovers)
