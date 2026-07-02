"""
Experiment G — A* navigation + touched reward + epsilon-greedy comm.

Combines all three fixes:
  - A* handles movement so the GAT receives all policy gradient
  - eps_comm forces epidemic during early training, seeding the value
    function with high-delivery episodes while A* keeps navigation on track
  - Touched reward credits every relay agent when a packet reaches the BS

MRO: AStarEpsComm → AStarComm → TouchedReward → MLPGATEnv
     step() override lives in AStarComm (replaces action[0] with A*)
     _handle_communication_step() override lives here (eps override)
     then TouchedReward adds relay credit, then base does actual comm
"""

import random

from .marl_env_astar_comm import LunarRoverMeshAStarCommEnv


class LunarRoverMeshAStarEpsCommEnv(LunarRoverMeshAStarCommEnv):
    eps_comm: float = 1.0  # set by EpsCommDecayCallback during training

    def _handle_communication_step(self, actions, rewards, infos):
        if self.eps_comm > 0.0:
            actions = dict(actions)
            for agent_id, action in list(actions.items()):
                if random.random() < self.eps_comm:
                    agent  = self.agent_map[agent_id]
                    action = action.copy()
                    others = sorted(
                        (self.agent_map[a] for a in self.possible_agents if a != agent_id),
                        key=lambda a: a.ue_id,
                    )
                    for i, peer in enumerate(others):
                        action[1 + i] = 1 if peer in agent.neighbors else 0
                    action[-1] = 1 if agent.bs_connected else 0
                    actions[agent_id] = action

        # super() → TouchedRewardEnv (relay credit) → MLPGATEnv (actual comm)
        return super()._handle_communication_step(actions, rewards, infos)
