"""
Experiment C — Epsilon-greedy communication.

With probability eps_comm, the policy's comm decisions are overridden with
epidemic routing (send to all reachable peers and the BS).  A callback in
the training script schedules eps_comm linearly from 1.0 → 0.0 over the
first half of training, bootstrapping the value function on high-delivery
episodes before handing control to the RL comm head.
"""

import random

from .marl_env_mlp_gat import LunarRoverMeshMLPGATEnv


class LunarRoverMeshEpsCommEnv(LunarRoverMeshMLPGATEnv):
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

        return super()._handle_communication_step(actions, rewards, infos)
