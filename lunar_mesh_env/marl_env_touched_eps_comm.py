"""
Experiment E — Touched relay credit + epsilon-greedy comm.

Combines Exp D (touched-based credit assignment) with Exp C
(epsilon-greedy comm decay): eps_comm forces epidemic routing during
early training to seed exploration of relay behavior; the touched
reward gives proper gradient when those relays contribute to delivery.

The decay schedule (1.0 → 0.0 over first half of training) is applied
by EpsCommDecayCallback in train_ppo_touched_eps_comm.py.
"""

import random

from .marl_env_touched_reward import LunarRoverMeshTouchedRewardEnv


class LunarRoverMeshTouchedEpsCommEnv(LunarRoverMeshTouchedRewardEnv):
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

        # super() → LunarRoverMeshTouchedRewardEnv (touched credit)
        #          → LunarRoverMeshMLPGATEnv (actual comm step)
        return super()._handle_communication_step(actions, rewards, infos)
