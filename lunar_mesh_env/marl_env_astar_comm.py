"""
Experiment F — A* navigation + RL communication.

The MLP movement head is bypassed: action[0] is replaced by the output of
heuristic_move_action() (A* pure-pursuit) before the env step executes.
Only the GAT comm head is trained by RL.

Benefit: the dense movement gradient (dist_delta * 2.0, every step) no
longer competes with the sparse comm gradient. Navigation rewards still
appear in the value function as a predictable baseline; the advantage
captures only the variation due to comm decisions, giving the GAT clean
signal.

Touched reward (Exp D) is inherited for proper relay credit assignment.
"""

import numpy as np

from .pathfinding import a_star_search_rm as a_star_search
from .marl_env_touched_reward import LunarRoverMeshTouchedRewardEnv


class LunarRoverMeshAStarCommEnv(LunarRoverMeshTouchedRewardEnv):

    def _astar_threshold(self):
        return (self.MAX_INCLINE_PER_STEP / max(1.0, self.MAX_DIST_PER_STEP)) * 0.9

    def _compute_path(self, agent):
        start = (int(agent.x),      int(agent.y))
        goal  = (int(agent.goal_x), int(agent.goal_y))
        path  = a_star_search(
            self.heightmap, self.bs_radio_map,
            start, goal,
            self._astar_threshold(), self.radio_bias,
        )
        agent.nav_path = path if path else []

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        for agent in self.agent_map.values():
            self._compute_path(agent)
        return obs, info

    def step(self, actions):
        # Replace the policy's movement output with A* direction
        astar_actions = {}
        for aid, action in actions.items():
            move        = self.heuristic_move_action(aid)
            new_action  = np.array(action)
            new_action[0] = move
            astar_actions[aid] = new_action

        result = super().step(astar_actions)

        # Recompute path for agents whose nav_path was cleared by goal arrival
        for agent in self.agent_map.values():
            if not agent.nav_path and not self.mission_done.get(agent.id, False):
                self._compute_path(agent)

        return result
