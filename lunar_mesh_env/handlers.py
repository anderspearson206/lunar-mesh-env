# handlers.py

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Dict
from mobile_env.handlers.handler import Handler

class MeshCentralHandler(Handler):
    """
    handler for the mesh network.
    It assumes actions are agent-to-agent and observations are agent-centric.
    """
    @classmethod
    def action_space(cls, env) -> spaces.MultiDiscrete:
        """
        - Uses `env.agents.values()` instead of `env.users`.
        - Action size is `env.NUM_AGENTS + 1` (0=NOOP, 1-N=connect to agent_id).
        """
        # this is weird for polymorphism
        return spaces.MultiDiscrete([env.NUM_AGENTS + 1 for _ in env.agents.values()])

    @classmethod
    def observation_space(cls, env) -> spaces.Space:
        """
        Placeholder: Returns a Box space based on the new feature_sizes.
        need to implement features() for this
        """
        agent_obs_space = spaces.Dict(
            {
                "connections": spaces.Box(0, 1, (env.NUM_AGENTS,), dtype=np.float32),
                "snrs": spaces.Box(0, np.inf, (env.NUM_AGENTS,), dtype=np.float32),
                "utility": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "peer_utilities": spaces.Box(-np.inf, np.inf, (env.NUM_AGENTS,), dtype=np.float32),
                "peer_connections": spaces.Box(0, np.inf, (env.NUM_AGENTS,), dtype=np.float32),
            }
        )
        return spaces.Dict({agent_id: agent_obs_space for agent_id in env.agents.keys()})


    @classmethod
    def action(cls, env, action: np.ndarray) -> Dict[int, int]:
        """
        converts the MultiDiscrete array into a dict mapping agent_id -> action
        """
        return {agent_id: act for agent_id, act in zip(sorted(env.agents.keys()), action)}

    @classmethod
    def reward(cls, env) -> Dict[int, float]:
        """
        placeholder: Returns 0 reward for all active agents.
        """
        return {agent_id: 0.0 for agent_id in env.agents.keys()}
    
    @classmethod
    def info(cls, env) -> Dict:
        """placeholder: Returns an empty info dict."""
        return {}

    @classmethod
    def check(cls, env):
        """placeholder: No checks needed for this handler."""
        pass