"""
Experiment H — A* navigation + immediate forward reward + touched relay credit.

Combines:
  - A* navigation (AStarComm.step() replaces action[0])
  - Per-peer-transfer forward reward: fires immediately when a packet
    moves to a peer rover — no BS delivery delay needed
  - Touched relay credit: fires at BS delivery for every relay agent

MRO: AStarForwardReward → AStarComm → TouchedReward → ForwardReward → MLPGATEnv

_handle_communication_step() chain:
  1. TouchedReward:   snapshot BS packets, call super(), give touched credit
  2. ForwardReward:   full send loop with buffer-size tracking per peer send
"""

from .marl_env_astar_comm import LunarRoverMeshAStarCommEnv
from .marl_env_forward_reward import LunarRoverMeshForwardRewardEnv


class LunarRoverMeshAStarForwardRewardEnv(
    LunarRoverMeshAStarCommEnv,
    LunarRoverMeshForwardRewardEnv,
):
    pass
