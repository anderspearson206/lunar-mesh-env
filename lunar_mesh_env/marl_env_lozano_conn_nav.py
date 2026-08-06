"""
Connectivity-aware planned navigation + Lozano RL comm.

Navigation: radio-weighted A* (a_star_search_rm with RADIO_BIAS > 0) — the planner
explicitly favours paths with better BS signal, producing connectivity-aware detours
without any RL. Comm head: identical Lozano single-edge RL.

This is the key comparison bar for RL-learned navigation:
  - If waypoint-residual RL cannot beat this at heavy terrain shadowing, the result
    is a clean negative: navigation should be planned over the known radio map, not
    learned.
  - If RL wins, it found detour structure not captured by the radio-map cost alone.

Inherits from LunarRoverMeshLozanoEnv, which already uses a_star_search_rm via
LunarRoverMeshAStarCommEnv._compute_path with self.radio_bias as radio_weight.
Setting RADIO_BIAS at class level routes all A* calls through the radio-aware planner.
"""

from .marl_env_lozano import LunarRoverMeshLozanoEnv


class LunarRoverMeshLozanoConnNavEnv(LunarRoverMeshLozanoEnv):
    """
    LunarRoverMeshLozanoEnv with radio_bias > 0 so A* incorporates BS signal quality.

    radio_bias=0.5 is a reasonable starting point; tune up for stronger connectivity
    preference (at the cost of longer paths) or down toward 0 to approach terrain-only A*.
    """

    RADIO_BIAS: float = 0.5

    def __init__(self, *args, **kwargs):
        # Inject radio_bias unless caller provides it explicitly
        kwargs.setdefault("radio_bias", self.RADIO_BIAS)
        super().__init__(*args, **kwargs)