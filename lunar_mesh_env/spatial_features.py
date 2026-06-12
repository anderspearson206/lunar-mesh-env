# spatial_features.py
"""
Compact spatial observations for the AlphaStar-style policy.

Instead of shipping the raw (1, 256, 256) terrain and radio maps through
RLlib rollouts (512 KB/agent/step, ignored by the old MLP policy), this module
builds two small AlphaStar-inspired views:

    screen  (3, 64, 64): native-resolution local crop centered on the rover
        ch 0: terrain heightmap (edge-padded, z-score normalized)
        ch 1: own radio map    (signal from a TX at the rover's position)
        ch 2: BS radio map     (base-station coverage)

    minimap (7, 64, 64): global 256->64 view (4x4 average pooling)
        ch 0: terrain   ch 1: own radio   ch 2: BS radio
        ch 3: self pos  ch 4: goal pos    ch 5: other rovers   ch 6: BS pos
        (3-6 are scatter-connection-style entity planes: 3x3 blocks of 1.0)

`build_spatial_obs` is the single source of truth shared by the env wrapper
(rollouts) and the SL demonstration dataloader, guaranteeing feature parity
between supervised pretraining and RL fine-tuning.
"""

import functools

import numpy as np
from gymnasium import spaces

SCREEN_SIZE = 64
MINIMAP_SIZE = 64
SCREEN_CHANNELS = 3
MINIMAP_CHANNELS = 7
POOL = 256 // MINIMAP_SIZE          # 4x4 average pooling
RADIO_FLOOR_DBM = -200.0

# Keys passed through from the raw observation unchanged.
PASSTHROUGH_KEYS = ("position", "goal_vector", "move_history", "action_mask")
# Raw keys replaced by screen/minimap.
DROPPED_KEYS = ("terrain", "radio_map")


def pool4(map256: np.ndarray) -> np.ndarray:
    """(256, 256) -> (64, 64) via 4x4 average pooling."""
    m = np.squeeze(np.asarray(map256, dtype=np.float32))
    return m.reshape(MINIMAP_SIZE, POOL, MINIMAP_SIZE, POOL).mean(axis=(1, 3))


def normalize_radio(r_dbm: np.ndarray) -> np.ndarray:
    """dBm in [-200, 0] -> [0, 1]."""
    return (r_dbm - RADIO_FLOOR_DBM) / -RADIO_FLOOR_DBM


def crop64_edge(map256: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """64x64 crop centered on (cx, cy); out-of-bounds replicates edge values."""
    h = SCREEN_SIZE // 2
    ys = np.clip(np.arange(cy - h, cy + h), 0, map256.shape[0] - 1)
    xs = np.clip(np.arange(cx - h, cx + h), 0, map256.shape[1] - 1)
    return map256[np.ix_(ys, xs)]


def crop64_const(map256: np.ndarray, cx: int, cy: int, fill: float) -> np.ndarray:
    """64x64 crop centered on (cx, cy); out-of-bounds filled with `fill`."""
    h = SCREEN_SIZE // 2
    out = np.full((SCREEN_SIZE, SCREEN_SIZE), fill, dtype=np.float32)
    y0, x0 = cy - h, cx - h
    sy0, sy1 = max(y0, 0), min(cy + h, map256.shape[0])
    sx0, sx1 = max(x0, 0), min(cx + h, map256.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = map256[sy0:sy1, sx0:sx1]
    return out


def scatter_plane(points) -> np.ndarray:
    """(64, 64) plane with a 3x3 block of 1.0 at each pooled point location."""
    plane = np.zeros((MINIMAP_SIZE, MINIMAP_SIZE), dtype=np.float32)
    for (px, py) in points:
        mx = int(np.clip(px, 0, 255)) // POOL
        my = int(np.clip(py, 0, 255)) // POOL
        plane[max(my - 1, 0):my + 2, max(mx - 1, 0):mx + 2] = 1.0
    return plane


def build_spatial_obs(hm_norm: np.ndarray,
                      hm_pool_norm: np.ndarray,
                      own_radio_dbm: np.ndarray,
                      bs_radio_dbm: np.ndarray,
                      self_xy, goal_xy, bs_xy, other_xys,
                      bs_pool_norm: np.ndarray = None) -> dict:
    """Build {"screen": (3,64,64), "minimap": (7,64,64)} float32 arrays.

    hm_norm:       (256, 256) z-score-normalized heightmap
    hm_pool_norm:  (64, 64) pooled version of hm_norm
    own_radio_dbm: (256, 256) dBm map for a TX at the rover's position
    bs_radio_dbm:  (256, 256) dBm map for the base station
    bs_pool_norm:  optional precomputed pooled+normalized BS map (static per episode)
    """
    own_radio_dbm = np.squeeze(np.asarray(own_radio_dbm, dtype=np.float32))
    bs_radio_dbm = np.squeeze(np.asarray(bs_radio_dbm, dtype=np.float32))
    cx, cy = int(round(float(self_xy[0]))), int(round(float(self_xy[1])))

    screen = np.stack([
        crop64_edge(hm_norm, cx, cy),
        normalize_radio(crop64_const(own_radio_dbm, cx, cy, RADIO_FLOOR_DBM)),
        normalize_radio(crop64_const(bs_radio_dbm, cx, cy, RADIO_FLOOR_DBM)),
    ]).astype(np.float32)

    if bs_pool_norm is None:
        bs_pool_norm = normalize_radio(pool4(bs_radio_dbm))

    minimap = np.stack([
        hm_pool_norm,
        normalize_radio(pool4(own_radio_dbm)),
        bs_pool_norm,
        scatter_plane([self_xy]),
        scatter_plane([goal_xy]),
        scatter_plane(other_xys),
        scatter_plane([bs_xy]),
    ]).astype(np.float32)

    return {"screen": screen, "minimap": minimap}


class SpatialObsWrapper:
    """PettingZoo ParallelEnv wrapper replacing raw maps with screen/minimap.

    Apply to the raw LunarRoverMeshEnv BEFORE ParallelPettingZooEnv.
    """

    def __init__(self, raw_env):
        self.env = raw_env
        hm = raw_env.heightmap.astype(np.float32)
        std = float(hm.std())
        self._hm_mean = float(hm.mean())
        self._hm_std = std if std > 1e-6 else 1.0
        self._hm_norm = (hm - self._hm_mean) / self._hm_std
        self._hm_pool_norm = pool4(self._hm_norm)
        self._bs_pool_norm = None

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def unwrapped(self):
        return self.env

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        raw = self.env.observation_space(agent)
        d = {k: raw[k] for k in PASSTHROUGH_KEYS}
        d["screen"] = spaces.Box(low=-20.0, high=20.0,
                                 shape=(SCREEN_CHANNELS, SCREEN_SIZE, SCREEN_SIZE),
                                 dtype=np.float32)
        d["minimap"] = spaces.Box(low=-20.0, high=20.0,
                                  shape=(MINIMAP_CHANNELS, MINIMAP_SIZE, MINIMAP_SIZE),
                                  dtype=np.float32)
        return spaces.Dict(d)

    @property
    def observation_spaces(self):
        return {a: self.observation_space(a) for a in self.env.possible_agents}

    def action_space(self, agent):
        return self.env.action_space(agent)

    @property
    def action_spaces(self):
        return self.env.action_spaces

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._bs_pool_norm = normalize_radio(pool4(self.env.bs_radio_map))
        return {a: self._transform(a, o) for a, o in obs.items()}, infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return ({a: self._transform(a, o) for a, o in obs.items()},
                rewards, terms, truncs, infos)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate anything else (metadata, render_mode, agent_map, ...).
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Observation transform
    # ------------------------------------------------------------------

    def _transform(self, agent_id, obs):
        agent = self.env.agent_map[agent_id]
        self_xy = (agent.x, agent.y)
        goal_xy = (agent.goal_x, agent.goal_y)
        bs_xy = (self.env.base_station.x, self.env.base_station.y)
        other_xys = [(a.x, a.y) for aid, a in self.env.agent_map.items()
                     if aid != agent_id]

        spatial = build_spatial_obs(
            self._hm_norm, self._hm_pool_norm,
            own_radio_dbm=obs["radio_map"],
            bs_radio_dbm=self.env.bs_radio_map,
            self_xy=self_xy, goal_xy=goal_xy, bs_xy=bs_xy, other_xys=other_xys,
            bs_pool_norm=self._bs_pool_norm,
        )

        out = {k: obs[k] for k in PASSTHROUGH_KEYS}
        out.update(spatial)
        return out
