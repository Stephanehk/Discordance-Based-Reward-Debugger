"""Environment configuration for the highway-env `two-way-v0` task.

The ego drives on a two-lane road shared with oncoming traffic and slower
same-direction lead vehicles, and may overtake by entering the oncoming lane.
The episode ends on collision (and at the 15-step horizon).

Crash signal
------------
highway-env's authoritative collision flag is `info["crashed"]`, which the
reward and trajectory analysis never receive (they only see
`(obs, act, next_obs)`). To make it observable, `make_env` wraps the env so
that `info["crashed"]` is routed into the ego presence cell `obs[0, 0]`
(normally always 1.0): after a crashing step, `next_obs[0, 0] == 0.0`.
"""

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  (registers env ids)

CAPTURE_FRAMES = True

_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
    },
}

OBS_DESCRIPTION = """\
Box(shape=(5, 5), dtype=float32). A (V, F) matrix of the V=5 nearest vehicles
(row 0 is the ego), each with F=5 features in absolute world units:
  col 0: presence - 1.0 if the row is a real vehicle, 0.0 if padding. On the
                    ego row (0) it is also set to 0.0 on a step that ends in a
                    collision (otherwise 1.0).
  col 1: x        - longitudinal position (meters).
  col 2: y        - lateral position (meters); lane width is ~4 m.
  col 3: vx       - longitudinal velocity (m/s).
  col 4: vy       - lateral velocity (m/s).
Rows 1..4 are the other vehicles (including oncoming traffic, which travels in
the opposite direction).
"""

ACTION_DESCRIPTION = """\
Discrete(5) - DiscreteMetaAction high-level manoeuvres (1 step = 1 s):
  0: LANE_LEFT  - change one lane to the left (toward the oncoming lane).
  1: IDLE       - keep lane and speed.
  2: LANE_RIGHT - change one lane to the right.
  3: FASTER     - increase target speed one increment (speeds: 20/25/30 m/s).
  4: SLOWER     - decrease target speed one increment.
"""

MAX_EPISODE_STEPS = 15


class _CrashFlagWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("crashed", False):
            obs = np.asarray(obs).copy()
            obs[0, 0] = 0.0
        return obs, reward, terminated, truncated, info


def make_env(render_mode: str | None = None):
    kwargs = {"config": _CONFIG}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return _CrashFlagWrapper(gym.make("two-way-v0", **kwargs))
