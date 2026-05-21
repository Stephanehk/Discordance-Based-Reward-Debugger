"""Environment configuration for the highway-env `merge-v0` task.

The ego vehicle drives on a highway and must merge from an on-ramp into
mainline traffic populated by IDM-controlled vehicles. The episode ends
on collision or when the ego reaches the end of the road segment.

Crash detection
---------------
The canonical crash signal in highway-env is ``info["crashed"]`` returned
by ``env.step``. **This is the signal that crashes should be detected by**
— the env's own collision detector knows about real vehicle geometry, so
no distance proxy on the normalised Kinematics observation can match it
reliably. (Empirically, real collisions report a normalised minimum
inter-vehicle distance of ~0.025, not 0.0, so any distance-threshold proxy
that triggers below ~0.02 will silently miss every collision.)

The reward function and trajectory features only see ``(obs, act,
next_obs)`` — they never receive ``info``. To plumb the crash signal
through without breaking that invariant, ``make_env`` wraps the env in
``_CrashFlagWrapper`` which routes ``info["crashed"]`` into ``obs[0, 0]``
(the ego presence cell, which is always 1.0 in the unwrapped env). After
wrapping, downstream consumers can read ``next_obs[0, 0] < 0.5`` to
detect that the most recent step ended in a crash.
"""

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  (registers the env ids)


OBS_DESCRIPTION = """\
Box(low=-inf, high=inf, shape=(5, 5), dtype=float32).

Kinematics observation: a (V, F) matrix listing the V=5 closest vehicles
(row 0 is always the ego vehicle), each described by F=5 features:
  col 0: presence  - 1.0 if this row is a real vehicle, 0.0 if padding.
                     ALSO overloaded on the ego row (row 0): set to 0.0
                     when info["crashed"] is True for this step, so that
                     consumers without access to `info` can detect crashes
                     by reading next_obs[0, 0] < 0.5.
  col 1: x         - longitudinal position, normalised by the road extent.
  col 2: y         - lateral position, normalised by the road width.
  col 3: vx        - longitudinal velocity, normalised by reward speed range.
  col 4: vy        - lateral velocity, normalised by reward speed range.

Positions/velocities of rows 1..4 are *relative* to the ego (defaults of
highway-env Kinematics observation). The ego row (0) is in absolute
normalised coordinates.
"""

ACTION_DESCRIPTION = """\
Discrete(5) - DiscreteMetaAction. Each action is a high-level manoeuvre
that the underlying lateral/longitudinal controllers execute over one
policy step (1 second at the default 1 Hz policy frequency):
  0: LANE_LEFT  - request a lane change to the left.
  1: IDLE       - keep current lane and target speed.
  2: LANE_RIGHT - request a lane change to the right.
  3: FASTER     - increase target speed by one increment.
  4: SLOWER     - decrease target speed by one increment.
"""

# `merge-v0` does not declare a max_episode_steps; episodes terminate
# naturally on crash or end-of-road. Cap at 100 steps as a safety net.
MAX_EPISODE_STEPS = 100


class _CrashFlagWrapper(gym.Wrapper):
    """Route ``info["crashed"]`` into ``obs[0, 0]`` so consumers that only
    see ``(obs, act, next_obs)`` can detect collisions reliably. See the
    module docstring for the rationale.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("crashed", False):
            obs = np.asarray(obs).copy()
            obs[0, 0] = 0.0
        return obs, reward, terminated, truncated, info


def make_env():
    return _CrashFlagWrapper(gym.make("merge-v0"))
