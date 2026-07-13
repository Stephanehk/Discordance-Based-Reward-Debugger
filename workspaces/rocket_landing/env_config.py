"""Environment configuration for the PyFlyt `Rocket-Landing-v4` task.

A rocket booster is dropped from ~450 m at terminal velocity with only 5% of its
fuel remaining, and must fly itself down to a soft, upright touchdown on a fixed
landing pad at the origin. The rocket is controlled through its grid finlets,
main-engine ignition/throttle, and thrust-vector (gimbal) angles.

Observation
-----------
The env already returns a flat 30-vector (default `angle_representation=
"quaternion"`), so no wrapping is needed. See OBS_DESCRIPTION for the exact
per-dimension layout. The reward and any trajectory analysis only ever receive
`(obs, act, next_obs)`; every physical quantity the designed reward needs (pose,
velocities, pad-contact flag) is already present in the observation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Registers the `PyFlyt/Rocket-Landing-v4` id on import.
import PyFlyt.gym_envs  # noqa: F401

CAPTURE_FRAMES = True

_ENV_ID = "PyFlyt/Rocket-Landing-v4"

OBS_DESCRIPTION = """\
Box(shape=(30,), dtype=float64). Flat rocket state (angle_representation=
"quaternion"):
  [0:3]   angular velocity (rad/s)
  [3:7]   orientation quaternion (x, y, z, w), PyBullet convention
  [7:10]  linear velocity (m/s), body frame
  [10:13] linear position (m), world frame; [12] is height above the pad,
          [10:12] is the horizontal offset from the pad (pad sits at the origin)
  [13:20] previous action buffer (7,), same layout as the action space
  [20:24] four grid-finlet actuation states, each in [-1, 1]
  [24]    main-booster ignition state (0/1)
  [25]    remaining fuel ratio (starts at 0.05, i.e. 5% fuel)
  [26]    booster throttle state
  [27:29] booster gimbal state (x, y)
  [29]    landing-pad contact indicator (0.0 / 1.0)
"""

ACTION_DESCRIPTION = """\
Box(shape=(7,), dtype=float64), low=[-1,-1,-1, 0, 0,-1,-1], high=[1,1,1,1,1,1,1]:
  [0] finlet x setpoint (pitch grid-fin), [-1, 1]
  [1] finlet y setpoint (yaw grid-fin), [-1, 1]
  [2] finlet roll setpoint, [-1, 1]
  [3] main-booster ignition (>0.5 ignites the engine), [0, 1]
  [4] throttle, [0, 1]
  [5] booster gimbal x (thrust vectoring), [-1, 1]
  [6] booster gimbal y (thrust vectoring), [-1, 1]
"""

# agent_hz (40) * max_duration_seconds (30) = 1200-step truncation horizon.
# Episodes usually end far earlier (crash / touchdown / out-of-bounds).
MAX_EPISODE_STEPS = 1200


class _RGBRenderWrapper(gym.Wrapper):
    """PyFlyt's PyBullet camera returns RGBA frames; drop the alpha channel so
    the rollout's mp4 writer receives standard 3-channel RGB."""

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame


def make_env(render_mode: str | None = None):
    kwargs = {"disable_env_checker": True}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(_ENV_ID, **kwargs)
    if render_mode is not None:
        env = _RGBRenderWrapper(env)
    return env
