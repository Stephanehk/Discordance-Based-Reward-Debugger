"""Designed reward for PyFlyt/Rocket-Landing-v4.

A rocket booster is dropped from ~450 m at terminal velocity with only 5% of its
fuel remaining, and must fly itself down to a soft, upright touchdown on a fixed
landing pad at the origin, using its grid finlets, main-engine throttle, and
thrust-vector (gimbal) angles.

The reward and any trajectory analysis only ever receive `(obs, act, next_obs)`.
Every physical quantity you need is present in the observation vector.

Observation (30,), float64:
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

Action (7,), float64, low=[-1,-1,-1, 0, 0,-1,-1], high=[1,1,1,1,1,1,1]:
  [0] finlet x setpoint (pitch grid-fin), [-1, 1]
  [1] finlet y setpoint (yaw grid-fin), [-1, 1]
  [2] finlet roll setpoint, [-1, 1]
  [3] main-booster ignition (>0.5 ignites the engine), [0, 1]
  [4] throttle, [0, 1]
  [5] booster gimbal x (thrust vectoring), [-1, 1]
  [6] booster gimbal y (thrust vectoring), [-1, 1]
"""


"""Designed reward for PyFlyt `Rocket-Landing-v4` (the env's default dense reward).

The agent flies a falling rocket booster down to a soft, upright touchdown on a
landing pad at the origin. This reproduces PyFlyt's built-in dense reward
(`sparse_reward=False`) from a single `(obs, act, next_obs)` transition. See
env_config.OBS_DESCRIPTION for the observation layout.

Shaping terms (computed from the obs -> next_obs transition):
  - a constant -0.3 offset discouraging loitering in the air,
  - +0.3 / (horizontal_distance_to_pad + 0.1): staying over the pad,
  - +10 * (decrease in horizontal distance to the pad): lateral progress,
  - +0.2 * (decrease in height): descending,
  - +4 * deceleration_progress: slowing the vertical descent, weighted heavily
    near the ground (1/exp(height)) and flipped to a penalty once ascending,
  - -|yaw rate|: minimize spin,
  - -||roll, pitch||: minimize tilt.
Plus terminal contact bonuses:
  - on pad contact: +5 - 0.3*|vertical speed| (softer touchdown keeps more),
  - clean upright, near-zero-velocity touchdown: an extra +3.

Note: the env accumulates this shaping over its 3 physics substeps per agent
step, advancing the "previous" state each substep. We only observe the agent-step
endpoints (obs, next_obs), so this evaluates the composite once over the whole
transition. The distance/progress terms telescope exactly; the instantaneous
terms (offset, stay-over-pad, deceleration, spin, tilt) are the single-transition
values rather than a 3-substep sum.
"""

import numpy as np


def _quat_to_rotmat(q):
    """Body->world rotation matrix from a PyBullet (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _roll_pitch(q):
    """Roll and pitch (rad) from a PyBullet (x, y, z, w) quaternion.

    Matches p.getEulerFromQuaternion's roll/pitch to numerical precision.
    """
    x, y, z, w = q
    sarg = np.clip(-2.0 * (x * z - w * y), -1.0, 1.0)
    pitch = np.arcsin(sarg)
    roll = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    return roll, pitch


def _ground_vertical_speed(state):
    """World-frame vertical velocity: (R @ body_lin_vel)[2]."""
    quat = state[3:7]
    lin_vel = state[7:10]
    rot = _quat_to_rotmat(quat)
    return float(rot[2, :] @ lin_vel)


def designed_reward_fn(obs, act, next_obs):
    prev = np.asarray(obs, dtype=np.float64)
    curr = np.asarray(next_obs, dtype=np.float64)

    prev_lin_pos = prev[10:13]
    lin_pos = curr[10:13]

    lateral_progress = float(np.linalg.norm(prev_lin_pos[:2]) - np.linalg.norm(lin_pos[:2]))
    vertical_progress = float(prev_lin_pos[2] - lin_pos[2])
    lateral_distance = float(np.linalg.norm(lin_pos[:2])) + 0.1

    gv = _ground_vertical_speed(curr)
    prev_gv = _ground_vertical_speed(prev)
    deceleration_progress = (
        (gv - prev_gv + 1.0)
        / np.exp(lin_pos[2])
        * (1.0 if gv < 0.0 else -1.0)
    )

    ang_vel = curr[0:3]
    roll, pitch = _roll_pitch(curr[3:7])
    tilt = float(np.hypot(roll, pitch))

    reward = (
        -0.3
        + (0.3 / lateral_distance)
        + (10.0 * lateral_progress)
        + (0.2 * vertical_progress)
        + (4.0 * float(deceleration_progress))
        - (1.0 * abs(float(ang_vel[-1])))
        - (1.0 * tilt)
    )

    # Terminal contact bonuses. The obs carries a pad-contact indicator (idx 29).
    if curr[29] >= 0.5:
        reward += 5.0 - (0.3 * abs(gv))
        prev_ang_vel = prev[0:3]
        prev_lin_vel = prev[7:10]
        fatal = (
            np.linalg.norm(prev_ang_vel) > 0.35
            or np.linalg.norm(prev_lin_vel) > 1.0
        )
        landed = (
            np.linalg.norm(prev_ang_vel) < 0.02
            and np.linalg.norm(prev_lin_vel) < 0.02
            and tilt < 0.1
        )
        if not fatal and landed:
            reward += 3.0

    return float(reward)
