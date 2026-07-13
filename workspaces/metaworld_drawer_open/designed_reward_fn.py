"""Designed reward for the Meta-World `drawer-open-v3` task.

This is the environment's own dense reward (`SawyerDrawerOpenEnvV3.compute_reward`
with the default `reward_function_version="v2"`), reproduced here so it can be
evaluated from a `(obs, act, next_obs)` transition alone. It is the sum of a
*caging* term (bring the gripper to the handle) and an *opening* term (bring the
handle to the fully-open target), each shaped with dm_control's `tolerance`,
scaled by 5. It is computed on `next_obs`, matching the env, which evaluates the
post-step observation. The action is unused.

Observation — Box(shape=(45,), float64):
  [0:3]   hand (end-effector) position (x, y, z), meters
  [3]     gripper aperture (open > 0)
  [4:7]   drawer handle position (x, y, z), meters
  [7:11]  drawer handle orientation quaternion (w, x, y, z)
  [11:14] second tracked object position (unused by this task), meters
  [14:18] second tracked object orientation quaternion
  [18:36] previous timestep's copy of indices [0:18]
  [36:39] task goal position (x, y, z), meters
  [39:42] target handle position when the drawer is fully open (x, y, z), meters
  [42:45] hand position at the start of the episode (x, y, z), meters

Action — Box(low=-1, high=1, shape=(4,), float32):
  [0] delta x of the end-effector
  [1] delta y of the end-effector
  [2] delta z of the end-effector
  [3] gripper command (> 0 closes the fingers, < 0 opens them)
"""

import numpy as np

MAX_DIST = 0.2  # meters the handle travels from closed to fully open
VALUE_AT_MARGIN = 0.1  # tolerance() output once the error reaches `margin`
XY_SCALE = np.array([3.0, 3.0, 1.0])  # emphasize XY hand-to-handle error over Z
REWARD_SCALE = 5.0  # env multiplies the summed terms by this


def _long_tail(x: float) -> float:
    """dm_control 'long_tail' sigmoid: 1 at x=0, VALUE_AT_MARGIN at x=1."""
    scale = np.sqrt(1.0 / VALUE_AT_MARGIN - 1.0)
    return float(1.0 / ((x * scale) ** 2 + 1.0))


def _tolerance(x: float, upper_bound: float, margin: float) -> float:
    """1 while `x` <= `upper_bound`, decaying to ~0 as `x` grows past it (x >= 0)."""
    if x <= upper_bound:
        return 1.0
    return _long_tail((x - upper_bound) / margin)


def designed_reward_fn(obs, act, next_obs) -> float:
    next_obs = np.asarray(next_obs, dtype=np.float64)

    gripper = next_obs[0:3]  # hand position, meters
    handle = next_obs[4:7]  # drawer-handle position, meters
    target_pos = next_obs[39:42]  # handle position when fully open, meters
    init_tcp = next_obs[42:45]  # hand position at episode start, meters

    # Opening: how far the handle still is from the fully-open pose, meters.
    handle_error = float(np.linalg.norm(handle - target_pos))
    reward_for_opening = _tolerance(handle_error, upper_bound=0.02, margin=MAX_DIST)

    # Caging: XY-emphasized hand-to-handle error, normalized by the same error at
    # episode start (hand at its initial pose, drawer closed).
    handle_pos_init = target_pos + np.array([0.0, MAX_DIST, 0.0])
    gripper_error = float(np.linalg.norm((handle - gripper) * XY_SCALE))
    gripper_error_init = float(np.linalg.norm((handle_pos_init - init_tcp) * XY_SCALE))
    reward_for_caging = _tolerance(
        gripper_error, upper_bound=0.01, margin=gripper_error_init
    )

    return REWARD_SCALE * (reward_for_caging + reward_for_opening)
