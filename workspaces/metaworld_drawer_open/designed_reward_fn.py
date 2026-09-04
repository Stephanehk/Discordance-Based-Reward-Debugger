"""Designed reward for the Meta-World `drawer-open-v3` task.

What this reward pays for
-------------------------
Over an episode, the rewards below sum to

    return(tau)  ~=  OPEN_WEIGHT * (how far open the drawer is at the END)
                     + a small, bounded guidance term for reaching the handle

so a trajectory is worth exactly the state it leaves the drawer in. Opening the
drawer and letting it slide back gives that credit up; opening it and shoving it
shut again is worth nothing at all. Nothing is paid for *dwelling* near the goal,
so the reward cannot be collected by parking the arm half-way and idling.

How it gets there
-----------------
Both terms are written as **differences** between the step's two observations
rather than as levels. A per-step level is integrated over the episode, which is
what makes an idle-in-a-good-pose trajectory score like a finished one. A per-step
difference telescopes: summed over the episode it collapses to (final - initial),
i.e. a property of the endpoints alone.

  * Opening: `OPEN_WEIGHT * (openness(next_obs) - openness(obs))`, where openness
    runs 0 (shut, as the episode starts) to 1 (fully open). It sums to
    `OPEN_WEIGHT * openness(final)`. Progress pays, backsliding costs the same,
    and standing still pays zero.

  * Caging: the same hand-to-handle term the task has always used, but entered as
    a *potential* — only its change is paid. It sums to
    `CAGE_WEIGHT * (cage(final) - cage(initial))`, which is bounded by
    +/- CAGE_WEIGHT and cannot outweigh the opening term. It keeps a dense
    "get to the handle" gradient for learning without letting a trajectory earn
    anything by hovering there.

The action is unused: how hard or how jerkily the arm moves is not scored.

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

OPEN_WEIGHT = 10.0  # a full open, held to the end, is worth this much
CAGE_WEIGHT = 0.5  # guidance only; bounded by +/- this over a whole episode


def _long_tail(x: float) -> float:
    """dm_control 'long_tail' sigmoid: 1 at x=0, VALUE_AT_MARGIN at x=1."""
    scale = np.sqrt(1.0 / VALUE_AT_MARGIN - 1.0)
    return float(1.0 / ((x * scale) ** 2 + 1.0))


def _tolerance(x: float, upper_bound: float, margin: float) -> float:
    """1 while `x` <= `upper_bound`, decaying to ~0 as `x` grows past it (x >= 0)."""
    if x <= upper_bound:
        return 1.0
    return _long_tail((x - upper_bound) / margin)


def _openness(o) -> float:
    """How far open the drawer is: 0.0 = shut (as it starts), 1.0 = fully open."""
    handle = o[4:7]
    target_pos = o[39:42]  # handle position when fully open
    remaining = float(np.linalg.norm(handle - target_pos))
    return float(np.clip(1.0 - remaining / MAX_DIST, 0.0, 1.0))


def _caging_potential(o) -> float:
    """Hand-to-handle proximity, 0 (far) to 1 (at the handle). Guidance only."""
    gripper = o[0:3]  # hand position
    handle = o[4:7]  # drawer-handle position
    target_pos = o[39:42]
    init_tcp = o[42:45]  # hand position at episode start

    # Normalize by the same error at episode start: hand at its initial pose,
    # drawer shut (so the handle sits MAX_DIST from its fully-open target).
    handle_pos_init = target_pos + np.array([0.0, MAX_DIST, 0.0])
    gripper_error = float(np.linalg.norm((handle - gripper) * XY_SCALE))
    gripper_error_init = float(np.linalg.norm((handle_pos_init - init_tcp) * XY_SCALE))
    return _tolerance(gripper_error, upper_bound=0.01, margin=gripper_error_init)


def designed_reward_fn(obs, act, next_obs) -> float:
    obs = np.asarray(obs, dtype=np.float64)
    next_obs = np.asarray(next_obs, dtype=np.float64)

    # Both terms are paid on their *change* over the step, so that summing them
    # over an episode leaves only the endpoints (see the module docstring).
    opening = _openness(next_obs) - _openness(obs)
    caging = _caging_potential(next_obs) - _caging_potential(obs)

    return OPEN_WEIGHT * opening + CAGE_WEIGHT * caging
