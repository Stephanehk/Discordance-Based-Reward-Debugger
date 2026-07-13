"""Designed reward for the Meta-World `drawer-open-v3` task.

A Sawyer arm must pull a drawer open until its handle reaches the target
fully-open position. Each transition is (obs, act, next_obs); return one float.

Observation (Box shape=(45,), float64) — world-frame, meters unless noted:
  [0:3]   hand (end-effector) position (x, y, z)
  [3]     gripper aperture (open > 0)
  [4:7]   drawer handle position (x, y, z)
  [7:11]  drawer handle orientation quaternion (w, x, y, z)
  [11:14] second tracked object position (unused by this task)
  [14:18] second tracked object orientation quaternion
  [18:36] previous timestep's copy of indices [0:18]
  [36:39] task goal position (x, y, z)
  [39:42] target handle position when the drawer is fully open (x, y, z)
  [42:45] hand position at the start of the episode (x, y, z)

Action (Box low=-1, high=1, shape=(4,), float32) — delta end-effector control:
  [0] delta x of the end-effector
  [1] delta y of the end-effector
  [2] delta z of the end-effector
  [3] gripper command (>0 closes the fingers, <0 opens them)

Fill in the body. Signature must be unchanged.
"""


"""Designed reward for drawer-open.

Revised to reflect the preferences labeled over the surfaced trajectory
comparisons. The original reward summed a caging term (gripper hugging the
handle) and an opening term (handle near its fully-open target), each on a
[0,1] scale, scaled by 5. Those labels showed the reward over-credited hugging
the handle and gave too little credit for the drawer *actually* ending open, so
the reward now:

  (a) rewards the handle reaching its fully-open target, with a *sharp* tolerance
      (margin 0.03 m instead of 0.2) so near-misses earn much less than a real
      full open;
  (b) adds a per-step bonus for every step the drawer is within ~2 cm of open,
      crediting time actually spent open; and
  (c) turns the caging term into a small *penalty*, so hugging the handle
      without opening the drawer is no longer valued.

It is a function of the resulting state `next_obs` (see
env_config.OBS_DESCRIPTION for the observation layout).
"""

import numpy as np

_MAX_DIST = 0.2  # nominal handle travel from closed to open

# Weights over the per-step terms (opening / time-open / caging).
_W_OPEN = 15.0     # handle at the fully-open target (sharp tolerance)
_W_STEP_OPEN = 1.0  # per-step bonus for the drawer being open
_W_CAGE = 4.0      # penalty on hugging the handle (subtracted)
_OPEN_MARGIN = 0.03  # sharp falloff: near-misses lose credit fast
_OPEN_THRESH = 0.02  # handle within this of target counts as "open"


def _tolerance(x, lower, upper, margin, value_at_margin=0.1):
    """1 inside [lower, upper], decaying toward value_at_margin at `margin`
    distance from the nearest bound (long-tail falloff)."""
    if lower <= x <= upper:
        return 1.0
    if margin <= 0:
        return 0.0
    d = (lower - x if x < lower else x - upper) / margin
    scale = (1.0 / value_at_margin - 1.0) ** 0.5
    return 1.0 / ((d * scale) ** 2 + 1.0)


def designed_reward_fn(obs, act, next_obs):
    s = np.asarray(next_obs, dtype=np.float64)

    gripper = s[0:3]
    handle = s[4:7]
    target = s[39:42]
    init_tcp = s[42:45]

    handle_error = float(np.linalg.norm(handle - target))
    reward_for_opening = _tolerance(handle_error, 0.0, _OPEN_THRESH, _OPEN_MARGIN)
    bonus_for_open = 1.0 if handle_error <= _OPEN_THRESH else 0.0

    handle_pos_init = target + np.array([0.0, _MAX_DIST, 0.0])
    scale = np.array([3.0, 3.0, 1.0])
    gripper_error = (handle - gripper) * scale
    gripper_error_init = (handle_pos_init - init_tcp) * scale
    penalty_for_caging = _tolerance(
        float(np.linalg.norm(gripper_error)), 0.0, 0.01, float(np.linalg.norm(gripper_error_init))
    )

    return float(
        _W_OPEN * reward_for_opening
        + _W_STEP_OPEN * bonus_for_open
        - _W_CAGE * penalty_for_caging
    )
