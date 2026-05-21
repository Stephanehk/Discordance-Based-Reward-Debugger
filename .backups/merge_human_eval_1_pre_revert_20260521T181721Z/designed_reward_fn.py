"""Designed reward for merge_human_eval_1.

Starting point: crashes are heavily penalized and otherwise the only
objective was to maximize the ego vehicle's forward speed.

Revision (user-directed): the raw forward-speed bonus was removed and
replaced by two penalties (no other positive term remains besides the
absence of penalty) —
  1. a penalty when ego speed deviates from a 25 m/s target, and
  2. a penalty when the nearest vehicle is closer than 10 m.
The speed denormalization was also corrected from 30 to the
env-documented half-width of 80 m/s (env_config.py flags 30 as wrong),
so the speed reads true m/s.
"""

import numpy as np


# Crash signal: env_config.py routes info["crashed"] into obs[0, 0].
# obs[0, 0] is 1.0 on a healthy step and 0.0 on a crash step.
_CRASH_PENALTY = -100.0  # dimensionless reward units; paid on the crash step

# highway-env Kinematics features_range for merge-v0 normalises vx to
# [-1, 1] over [-80, 80] m/s. Multiply normalised vx by this half-width
# to recover ego speed in m/s (env_config.py verifies 80, not 30).
_VX_HALFWIDTH_MPS = 80.0  # m/s — Kinematics vx half-range
_X_HALFWIDTH_M = 200.0    # m — neighbour longitudinal position half-range
_Y_HALFWIDTH_M = 8.0      # m — neighbour lateral position half-range

# Target-speed penalty: cost per m/s that ego speed differs from target.
_TARGET_SPEED_MPS = 25.0   # m/s — desired cruising speed
_SPEED_DEV_WEIGHT = 2.0    # reward units per m/s of deviation from target

# Proximity penalty: cost per metre the nearest neighbour is inside the
# minimum desired gap. No penalty once the gap is >= the threshold.
_MIN_SAFE_GAP_M = 10.0     # m — minimum desired distance to nearest vehicle
_GAP_VIOLATION_WEIGHT = 5.0  # reward units per m inside the minimum gap
_FAR_GAP_M = 200.0         # m — gap reported when no neighbour is present


def _nearest_gap_m(obs):
    # Distance (m) to the nearest present neighbour (rows 1..4, relative
    # coords); _FAR_GAP_M when no neighbour is observed.
    obs = np.asarray(obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:  # present (real, non-padding) neighbour
            rx = obs[i, 1] * _X_HALFWIDTH_M  # m, relative longitudinal
            ry = obs[i, 2] * _Y_HALFWIDTH_M  # m, relative lateral
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else _FAR_GAP_M


def designed_reward_fn(obs, act, next_obs) -> float:
    next_obs = np.asarray(next_obs)

    # Heavy penalty whenever the env's crash detector fired this step.
    if next_obs[0, 0] < 0.5:
        return _CRASH_PENALTY

    # Ego forward speed in m/s.
    ego_speed_mps = float(next_obs[0, 3]) * _VX_HALFWIDTH_MPS  # m/s

    # Penalty for deviating from the 25 m/s target (m/s -> reward units).
    speed_dev_penalty = _SPEED_DEV_WEIGHT * abs(ego_speed_mps - _TARGET_SPEED_MPS)

    # Penalty for getting closer than 10 m to the nearest vehicle.
    gap_m = _nearest_gap_m(next_obs)  # m
    gap_penalty = _GAP_VIOLATION_WEIGHT * max(0.0, _MIN_SAFE_GAP_M - gap_m)

    return -speed_dev_penalty - gap_penalty
