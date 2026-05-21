"""Designed reward for merge_human_eval_2 (revised).

Per non-crash transition the reward now sums:
  1. a speed term that is maximal when the ego's forward speed equals a
     target speed and falls off linearly as the speed moves away from it
     (in either direction), replacing the original "raw forward speed"
     objective; and
  2. a small per-step bonus when the ego is in the left mainline lane,
     acting as an all-else-equal tiebreaker.
A crash still incurs a large flat penalty.

Speed is read at the verified physical scale: the env's Kinematics
observation normalises longitudinal velocity over a +/-80 m/s range, so
true ego speed (m/s) = next_obs[0, 3] * 80.0. (The earlier version used
an internal 30.0 half-width, which under-reported true speed and made a
25 m/s target unreachable; 80.0 is the value verified against the live
simulator.)
"""


# Crash signal: env_config.py routes info["crashed"] into obs[0, 0].
# obs[0, 0] is 1.0 on a healthy step and 0.0 on a crash step.
_CRASH_PENALTY = -100.0  # dimensionless reward units; paid on the crash step

# Denormalisation (verified against the live simulator).
_VX_HALFWIDTH_MPS = 80.0  # m/s — true ego speed = normalised vx * 80.0
_LAT_HALFWIDTH_M = 8.0     # m — ego lateral position (m) = normalised y * 8.0

# Speed term: peaks at the target speed, falls off linearly with deviation.
_TARGET_SPEED_MPS = 25.0    # m/s — speed the reward aims for
_SPEED_BONUS_MAX = 10.0     # reward units at exactly the target speed
_SPEED_TOLERANCE_MPS = 10.0  # m/s — deviation at which the speed bonus reaches 0

# Left-lane tiebreaker: the left mainline lane sits at y ~= 0 m and the
# right lane at y ~= 4 m, so a lateral position below ~2 m means the ego
# is in the left lane. Small per-step bonus relative to the speed term.
_LANE_MIDLINE_M = 2.0       # m — boundary between left (y~0) and right (y~4) lanes
_LEFT_LANE_BONUS = 0.5      # reward units per step spent in the left lane


def designed_reward_fn(obs, act, next_obs) -> float:
    # Heavy penalty whenever the env's crash detector fired this step.
    if next_obs[0, 0] < 0.5:
        return _CRASH_PENALTY

    # True ego forward speed this step, in m/s.
    ego_speed_mps = float(next_obs[0, 3]) * _VX_HALFWIDTH_MPS

    # Speed term: 10 at the target, decreasing linearly to 0 once the speed
    # is _SPEED_TOLERANCE_MPS away from the target (never goes negative, so
    # every non-crash step still contributes non-negatively).
    deviation_mps = abs(ego_speed_mps - _TARGET_SPEED_MPS)  # m/s from target
    speed_term = _SPEED_BONUS_MAX * max(0.0, 1.0 - deviation_mps / _SPEED_TOLERANCE_MPS)

    # Left-lane tiebreaker: small bonus when the ego sits in the left lane.
    ego_lateral_m = float(next_obs[0, 2]) * _LAT_HALFWIDTH_M  # m
    lane_term = _LEFT_LANE_BONUS if ego_lateral_m < _LANE_MIDLINE_M else 0.0

    return speed_term + lane_term
