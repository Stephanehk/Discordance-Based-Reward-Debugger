"""Designed reward for merge_human_eval_2 — STARTING POINT.

This is the *initial* reward function the human has been given to revise.
It is intentionally misaligned with the reward criteria documented in
``eval_spec.md``: crashes are heavily penalized, and otherwise the only
objective is to maximize the ego vehicle's forward speed. It does NOT
encode the distance-keeping requirement or the ~25 m/s speed target
from ``eval_spec.md`` — those are flaws the human is supposed to
surface (with the help of the /repair-* skills) and repair.
"""


# Crash signal: env_config.py routes info["crashed"] into obs[0, 0].
# obs[0, 0] is 1.0 on a healthy step and 0.0 on a crash step.
_CRASH_PENALTY = -100.0  # dimensionless reward units; paid on the crash step

# highway-env's Kinematics observation normalises longitudinal velocity
# to [-1, 1] over a features_range that spans roughly [-30, 30] m/s
# for the merge env. Multiplying the normalised vx by this half-width
# recovers unnormalised ego speed in m/s.
_VX_HALFWIDTH_MPS = 30.0  # m/s — matches highway-env's default Kinematics vx range


def designed_reward_fn(obs, act, next_obs) -> float:
    # Heavy penalty whenever the env's crash detector fired this step.
    if next_obs[0, 0] < 0.5:
        return _CRASH_PENALTY

    # Otherwise: reward proportional to ego forward speed, in m/s.
    ego_vx_normalised = float(next_obs[0, 3])  # dimensionless, in [-1, 1]
    ego_speed_mps = ego_vx_normalised * _VX_HALFWIDTH_MPS  # m/s
    return ego_speed_mps  # sole objective: maximize forward speed
