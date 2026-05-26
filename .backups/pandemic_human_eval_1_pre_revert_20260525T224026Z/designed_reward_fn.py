"""Designed reward for pandemic_human_eval_1.

Observation is a numpy array of shape (history, 1, 13); the latest row
next_obs[-1, 0] holds [time_day, stage, (unused legacy field),
alpha(critical, dead, infected, none, recovered),
delta(critical, dead, infected, none, recovered)], where every category
field is a proportion of the population (each strain's 5 fields sum to 1).
"""

import numpy as np

W_DEATH = 100.0            # penalty per unit cumulative death proportion (alpha + delta)
W_OVERREGULATE = 10.0      # flat per-step penalty when locking down (stage > 0) while critical < CRITICAL_LOW
W_STAGE4 = 1.0             # flat per-step penalty for being in the strictest lockdown stage (stage == 4)
W_ILL = 80.0               # per-step penalty on (currently infected + critical) proportion
W_CHURN = 1.0              # per-step penalty whenever the regulation stage changes from the previous step

CRITICAL_LOW = 0.01        # critical-proportion threshold below which lockdown should be 0


def designed_reward_fn(obs, act, next_obs) -> float:
    prev = np.asarray(obs)[-1, 0]                  # length-13 state before the action
    cur = np.asarray(next_obs)[-1, 0]              # length-13 state after the action
    stage = float(cur[1])                          # regulation stage in force, 0..4
    prev_stage = float(prev[1])                    # regulation stage in force one step earlier
    critical = float(cur[3]) + float(cur[8])       # current critical proportion, alpha + delta
    infected = float(cur[5]) + float(cur[10])      # currently-infected (non-critical) proportion, alpha + delta
    dead = float(cur[4]) + float(cur[9])           # cumulative death proportion, alpha + delta

    reward = -W_DEATH * dead                       # avoid deaths

    # Per-step penalty on active illness load (infected + critical).
    reward -= W_ILL * (infected + critical)

    # Large penalty for any non-zero lockdown when critical cases are very low.
    if critical < CRITICAL_LOW and stage > 0:
        reward -= W_OVERREGULATE

    # Penalize being in the strictest lockdown stage.
    if stage >= 4:
        reward -= W_STAGE4

    # Small penalty whenever the regulation stage changes (either direction).
    if stage != prev_stage:
        reward -= W_CHURN

    return reward
