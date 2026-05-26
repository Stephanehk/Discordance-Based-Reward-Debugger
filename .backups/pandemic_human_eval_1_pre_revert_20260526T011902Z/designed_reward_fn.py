"""Designed reward for pandemic_human_eval_1.

Observation is a numpy array of shape (history, 1, 13); the latest row
``next_obs[-1, 0]`` holds ``[time_day, stage, (unused legacy field),
alpha(critical, dead, infected, none, recovered),
delta(critical, dead, infected, none, recovered)]``, where every category
field is a proportion of the population (each strain's 5 fields sum to 1).
Critical proportion = ``state[3] + state[8]`` (alpha + delta).
"""

import numpy as np

# Existing term (carried over from the starting point).
W_DEATH = 100.0          # weight on cumulative death proportion (alpha + delta)

# New terms added per user request.
W_LOW_CRITICAL_STAGE = 5.0    # penalty when stage > 0 while critical proportion < 0.01
W_MAX_STAGE = 10.0            # penalty per step spent at stage 4
W_STAGE_CHANGE = 1.0          # small penalty when the stage changes (either direction)
W_CRIT_LEVEL = 1_000_000.0    # largest penalty: quadratic in the current critical proportion (per step)

LOW_CRITICAL_THRESHOLD = 0.01 # threshold defining "low" critical proportion (dimensionless)


def designed_reward_fn(obs, act, next_obs) -> float:
    prev = np.asarray(obs)[-1, 0]             # state before the action (length 13)
    cur = np.asarray(next_obs)[-1, 0]         # state after the action (length 13)

    prev_stage = float(prev[1])               # lockdown stage in {0,1,2,3,4} before the action
    cur_stage = float(cur[1])                 # lockdown stage in {0,1,2,3,4} after the action
    prev_critical = float(prev[3]) + float(prev[8])   # critical proportion before (dimensionless)
    cur_critical = float(cur[3]) + float(cur[8])      # critical proportion after (dimensionless)
    dead = float(cur[4]) + float(cur[9])              # cumulative death proportion (dimensionless)

    # Existing death penalty.
    r = -W_DEATH * dead

    # Penalty for nonzero stage when critical proportion is below the low threshold.
    if cur_critical < LOW_CRITICAL_THRESHOLD and cur_stage > 0:
        r -= W_LOW_CRITICAL_STAGE

    # Penalty per step at the strictest stage.
    if cur_stage == 4:
        r -= W_MAX_STAGE

    # Small penalty for any change in stage, in either direction.
    if cur_stage != prev_stage:
        r -= W_STAGE_CHANGE

    # Largest penalty: quadratic in the current critical proportion (level-based).
    r -= W_CRIT_LEVEL * (cur_critical ** 2)

    return r
