"""Designed reward for the Pandemic Simulator (tiny-town COVID control).

Filled reward-design template (verbatim from the user):

  1. What outcomes should the agent try to achieve?
     - Keep critical cases below hospital capacity.
     - Keep lockdown regulations as low as possible.

  2. What outcomes should the agent avoid?
     - High death rates.
     - High lockdown levels.
     - Fluctuating lockdown levels.

  3. How should the agent trade off between objectives?
     - Preventing deaths outweighs economic cost, but once the outbreak is
       contained, restrictions should be relaxed promptly.

Observation is a numpy array of shape (history, 1, 13); the latest row
next_obs[-1, 0] holds [time_day, stage, (unused legacy field),
alpha(critical, dead, infected, none, recovered),
delta(critical, dead, infected, none, recovered)], where every category
field is a proportion of the population (each strain's 5 fields sum to 1).
"""

import numpy as np

# --- Weights (all dimensionless; chosen so deaths dominate every other term) ---
W_DEATH = 100.0      # per unit death fraction (alpha+delta), summed over strains
W_CRITICAL = 30.0    # per unit critical-case fraction (alpha+delta)
W_STAGE = 1.0        # linear economic/social cost of lockdown, per normalised stage
W_HIGH_STAGE = 3.0   # extra convex cost (quadratic in stage) so high lockdown levels hurt disproportionately
W_SMOOTH = 0.5       # penalty for changing the lockdown level day to day


def designed_reward_fn(obs, act, next_obs) -> float:
    prev = np.asarray(obs)[-1, 0]        # length-13 state before the action
    cur = np.asarray(next_obs)[-1, 0]    # length-13 state after the action

    # Indices into the 13-vector (see module docstring / OBS_DESCRIPTION).
    prev_stage = float(prev[1])          # regulation stage in force before, 0..4
    cur_stage = float(cur[1])            # regulation stage in force after, 0..4

    dead = float(cur[4]) + float(cur[9])         # death fraction, alpha + delta (0..2)
    critical = float(cur[3]) + float(cur[8])     # critical fraction, alpha + delta (0..2)

    # --- Avoid deaths (dominant term) ---
    death_penalty = -W_DEATH * dead

    # --- Keep critical load low (proxy for staying within hospital capacity) ---
    critical_penalty = -W_CRITICAL * critical                 # discourage critical load

    # --- Keep lockdown as low as possible (economic cost; relaxes when safe) ---
    stage_penalty = -W_STAGE * (cur_stage / 4.0)              # linear cost, normalised stage in [0,1]
    # Convex surcharge: penalizes high lockdown levels (esp. full lockdown)
    # disproportionately, so keeping the stage maxed out is discouraged.
    high_stage_penalty = -W_HIGH_STAGE * (cur_stage / 4.0) ** 2

    # --- Avoid fluctuating lockdown levels ---
    smooth_penalty = -W_SMOOTH * (abs(cur_stage - prev_stage) / 4.0)  # normalised |Δstage|

    return (
        death_penalty
        + critical_penalty
        + stage_penalty
        + high_stage_penalty
        + smooth_penalty
    )
