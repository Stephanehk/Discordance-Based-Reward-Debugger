"""Designed reward for pandemic_human_eval_1 — STARTING POINT.

This is the *initial* reward function the human has been given to revise.
It is intentionally misaligned with the reward criteria documented in
``eval_spec.md``. Its sole objective is to penalize deaths: it subtracts a
penalty proportional to the cumulative death proportion (alpha + delta) and
does nothing else. It says nothing about the regulation stage — so it does
not tie the stage to the critical-case proportion, does not single out any
stage, and has no notion of how fast the stage changes from day to day.
Those gaps are the flaws the human is supposed to surface (with the help of
the /repair-* skills) and repair.

Observation is a numpy array of shape (history, 1, 13); the latest row
next_obs[-1, 0] holds [time_day, stage, (unused legacy field),
alpha(critical, dead, infected, none, recovered),
delta(critical, dead, infected, none, recovered)], where every category
field is a proportion of the population (each strain's 5 fields sum to 1).
"""

import numpy as np

W_DEATH = 100.0     # penalty per unit death proportion (alpha + delta), dimensionless
W_STAGE4 = 10.0     # large per-step penalty for sitting at the strictest lockdown (stage 4)
W_STAGE0 = 1.0      # per-step bonus for being at no-restrictions (stage 0)
W_CRIT = 1000.0     # per-step penalty per unit combined critical-case proportion (alpha + delta)
W_STAGE2 = 10.0     # per-step penalty for any day spent at stage 2


def designed_reward_fn(obs, act, next_obs) -> float:
    cur = np.asarray(next_obs)[-1, 0]         # length-13 state after the action
    stage = float(cur[1])                     # regulation stage in 0..4
    dead = float(cur[4]) + float(cur[9])      # cumulative death proportion, alpha + delta
    critical = float(cur[3]) + float(cur[8])  # combined critical-case proportion, alpha + delta

    reward = -W_DEATH * dead                  # objective: avoid deaths
    reward -= W_CRIT * critical               # penalize critical-case load (hospital strain)
    if stage == 4:
        reward -= W_STAGE4                    # discourage time at the strictest lockdown
    if stage == 0:
        reward += W_STAGE0                    # reward time at no restrictions
    if stage == 2:
        reward -= W_STAGE2                    # discourage any time at stage 2
    return reward
