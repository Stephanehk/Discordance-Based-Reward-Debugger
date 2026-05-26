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

W_DEATH = 100.0   # penalty per unit death proportion (alpha + delta)


def designed_reward_fn(obs, act, next_obs) -> float:
    cur = np.asarray(next_obs)[-1, 0]         # length-13 state after the action
    dead = float(cur[4]) + float(cur[9])      # cumulative death proportion, alpha + delta
    return -W_DEATH * dead                    # sole objective: avoid deaths
