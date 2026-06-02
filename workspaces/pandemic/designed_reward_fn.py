"""Designed reward for the pandemic-regulator agent.

Reward design template (verbatim from the user):

  Achieve:  low lockdown levels, low death/critical proportions
  Avoid:    lots of time in stage 4 or outbreaks resulting in deaths
  Tradeoff: tradeoff between time in high lockdown stage vs. critical
            proportions rising. you should almost never be in stage 4;
            that should come with a high penalty. but you should also
            never let an outbreak go uncontrolled.

Per-step reward built from the latest row of next_obs:
  + small bonus for keeping the regulation stage low (economy open)
  - large fixed penalty whenever stage == 4 (strictest lockdown)
  - penalty proportional to the share of the population currently in
    critical condition (summed across alpha and delta strains)
  - much larger penalty proportional to the cumulative share of the
    population that has died (summed across strains)

Scale rationale: in this town a few percent of the population in critical
condition is already a major outbreak, and the deceased share is small
but monotonically growing. The weights below are chosen so that
(a) a single percentage-point of critical cases outweighs a day's worth
of openness bonus, and (b) any deaths persistently dominate the per-step
reward, matching the user's "never let an outbreak go uncontrolled".
"""

import numpy as np


# Weights (all dimensionless; the reward itself is dimensionless per step).
OPENNESS_BONUS_MAX = 1.0      # bonus per day at stage 0; tapers to 0 at stage 4.
STAGE4_PENALTY     = 5.0      # extra per-day penalty for sitting in stage 4.
CRITICAL_WEIGHT    = 200.0    # multiplied by the critical-case population share.
DEATH_WEIGHT       = 2000.0   # multiplied by the cumulative deceased share.


def designed_reward_fn(obs, act, next_obs) -> float:
    """Return the per-step designed reward for one (obs, act, next_obs) transition."""
    # next_obs has shape (history, 1, 13); take the most recent observation row.
    state = np.asarray(next_obs)[-1, 0]

    stage = int(state[1])                      # regulation stage in force, integer 0..4

    # Population share currently in critical condition, summed across strains.
    # Dimensionless (fraction of the town's population).
    critical = float(state[3]) + float(state[8])

    # Cumulative population share that has died, summed across strains.
    # Dimensionless and monotonically non-decreasing over an episode.
    dead = float(state[4]) + float(state[9])

    # Openness bonus: +1 at stage 0, +0.75 at stage 1, ..., 0 at stage 4.
    # Rewards keeping the economy open whenever it's safe to do so.
    openness_bonus = OPENNESS_BONUS_MAX * (4 - stage) / 4.0

    # Strictest-lockdown penalty: only applied when the agent is in stage 4.
    # The user asked for stage 4 to "come with a high penalty".
    stage4_penalty = STAGE4_PENALTY if stage == 4 else 0.0

    # Outbreak penalty: scales with the share of the population in critical
    # condition right now. CRITICAL_WEIGHT is tuned so that ~0.5% of the
    # population in critical condition already wipes out the openness bonus.
    critical_penalty = CRITICAL_WEIGHT * critical

    # Death penalty: heavy and persistent — once people have died, the
    # deceased share stays in the observation and keeps subtracting.
    death_penalty = DEATH_WEIGHT * dead

    return openness_bonus - stage4_penalty - critical_penalty - death_penalty
