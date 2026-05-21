"""Candidate performance metrics for the Pandemic Simulator.

Each function scores a trajectory from one stakeholder's perspective. By
convention, higher is always better for that stakeholder, so naturally
lower-is-better quantities are returned negated. These are deliberately
*different* stakeholders from the reward designer, so disagreements with
the designed reward surface where the reward may be over- or under-valuing
some outcome.

A trajectory is a list of (obs, act, next_obs); the latest state row is
next_obs[-1, 0], a length-13 vector (see env_config.OBS_DESCRIPTION):
  [1] stage,
  alpha = [3]critical [4]dead [5]infected,
  delta = [8]critical [9]dead [10]infected.
All category fields are proportions of the population.
"""

import numpy as np


def _row(o):
    return np.asarray(o)[-1, 0]


def performance_metric_economy(traj) -> float:
    """Local-business stakeholder: prefers the lowest, least costly restriction levels."""
    if not traj:
        return 0.0
    stages = [float(_row(next_obs)[1]) for _, _, next_obs in traj]
    return -float(np.mean(stages))  # higher = lighter restrictions on average


def performance_metric_hospital_capacity(traj) -> float:
    """Hospital administrator: prefers the lowest sustained critical-case load on hospital resources."""
    if not traj:
        return 0.0
    critical = [float(_row(next_obs)[3]) + float(_row(next_obs)[8]) for _, _, next_obs in traj]
    return -float(np.mean(critical))  # higher = lighter average load of critical cases


def performance_metric_total_infection_control(traj) -> float:
    """Epidemiologist: prefers fewer total infections, including mild ones, to limit spread."""
    if not traj:
        return 0.0
    infected = [float(_row(next_obs)[5]) + float(_row(next_obs)[10]) for _, _, next_obs in traj]
    return -float(np.mean(infected))  # higher = smaller active-infection burden overall


def performance_metric_mortality(traj) -> float:
    """Bereaved-families advocate: prefers the fewest deaths, weighing mortality above all else."""
    if not traj:
        return 0.0
    last = _row(traj[-1][2])
    return -(float(last[4]) + float(last[9]))  # higher = fewer cumulative deaths


def performance_metric_policy_stability(traj) -> float:
    """Public-communications stakeholder: prefers stable, predictable restriction levels over erratic swings."""
    if not traj:
        return 0.0
    changes = [abs(float(_row(next_obs)[1]) - float(_row(obs)[1])) for obs, _, next_obs in traj]
    return -float(np.mean(changes))  # higher = fewer abrupt changes in restriction level
