"""Candidate performance metrics for the pandemic-control environment.

Each function below encodes the outcome an *independent stakeholder* cares
about, judged from generic domain knowledge of an epidemic-control simulator
alone. Every metric:
  - takes a trajectory `traj = [(obs, act, next_obs), ...]`,
  - reads the latest 13-dim state vector from each step as
    `np.asarray(next_obs)[-1, 0]`,
  - returns a single float oriented so that HIGHER = BETTER for that
    stakeholder (naturally-lower-is-better quantities are negated).

State vector layout (proportions of the population unless noted):
  [0] time_day      [1] stage (0..4)         [2] legacy flag (DO NOT READ)
  [3] alpha_critical [4] alpha_dead [5] alpha_infected [6] alpha_none [7] alpha_recovered
  [8] delta_critical [9] delta_dead [10] delta_infected [11] delta_none [12] delta_recovered
"""

import numpy as np

# Closure cost per regulation stage, from the venue/gathering rules:
# stage 0 fully open; each higher stage closes more venues and tightens
# gathering caps, so the social/economic burden rises monotonically.
_STAGE_CLOSURE_COST = {0: 0.0, 1: 0.2, 2: 0.55, 3: 0.75, 4: 1.0}


def _states(traj):
    """Yield the latest length-13 state vector from each step's next_obs."""
    return [np.asarray(next_obs)[-1, 0] for (_obs, _act, next_obs) in traj]


def performance_metric_hospital_capacity(traj):
    """Hospital capacity: a lower peak critical-case load is better."""
    # Peak combined critical proportion across the trajectory; hospitals
    # care most about the worst moment of strain, not the average.
    peak = 0.0
    for s in _states(traj):
        crit = float(s[3] + s[8])
        if crit > peak:
            peak = crit
    return -peak  # negate: lower peak strain is better


def performance_metric_infection_burden(traj):
    """Epidemic control: a lower total currently-infected burden is better."""
    # Mean combined currently-infected (non-critical) proportion over the
    # episode -- a stand-in for cumulative active-case burden.
    infected = [float(s[5] + s[10]) for s in _states(traj)]
    if not infected:
        return 0.0
    return -float(np.mean(infected))  # negate: less infection is better


def performance_metric_economic_openness(traj):
    """Business community: keeping venues open (lower closure cost) is better."""
    # Average per-day closure cost implied by the regulation stage in force.
    # Higher stages shut more venues, so this stakeholder wants low cost.
    costs = []
    for s in _states(traj):
        stage = int(round(float(s[1])))
        stage = min(max(stage, 0), 4)
        costs.append(_STAGE_CLOSURE_COST[stage])
    if not costs:
        return 0.0
    return -float(np.mean(costs))  # negate: less closure burden is better


def performance_metric_civil_liberties(traj):
    """Civil liberties: less time spent under heavy restrictions is better."""
    # Fraction of days at the two strictest stages (3 or 4), where gatherings
    # are banned and most venues closed; this stakeholder wants it minimized.
    stages = [int(round(float(s[1]))) for s in _states(traj)]
    if not stages:
        return 0.0
    heavy = sum(1 for st in stages if st >= 3)
    return -float(heavy) / len(stages)  # negate: less heavy-restriction time is better


def performance_metric_policy_stability(traj):
    """Planning stability: fewer and smaller swings in the regulation stage are better."""
    # Total absolute day-to-day change in stage; abrupt, frequent policy
    # shifts are costly to plan around for businesses, schools, and households.
    stages = [int(round(float(s[1]))) for s in _states(traj)]
    if len(stages) < 2:
        return 0.0
    churn = float(np.sum(np.abs(np.diff(stages))))
    return -churn  # negate: a steadier policy is better


def performance_metric_proportionality(traj):
    """Proportionality reviewer: restriction severity should track the actual disease level."""
    # Penalize the mismatch between how severe the stage is and how bad the
    # disease burden is on the same day. Both are scaled to [0, 1] so the
    # comparison is on a common footing; the reviewer wants the gap minimized,
    # i.e. heavy restrictions only when the disease is severe and vice versa.
    states = _states(traj)
    if not states:
        return 0.0
    mismatches = []
    for s in states:
        stage = min(max(int(round(float(s[1]))), 0), 4)
        severity = stage / 4.0  # normalized restriction severity
        crit = float(s[3] + s[8])
        infected = float(s[5] + s[10])
        # Disease level: critical strain weighted heavily, capped at 1.0.
        disease = min(1.0, 10.0 * crit + infected)
        mismatches.append(abs(severity - disease))
    return -float(np.mean(mismatches))  # negate: better when severity matches disease
