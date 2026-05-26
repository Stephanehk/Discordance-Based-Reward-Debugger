"""Candidate performance metrics for the pandemic-control environment.

Each metric encodes a *different* stakeholder's view of what makes a
public-health regulation trajectory good. Every metric is higher-is-better
by that stakeholder's lights (lower-is-better quantities are negated).

A trajectory is a list of (obs, action, next_obs) tuples. The latest state
vector is `np.asarray(next_obs)[-1, 0]`, length 13. Combined critical-case
proportion = obs[3] + obs[8]; combined currently-infected (non-critical)
proportion = obs[5] + obs[10]; combined recovered proportion = obs[7] + obs[12].
The chosen regulation stage each day is the action itself (an integer 0..4).
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector from a (history,1,13) obs."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    return float(state[3] + state[8])


def _infected(state):
    return float(state[5] + state[10])


def _stages(traj):
    """Per-day regulation stage chosen by the agent (the action), in order."""
    return np.array([float(act) for (_o, act, _n) in traj], dtype=float)


def _next_states(traj):
    """Per-day post-action state vectors, in episode order."""
    return [_latest(next_obs) for (_o, _a, next_obs) in traj]


def performance_metric_economy(traj):
    """The local-business stakeholder: higher means fewer venues were closed across the episode."""
    if len(traj) == 0:
        return 0.0
    stages = _stages(traj)
    # Each step above stage 0 closes more venues; total restrictiveness is the
    # sum of stages. Negate so higher = lighter touch on the economy.
    return -float(np.sum(stages))


def performance_metric_schools(traj):
    """The schools-and-families stakeholder: higher means schools were open on more days (stage < 2)."""
    if len(traj) == 0:
        return 0.0
    stages = _stages(traj)
    # Schools close at stage >= 2 (per the action description). Count days open.
    return float(np.sum(stages < 2))


def performance_metric_hospital(traj):
    """The hospital acute-care stakeholder: higher means the worst-day critical-case load was lower."""
    if len(traj) == 0:
        return 0.0
    crit = np.array([_critical(s) for s in _next_states(traj)], dtype=float)
    # Hospitals care about the peak, not the average — one overwhelmed day is
    # the failure mode. Negate so higher = lower peak.
    return -float(np.max(crit))


def performance_metric_civic_stability(traj):
    """The civic-stability stakeholder: higher means the regulation stage changed less often day-to-day."""
    if len(traj) <= 1:
        return 0.0
    stages = _stages(traj)
    # Whiplash policy erodes trust and compliance; count day-over-day changes
    # and negate so higher = steadier policy.
    return -float(np.sum(np.diff(stages) != 0))


def performance_metric_epidemiologist(traj):
    """The public-health epidemiologist: higher means a smaller total infection burden over the episode."""
    if len(traj) == 0:
        return 0.0
    inf = np.array([_infected(s) for s in _next_states(traj)], dtype=float)
    crit = np.array([_critical(s) for s in _next_states(traj)], dtype=float)
    # Total person-days of active illness (infected + critical) across the
    # episode. Negate so higher = less total disease burden.
    return -float(np.sum(inf + crit))
