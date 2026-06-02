"""Alternative-perspective performance metrics for the pandemic regulator.

Each metric scores a full trajectory under a perspective that is *not*
the current reward-designer's. Higher is always "better" by that
perspective's lights — metrics that are naturally lower-is-better
return the negation. Trajectory format: list of (obs, action, next_obs)
tuples, with obs shape (history, 1, 13).
"""

import numpy as np


def _latest(o):
    return np.asarray(o)[-1, 0]


def _stages(traj):
    return np.array([int(_latest(no)[1]) for (_o, _a, no) in traj])


def _critical(traj):
    return np.array([float(_latest(no)[3]) + float(_latest(no)[8]) for (_o, _a, no) in traj])


def _infected(traj):
    return np.array([float(_latest(no)[5]) + float(_latest(no)[10]) for (_o, _a, no) in traj])


def _dead_final(traj):
    if len(traj) == 0:
        return 0.0
    last = _latest(traj[-1][2])
    return float(last[4]) + float(last[9])


def performance_metric_public_health_hawk(traj) -> float:
    """Public-health hawk: any outbreak or any death is unacceptable; prefers strict suppression."""
    if len(traj) == 0:
        return 0.0
    peak_crit = float(np.max(_critical(traj)))
    final_dead = _dead_final(traj)
    # 200x peak-critical + 5000x final deaths, negated so higher = better.
    return -(200.0 * peak_crit + 5000.0 * final_dead)


def performance_metric_economy_first(traj) -> float:
    """Economic perspective: keeping the town open and businesses running is paramount."""
    if len(traj) == 0:
        return 0.0
    # Mean stage, negated so higher = more open = better.
    return -float(np.mean(_stages(traj)))


def performance_metric_regulatory_stability(traj) -> float:
    """Residents and businesses: regulation should be predictable; frequent stage changes are disruptive."""
    stages = _stages(traj)
    if len(stages) < 2:
        return 0.0
    n_changes = int(np.sum(np.diff(stages) != 0))
    # Frequency of policy flips, negated so higher = more stable = better.
    return -float(n_changes) / float(len(stages) - 1)


def performance_metric_total_infection_burden(traj) -> float:
    """Equity / patient-advocacy: every infection is a harm, regardless of severity or outcome."""
    if len(traj) == 0:
        return 0.0
    # Cumulative day-weighted infection share (non-critical + critical), negated.
    crit = _critical(traj)
    inf = _infected(traj)
    return -float(np.sum(crit + inf))
