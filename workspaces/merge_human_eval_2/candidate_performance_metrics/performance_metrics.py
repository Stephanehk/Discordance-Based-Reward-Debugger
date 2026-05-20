"""Candidate performance metrics for highway-env merge-v0.

Each function represents an independent stakeholder's view of a trajectory,
justified purely from general highway-driving domain knowledge. Higher always
means "better" by that stakeholder's lights; naturally lower-is-better
quantities are negated. All metrics compute from `traj` alone.
"""

import numpy as np

_SPEED_MPS = 80.0
_LONG_M = 200.0


def _ego_speed_mps(next_obs):
    return float(np.asarray(next_obs)[0, 3]) * _SPEED_MPS


def _crashed(next_obs):
    return float(np.asarray(next_obs)[0, 0]) < 0.5


def _nearest_lead_gap_m(next_obs):
    o = np.asarray(next_obs)
    gaps = []
    for r in range(1, o.shape[0]):
        if float(o[r, 0]) <= 0.5:
            continue
        dx = float(o[r, 1]) * _LONG_M
        if dx > 0.0:
            gaps.append(dx)
    return min(gaps) if gaps else _LONG_M


def performance_metric_safety(traj):
    """Safety reviewer: rewards keeping a wide following gap and no collisions."""
    if not traj:
        return 0.0
    gaps = [_nearest_lead_gap_m(no) for (_, _, no) in traj]
    crashed = any(_crashed(no) for (_, _, no) in traj)
    mean_gap = float(np.mean(gaps))
    score = mean_gap
    if crashed:
        score -= 100.0  # a collision is a catastrophic safety failure
    return float(score)


def performance_metric_throughput(traj):
    """Operations / road-network stakeholder: rewards forward progress completed."""
    if not traj:
        return 0.0
    # Distance covered, but progress is forfeited if the episode crashes
    # (a crashed vehicle blocks the merge and contributes no throughput).
    speeds = [_ego_speed_mps(no) for (_, _, no) in traj]
    distance = float(np.sum(speeds))  # 1-second steps
    if any(_crashed(no) for (_, _, no) in traj):
        return 0.0
    return distance


def performance_metric_comfort(traj):
    """Passenger comfort: penalises abrupt speed changes and frequent lane changes."""
    if len(traj) < 2:
        return 0.0
    speeds = np.asarray([_ego_speed_mps(no) for (_, _, no) in traj], dtype=float)
    accel = np.diff(speeds)  # m/s per 1-s step ~ m/s^2
    jerk_cost = float(np.sum(np.abs(accel)))
    lane_changes = 0
    for (_, act, _) in traj:
        a = int(act) if np.ndim(act) == 0 else int(np.asarray(act).flatten()[0])
        if a in (0, 2):
            lane_changes += 1
    # Negate: smoother (less |accel| and fewer lane changes) scores higher.
    return float(-(jerk_cost + 5.0 * lane_changes))


def performance_metric_eco(traj):
    """Energy / efficiency stakeholder: penalises high speed and hard acceleration."""
    if not traj:
        return 0.0
    speeds = np.asarray([_ego_speed_mps(no) for (_, _, no) in traj], dtype=float)
    # Aerodynamic drag/fuel scales superlinearly with speed; accelerations
    # cost energy. Both negated so a frugal, moderate-speed run scores higher.
    drag_cost = float(np.sum(speeds ** 2)) / 100.0
    accel_cost = float(np.sum(np.abs(np.diff(speeds)))) if len(speeds) > 1 else 0.0
    return float(-(drag_cost + accel_cost))
