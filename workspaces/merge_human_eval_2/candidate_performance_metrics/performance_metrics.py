"""Candidate performance metrics for the highway-env `merge-v0` task.

Each function represents an INDEPENDENT stakeholder's view of a merge
trajectory, justified purely from generic highway-driving domain
knowledge. For every metric, HIGHER ALWAYS MEANS BETTER for that
stakeholder; naturally lower-is-better quantities are negated. All
metrics compute from `traj` alone (no env access, no external state).

Verified denormalisation facts (against the live simulator):
  - ego forward speed (m/s)            = next_obs[0, 3] * 80.0
  - neighbour relative x (m)           = obs[i, 1] * 200.0   (i >= 1)
  - neighbour relative y (m)           = obs[i, 2] * 8.0     (i >= 1)
  - crash on a step                    = next_obs[0, 0] < 0.5
"""

import numpy as np

_SPEED_MPS = 80.0
_LONG_M = 200.0
_LAT_M = 8.0
_FAR_M = 200.0


def _as_int(action):
    arr = np.asarray(action).reshape(-1)
    if arr.size == 0:
        return 0
    return int(arr[0])


def _ego_speed_mps(next_obs):
    return float(np.asarray(next_obs)[0, 3]) * _SPEED_MPS


def _crashed(next_obs):
    return float(np.asarray(next_obs)[0, 0]) < 0.5


def _nearest_neighbour_gap_m(obs):
    """Straight-line distance (m) to the nearest present neighbour, or _FAR_M."""
    o = np.asarray(obs)
    best = None
    for r in range(1, o.shape[0]):
        if float(o[r, 0]) <= 0.5:
            continue
        dx = float(o[r, 1]) * _LONG_M
        dy = float(o[r, 2]) * _LAT_M
        d = float(np.hypot(dx, dy))
        if best is None or d < best:
            best = d
    return best if best is not None else _FAR_M


def performance_metric_safety(traj):
    """Safety reviewer: rewards keeping a wide buffer to other vehicles and avoiding collisions."""
    if not traj:
        return 0.0
    gaps = [_nearest_neighbour_gap_m(obs) for (obs, _, _) in traj]
    crashed = any(_crashed(no) for (_, _, no) in traj)
    score = float(np.mean(gaps))
    if crashed:
        score -= 100.0  # a collision is a catastrophic safety failure
    return float(score)


def performance_metric_throughput(traj):
    """Operations / road-network stakeholder: rewards forward progress, forfeited entirely on a crash."""
    if not traj:
        return 0.0
    # A crashed ego blocks the merge point and delivers no usable throughput.
    if any(_crashed(no) for (_, _, no) in traj):
        return 0.0
    speeds = [_ego_speed_mps(no) for (_, _, no) in traj]
    return float(np.sum(speeds))  # 1-second steps -> metres covered


def performance_metric_comfort(traj):
    """Passenger comfort: penalises abrupt speed changes and frequent lane changes."""
    if len(traj) < 2:
        return 0.0
    speeds = np.asarray([_ego_speed_mps(no) for (_, _, no) in traj], dtype=float)
    jerk_cost = float(np.sum(np.abs(np.diff(speeds))))  # |accel| over 1-s steps
    lane_changes = sum(1 for (_, act, _) in traj if _as_int(act) in (0, 2))
    # Negate: smoother speed and fewer lane changes score higher.
    return float(-(jerk_cost + 5.0 * lane_changes))


def performance_metric_eco(traj):
    """Energy / efficiency stakeholder: penalises high speed and hard acceleration."""
    if not traj:
        return 0.0
    speeds = np.asarray([_ego_speed_mps(no) for (_, _, no) in traj], dtype=float)
    # Aerodynamic drag/fuel use scales with speed^2; accelerations cost energy.
    drag_cost = float(np.sum(speeds ** 2)) / 100.0
    accel_cost = float(np.sum(np.abs(np.diff(speeds)))) if speeds.size > 1 else 0.0
    return float(-(drag_cost + accel_cost))
