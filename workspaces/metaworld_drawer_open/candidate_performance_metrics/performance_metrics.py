"""Alternative-perspective performance metrics for the drawer-open task.

Each metric is self-contained (computed from `traj` only) and written so
that a higher value is the more-desirable outcome from that perspective.
These are used under the hood to find attempts where the designed reward
and another reasonable viewpoint disagree.
"""

import numpy as np

MAX_DIST = 0.2
SUCCESS_REMAINING = 0.03
OPEN_THRESH = 0.9


def _openness(o):
    handle = o[4:7]
    target_pos = o[39:42]
    remaining = float(np.linalg.norm(handle - target_pos))
    return float(np.clip(1.0 - remaining / MAX_DIST, 0.0, 1.0))


def _states(traj):
    states = [np.asarray(traj[0][0], dtype=np.float64)]
    for (_, _, next_obs) in traj:
        states.append(np.asarray(next_obs, dtype=np.float64))
    return states


def performance_metric_task_success(traj) -> float:
    """Product/engineering view: did the drawer actually end fully open."""
    if not traj:
        return 0.0
    last = np.asarray(traj[-1][2], dtype=np.float64)
    remaining = float(np.linalg.norm(last[4:7] - last[39:42]))
    return 1.0 if remaining <= SUCCESS_REMAINING else 0.0


def performance_metric_energy_efficiency(traj) -> float:
    """Operations view: less control effort (smaller movement commands) is better."""
    if not traj:
        return 0.0
    acts = np.array([np.asarray(a, dtype=np.float64) for (_, a, _) in traj])
    mean_effort = float(np.mean(np.linalg.norm(acts, axis=1)))
    return -mean_effort


def performance_metric_motion_smoothness(traj) -> float:
    """Safety view: smoother, less jerky arm motion is better."""
    if len(traj) < 2:
        return 0.0
    acts = np.array([np.asarray(a, dtype=np.float64) for (_, a, _) in traj])
    jerk = float(np.mean(np.linalg.norm(np.diff(acts, axis=0), axis=1)))
    return -jerk


def performance_metric_speed(traj) -> float:
    """End-user view: opening the drawer sooner is better."""
    if not traj:
        return 0.0
    states = _states(traj)
    openness = np.array([_openness(s) for s in states])
    open_steps = np.where(openness >= OPEN_THRESH)[0]
    horizon = max(1, len(openness) - 1)
    if open_steps.size == 0:
        return -1.0  # never opened: worst
    return -float(open_steps[0]) / horizon


def performance_metric_sustained_open(traj) -> float:
    """Reliability view: once open, the drawer should stay open through the end."""
    if not traj:
        return 0.0
    states = _states(traj)
    openness = np.array([_openness(s) for s in states])
    open_steps = np.where(openness >= OPEN_THRESH)[0]
    if open_steps.size == 0:
        return 0.0
    first = open_steps[0]
    tail = openness[first:]
    return float(np.mean(tail >= OPEN_THRESH))
