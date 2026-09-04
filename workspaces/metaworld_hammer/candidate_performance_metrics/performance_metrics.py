"""Alternative-perspective performance metrics for the hammer task.

Each metric scores one trajectory from a perspective *other* than the designed
reward's. Higher is better in every case. Used under the hood to find
comparisons where the designed reward and a perspective disagree.
"""

import numpy as np

HEAD_OFFSET = np.array([0.16, 0.06, 0.0])  # meters; hammer head relative to hammer body
NAIL_SUCCESS_DEPTH = 0.09  # meters of nail travel that counts as fully driven in
HAMMER_AT_NAIL_TOL = 0.06  # meters; hammer head counts as "on the nail" within this
HELD_TOL = 0.07  # meters; tcp-to-hammer distance that counts as the hammer in hand
NAIL_MOVE_EPS = 1e-4  # meters/step; nail counts as advancing above this


def _series(traj):
    """Per-step arrays: nail depth, hammer-head->nail-head dist, tcp->hammer dist, actions."""
    nail, head_to_nail, tcp_to_hammer, acts = [], [], [], []
    for _obs, act, next_obs in traj:
        s = np.asarray(next_obs, dtype=np.float64)
        hammer_head = s[4:7] + HEAD_OFFSET
        nail.append(float(s[57]))
        head_to_nail.append(float(np.linalg.norm(hammer_head - s[11:14])))
        tcp_to_hammer.append(float(np.linalg.norm(s[45:48] - s[4:7])))
        acts.append(np.asarray(act, dtype=np.float64))
    return (
        np.array(nail),
        np.array(head_to_nail),
        np.array(tcp_to_hammer),
        np.array(acts),
    )


def performance_metric_task_success(traj) -> float:
    """Task success: the nail ends up fully driven into the wall."""
    if not traj:
        return 0.0
    nail, _, _, _ = _series(traj)
    return float(nail[-1] > NAIL_SUCCESS_DEPTH)


def performance_metric_time_to_finish(traj) -> float:
    """Impatient user: the nail should be driven in as early in the attempt as possible."""
    if not traj:
        return 0.0
    nail, _, _, _ = _series(traj)
    done = np.flatnonzero(nail > NAIL_SUCCESS_DEPTH)
    if done.size == 0:
        return 0.0
    return 1.0 - float(done[0]) / len(nail)  # 1.0 = driven immediately, 0.0 = never


def performance_metric_tool_use(traj) -> float:
    """Craft quality: the nail should be sunk by striking it with the hammer, not shoved by the arm."""
    if not traj:
        return 0.0
    nail, head_to_nail, _, _ = _series(traj)
    gains = np.diff(nail, prepend=nail[0])  # meters of nail travel per step
    moved = gains > NAIL_MOVE_EPS
    total = float(gains[moved].sum())
    if total <= 0.0:
        return 0.0  # the nail never moved; no craft to speak of
    with_hammer = float(gains[moved & (head_to_nail < HAMMER_AT_NAIL_TOL)].sum())
    return with_hammer / total  # fraction of the nail's travel delivered by the hammer


def performance_metric_energy_efficiency(traj) -> float:
    """Operations: finish with as little commanded arm motion as possible."""
    if not traj:
        return 0.0
    _, _, _, acts = _series(traj)
    effort = float(np.mean(np.linalg.norm(acts[:, :3], axis=1)))  # mean command size
    return 1.0 / (1.0 + effort)


def performance_metric_gentle_impact(traj) -> float:
    """Hardware maintenance: the nail should be tapped in progressively, never sunk in one violent blow."""
    if not traj:
        return 0.0
    nail, _, _, _ = _series(traj)
    gains = np.diff(nail, prepend=nail[0])
    worst = float(gains.max()) if gains.size else 0.0  # biggest single-step nail travel
    return 1.0 / (1.0 + worst / 0.01)  # 0.01 m in one step is already a hard strike


def performance_metric_tool_retention(traj) -> float:
    """Reliability: the arm should keep hold of the hammer rather than dropping or losing it."""
    if not traj:
        return 0.0
    _, _, tcp_to_hammer, _ = _series(traj)
    grasped = np.flatnonzero(tcp_to_hammer < HELD_TOL)
    if grasped.size == 0:
        return 0.0  # never picked the hammer up at all
    # once first grasped, what fraction of the remaining steps is it still in hand?
    after = tcp_to_hammer[grasped[0] :]
    return float(np.mean(after < HELD_TOL))
