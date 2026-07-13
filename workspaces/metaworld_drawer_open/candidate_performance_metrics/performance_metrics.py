"""Alternative-perspective performance metrics for the drawer-open task.

Each function scores one trajectory (higher = better from that perspective).
These are selection signals only; they are never shown to the user.
"""

import numpy as np

MAX_DIST = 0.2  # meters of handle travel from closed to fully open
SUCCESS_TOL = 0.03  # meters; the task's own "drawer is open" tolerance
NEAR_HANDLE_TOL = 0.03  # meters; hand counts as "at the handle" within this
HANDLE_MOVE_EPS = 1e-3  # meters/step; handle counts as moving above this
SAFE_HANDLE_SPEED = 0.01  # meters/step; handle motion faster than this counts as a yank


def _series(traj):
    """Per-step arrays: handle->target dist, hand->handle dist, aperture, handle step-motion."""
    handle_to_target, hand_to_handle, aperture, handle_speed = [], [], [], []
    for obs, _act, next_obs in traj:
        o = np.asarray(next_obs, dtype=np.float64)
        p = np.asarray(obs, dtype=np.float64)
        handle_to_target.append(float(np.linalg.norm(o[4:7] - o[39:42])))
        hand_to_handle.append(float(np.linalg.norm(o[4:7] - o[0:3])))
        aperture.append(float(o[3]))
        handle_speed.append(float(np.linalg.norm(o[4:7] - p[4:7])))
    return (
        np.asarray(handle_to_target),
        np.asarray(hand_to_handle),
        np.asarray(aperture),
        np.asarray(handle_speed),
    )


def performance_metric_task_success(traj) -> float:
    """Task success: the drawer is open at the end of the attempt."""
    if not traj:
        return 0.0
    handle_to_target, _, _, _ = _series(traj)
    return float(handle_to_target[-1] <= SUCCESS_TOL)


def performance_metric_time_to_open(traj) -> float:
    """Impatient user: the drawer should be open as early in the attempt as possible."""
    if not traj:
        return 0.0
    handle_to_target, _, _, _ = _series(traj)
    opened = handle_to_target <= SUCCESS_TOL
    if not opened.any():
        return 0.0
    first = int(np.argmax(opened))
    return float(1.0 - first / len(traj))


def performance_metric_energy_efficiency(traj) -> float:
    """Operations: achieve the task with as little commanded motion as possible."""
    if not traj:
        return 0.0
    actions = np.asarray([np.asarray(a, dtype=np.float64).reshape(-1) for _, a, _ in traj])
    return float(-np.abs(actions).mean())


def performance_metric_motion_smoothness(traj) -> float:
    """Safety reviewer: the arm should move smoothly, without jerky command reversals."""
    if len(traj) < 2:
        return 0.0
    actions = np.asarray([np.asarray(a, dtype=np.float64).reshape(-1) for _, a, _ in traj])
    return float(-np.abs(np.diff(actions, axis=0)).mean())


def performance_metric_proper_grasp(traj) -> float:
    """Demonstration quality: the drawer should be opened by gripping the handle, not hooking or shoving it."""
    if not traj:
        return 0.0
    _, hand_to_handle, aperture, handle_speed = _series(traj)
    moving = handle_speed > HANDLE_MOVE_EPS
    if not moving.any():
        return 0.0
    # Per-step grip quality while the drawer is actually moving: hand on the
    # handle, fingers closed around it (aperture 1.0 = wide open, 0.0 = shut).
    at_handle = (hand_to_handle <= NEAR_HANDLE_TOL).astype(float)
    closedness = np.clip(1.0 - aperture, 0.0, 1.0)
    return float((at_handle * closedness)[moving].mean())


def performance_metric_gentle_handling(traj) -> float:
    """Hardware maintenance: the drawer should never be yanked or slammed at high speed."""
    if not traj:
        return 0.0
    _, _, _, handle_speed = _series(traj)
    # Only speeds above a safe handling limit are penalized.
    return float(-max(0.0, float(handle_speed.max()) - SAFE_HANDLE_SPEED))
