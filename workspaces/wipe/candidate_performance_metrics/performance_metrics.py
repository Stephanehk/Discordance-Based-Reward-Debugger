"""Candidate performance metrics for the robosuite Wipe task.

Each metric represents a different perspective on what makes a Wipe
trajectory "good", and is computed directly from the trajectory's
appended reward-component channels (the last 10 floats of each obs;
see ``env_config.OBS_DESCRIPTION``). Higher values always mean better
from that perspective — naturally-lower-is-better metrics are
returned negated.
"""

from __future__ import annotations

import numpy as np


# Layout of the appended reward-component channels in obs / next_obs.
# Must stay in sync with ``env_config._WipeRewardEnv._reward_components``.
IDX_TOTAL_WIPED        = -10
IDX_NEW_WIPED          = -9
IDX_MEAN_DISTANCE      = -8
IDX_HAS_CONTACT        = -7
IDX_NCON               = -6
IDX_TOTAL_FORCE_EE     = -5
IDX_MEAN_EE_ACC        = -4
IDX_COLLISION_OR_QLIM  = -3
IDX_NUM_MARKERS        = -2
IDX_HORIZON            = -1

EXCESS_FORCE_THRESHOLD = 60.0


def _stack(trajectory) -> np.ndarray:
    return np.stack([np.asarray(t[2], dtype=np.float64) for t in trajectory], axis=0)


def performance_metric_task_completion(traj) -> float:
    """Operations lead: fraction of dirt actually cleaned by end of the episode."""
    if not traj:
        return 0.0
    S = _stack(traj)
    n = float(S[-1, IDX_NUM_MARKERS]) or 1.0
    return float(S[-1, IDX_TOTAL_WIPED]) / n


def performance_metric_safety(traj) -> float:
    """Safety reviewer: avoid arm collisions and joint-limit saturation (higher = safer)."""
    if not traj:
        return 0.0
    S = _stack(traj)
    return -float((S[:, IDX_COLLISION_OR_QLIM] > 0.5).mean())


def performance_metric_low_force(traj) -> float:
    """Maintenance engineer: avoid pressing too hard on the table or the wiping pad (higher = gentler)."""
    if not traj:
        return 0.0
    S = _stack(traj)
    contact = S[:, IDX_HAS_CONTACT] > 0.5
    if not contact.any():
        return 0.0
    return -float(S[contact, IDX_TOTAL_FORCE_EE].mean())


def performance_metric_motion_smoothness(traj) -> float:
    """Human collaborator: prefers smooth, low-acceleration arm motion (higher = smoother)."""
    if not traj:
        return 0.0
    S = _stack(traj)
    return -float(S[:, IDX_MEAN_EE_ACC].mean())


def performance_metric_cleaning_efficiency(traj) -> float:
    """Productivity: dirt cleaned per control step (higher = more cleaning per second)."""
    if not traj:
        return 0.0
    S = _stack(traj)
    steps = float(S.shape[0]) or 1.0
    return float(S[-1, IDX_TOTAL_WIPED]) / steps
