"""Alternative-perspective performance metrics for RoboPianist.

Each metric scores a trajectory `[(obs, act, next_obs), ...]` from a perspective
*other* than the designed reward, computed self-contained from the physical-
readout block at the tail of each observation vector. Higher = better for that
perspective. Docstrings are the one-line perspective labels.
"""

import numpy as np

_AUG_DIM = 339


def _unpack(next_obs):
    a = np.asarray(next_obs, dtype=np.float64)[-_AUG_DIM:]
    return {
        "goal_keys": a[0:88],
        "norm_state": a[89:177],
        "activation": a[177:265],
        "hand_power": a[336:338],
    }


def performance_metric_musical_accuracy(traj) -> float:
    """Audience: how accurately the actual notes match the song (note F1)."""
    tp = fp = fn = 0
    for _obs, _act, next_obs in traj:
        d = _unpack(next_obs)
        goal = d["goal_keys"] > 0.5
        act = d["activation"] > 0.5
        tp += int(np.sum(goal & act))
        fp += int(np.sum(~goal & act))
        fn += int(np.sum(goal & ~act))
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def performance_metric_energy_efficiency(traj) -> float:
    """Operations: less total actuator power spent is better."""
    powers = [float(_unpack(no)["hand_power"].sum()) for _obs, _act, no in traj]
    return -float(np.mean(powers)) if powers else 0.0


def performance_metric_motion_smoothness(traj) -> float:
    """Safety reviewer: smoother, lower-jerk hand motion is better."""
    acts = [np.asarray(act, dtype=np.float64) for _obs, act, _no in traj]
    if len(acts) < 2:
        return 0.0
    deltas = [float(np.linalg.norm(acts[i] - acts[i - 1])) for i in range(1, len(acts))]
    return -float(np.mean(deltas))


def performance_metric_hand_balance(traj) -> float:
    """Ergonomics: the two hands should share the workload, not overload one."""
    left = right = 0.0
    for _obs, _act, next_obs in traj:
        p = _unpack(next_obs)["hand_power"]
        right += float(p[0])
        left += float(p[1])
    total = right + left
    if total <= 0:
        return 1.0
    return 1.0 - abs(right - left) / total          # 1 = perfectly balanced


def performance_metric_instrument_gentleness(traj) -> float:
    """Piano owner: play softly — avoid mashing keys far past the sounding point."""
    depths = []
    for _obs, _act, next_obs in traj:
        d = _unpack(next_obs)
        pressed = d["activation"] > 0.5
        if np.any(pressed):
            depths.append(float(np.mean(d["norm_state"][pressed])))
    return -float(np.mean(depths)) if depths else 0.0
