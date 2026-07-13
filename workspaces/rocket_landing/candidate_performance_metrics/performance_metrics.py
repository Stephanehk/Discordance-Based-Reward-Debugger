"""Alternative-perspective performance metrics for PyFlyt/Rocket-Landing-v4.

Each function scores one trajectory from a perspective *other* than the designed
reward, higher = better for that perspective. Used only under the hood to find
disagreements; the docstrings are the only thing ever shown to the user.
"""

import numpy as np


def _quat_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _ground_vertical_speed(state):
    rot = _quat_to_rotmat(state[3:7])
    return float(rot[2, :] @ state[7:10])


def _first_contact_index(traj):
    """Index of the first transition whose next_obs shows pad contact, else None."""
    for i, (_, _, next_obs) in enumerate(traj):
        if np.asarray(next_obs, dtype=np.float64)[29] >= 0.5:
            return i
    return None


def performance_metric_landing_accuracy(traj):
    """Precision: land as close to the pad center as possible."""
    last = np.asarray(traj[-1][2], dtype=np.float64)
    return -float(np.linalg.norm(last[10:12]))


def performance_metric_fuel_economy(traj):
    """Operations: use as little engine thrust / fuel as possible."""
    # Fraction of steps the engine burned; less burning is better.
    on = sum(1 for _, _, no in traj if np.asarray(no, dtype=np.float64)[24] >= 0.5)
    return -float(on) / max(len(traj), 1)


def performance_metric_gentle_touchdown(traj):
    """Safety: touch down as slowly as possible (avoid a hard impact)."""
    idx = _first_contact_index(traj)
    if idx is None:
        # Never landed: judge by the worst downward speed reached near the ground.
        near_ground = [
            abs(_ground_vertical_speed(np.asarray(no, dtype=np.float64)))
            for _, _, no in traj
            if np.asarray(no, dtype=np.float64)[12] < 20.0
        ]
        return -float(max(near_ground)) if near_ground else -100.0
    contact = np.asarray(traj[idx][2], dtype=np.float64)
    return -abs(_ground_vertical_speed(contact))


def performance_metric_quick_landing(traj):
    """End user: reach the pad in as few steps as possible."""
    idx = _first_contact_index(traj)
    steps_to_land = idx + 1 if idx is not None else len(traj)
    return -float(steps_to_land)


def performance_metric_control_smoothness(traj):
    """Comfort/regulator: prefer smooth control, avoid jerky action changes."""
    if len(traj) < 2:
        return 0.0
    acts = [np.asarray(a, dtype=np.float64) for _, a, _ in traj]
    deltas = [float(np.linalg.norm(acts[i] - acts[i - 1])) for i in range(1, len(acts))]
    return -float(np.mean(deltas))
