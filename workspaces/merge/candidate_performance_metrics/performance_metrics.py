"""Candidate alternative-perspective metrics for highway-env merge-v0.

Each metric returns a float and uses the convention "higher is better
by this perspective's lights"; metrics that are naturally
lower-is-better (counts of disruptive actions, fuel-burning maneuvers)
return the negation.

These represent perspectives *other* than the one who specified the
designed reward (which emphasises ego speed and avoiding collisions /
proximity). They cover passenger comfort, fleet/operations efficiency,
merge-completion success, safety-reviewer headroom, and other-driver
courtesy.
"""

import numpy as np


# Action ids (Discrete(5)): 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT,
# 3=FASTER, 4=SLOWER.
_LANE_CHANGE_ACTIONS = (0, 2)
_SPEED_CHANGE_ACTIONS = (3, 4)

# Threshold (normalised inter-vehicle distance) at which a lane change
# is "next to other traffic" and therefore disruptive to those drivers.
_CLOSE_FOR_COURTESY = 0.15


def _min_other_distance(obs) -> float:
    others = np.asarray(obs[1:5])
    present = others[:, 0] > 0.5
    if not np.any(present):
        return 1.0
    rel = others[present, 1:3]
    return float(np.linalg.norm(rel, axis=1).min())


def _crashed_any(traj) -> bool:
    return any(float(t[2][0, 0]) < 0.5 for t in traj)


def performance_metric_passenger_comfort(traj) -> float:
    """Passenger comfort: fewer maneuvers (lane changes + speed changes) make for a smoother ride."""
    if not traj:
        return 0.0
    n = sum(
        1 for (_, a, _) in traj
        if int(a) in _LANE_CHANGE_ACTIONS or int(a) in _SPEED_CHANGE_ACTIONS
    )
    return -float(n)


def performance_metric_fuel_efficiency(traj) -> float:
    """Fleet operator: minimise accelerations and decelerations to reduce fuel burn."""
    if not traj:
        return 0.0
    n = sum(1 for (_, a, _) in traj if int(a) in _SPEED_CHANGE_ACTIONS)
    return -float(n)


def performance_metric_merge_completion(traj) -> float:
    """Engineering/product: complete the merge — make it all the way down the road segment without crashing."""
    if not traj:
        return 0.0
    if _crashed_any(traj):
        return -1.0
    return float(traj[-1][2][0, 1])


def performance_metric_safety_margin(traj) -> float:
    """Safety reviewer: maintain a generous distance from other vehicles throughout the trajectory."""
    if not traj:
        return 0.0
    dists = np.array([_min_other_distance(t[2]) for t in traj], dtype=float)
    return float(dists.mean())


def performance_metric_traffic_courtesy(traj) -> float:
    """Other drivers: avoid lane changes performed next to nearby traffic (disruptive cut-ins)."""
    if not traj:
        return 0.0
    disruptive = 0
    for (obs, a, _) in traj:
        if int(a) in _LANE_CHANGE_ACTIONS and _min_other_distance(obs) < _CLOSE_FOR_COURTESY:
            disruptive += 1
    return -float(disruptive)
