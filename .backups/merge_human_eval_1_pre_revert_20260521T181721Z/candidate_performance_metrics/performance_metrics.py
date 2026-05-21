"""Candidate performance metrics for the highway `merge-v0` task.

Each function below encodes a *different* stakeholder's view of what a
good merge trajectory looks like, computed directly from the trajectory
(a list of (obs, act, next_obs) tuples). Every metric is oriented so that
**higher is better** from that stakeholder's perspective; perspectives
that are naturally "lower is better" return a negation.

Denormalization (verified against the live simulator's ground-truth state):
  - ego forward speed (m/s) = obs[0, 3] * VX_HALFWIDTH_MPS
  - neighbour relative position (m): x = col1 * X_HALFWIDTH_M,
                                      y = col2 * Y_HALFWIDTH_M
  - neighbour relative velocity (m/s): vx = col3 * VX_HALFWIDTH_MPS,
                                       vy = col4 * VX_HALFWIDTH_MPS
Rows 1..4 of the Kinematics observation hold the closest other vehicles
in coordinates relative to the ego; col 0 is presence. A crash on a step
is signalled by next_obs[0, 0] < 0.5 (ego presence forced to 0).
"""

import numpy as np

# highway-env Kinematics features_range for merge-v0.
VX_HALFWIDTH_MPS = 80.0  # m/s — normalised vx/vy in [-1, 1] map back via *80
X_HALFWIDTH_M = 200.0    # m — normalised longitudinal position half-range
Y_HALFWIDTH_M = 8.0      # m — normalised lateral position half-range

# Action ids (DiscreteMetaAction).
_LANE_LEFT, _IDLE, _LANE_RIGHT, _FASTER, _SLOWER = 0, 1, 2, 3, 4
_LANE_CHANGE_ACTIONS = (_LANE_LEFT, _LANE_RIGHT)

# Gap used when no neighbour is present in an observation, in metres.
_FAR_GAP_M = 200.0


def _ego_speed_mps(obs):
    """Ego forward speed in m/s from one observation matrix."""
    return float(np.asarray(obs)[0, 3]) * VX_HALFWIDTH_MPS


def _nearest_gap_m(obs):
    """Euclidean distance (m) to the nearest present neighbour; _FAR_GAP_M if none."""
    obs = np.asarray(obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:  # present (real, non-padding) neighbour
            rx = obs[i, 1] * X_HALFWIDTH_M  # m, relative longitudinal
            ry = obs[i, 2] * Y_HALFWIDTH_M  # m, relative lateral
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else _FAR_GAP_M


def _crashed(traj):
    """True if any step in the trajectory ended in a collision."""
    return any(float(np.asarray(no)[0, 0]) < 0.5 for _, _, no in traj)


def _action_id(act):
    """Coerce an action to an int id."""
    return int(np.asarray(act).item()) if np.ndim(act) else int(act)


def performance_metric_collision_safety(traj) -> float:
    """Safety regulator: a trajectory is acceptable only if it never collides."""
    if not traj:
        return 0.0
    return 0.0 if _crashed(traj) else 1.0


def performance_metric_following_distance(traj) -> float:
    """Insurance / safety reviewer: prefers larger headway to the nearest vehicle (m)."""
    if not traj:
        return 0.0
    # Use the worst (smallest) observed gap as the headway the reviewer judges:
    # the tightest moment dominates the safety assessment.
    min_gaps = [_nearest_gap_m(next_obs) for _, _, next_obs in traj]
    return float(min(min_gaps))


def performance_metric_throughput(traj) -> float:
    """Highway operator: prefers higher sustained forward speed for road throughput (m/s)."""
    if not traj:
        return 0.0
    speeds = [_ego_speed_mps(next_obs) for _, _, next_obs in traj]
    return float(np.mean(speeds))


def performance_metric_ride_comfort(traj) -> float:
    """Passenger comfort: penalizes abrupt speed changes (negated mean speed jerk, m/s per step)."""
    if len(traj) < 2:
        return 0.0
    speeds = np.asarray([_ego_speed_mps(next_obs) for _, _, next_obs in traj])
    accel = np.abs(np.diff(speeds))  # m/s change between consecutive steps
    return -float(np.mean(accel))


def performance_metric_maneuver_economy(traj) -> float:
    """Fleet maintenance: penalizes frequent lane-change maneuvers (negated fraction of steps)."""
    if not traj:
        return 0.0
    lane_changes = sum(1 for _, act, _ in traj if _action_id(act) in _LANE_CHANGE_ACTIONS)
    return -float(lane_changes / len(traj))
