"""Candidate performance metrics for the highway `merge-v0` task.

Each function scores a trajectory from one stakeholder's perspective.
Higher always means "better" by that stakeholder's lights. Metrics are
self-contained: they read only ``traj`` (a list of (obs, act, next_obs)
tuples) and denormalise the observation the same way the trajectory
features do (verified against the live simulator).

The designed reward currently maximises raw forward speed (and penalises
crashes). These metrics deliberately span *other* viewpoints, so pairs
where the designed reward and a metric disagree expose what the
speed-only reward leaves out. Gaps are treated as continuous metres; no
fixed "safe distance" threshold is assumed.
"""

import numpy as np

VX_HALFWIDTH_MPS = 80.0  # m/s — normalised vx maps back via *80
X_HALFWIDTH_M = 200.0    # m — normalised longitudinal position half-range
Y_HALFWIDTH_M = 8.0      # m — normalised lateral position half-range

_LANE_CHANGE_ACTIONS = (0, 2)  # DiscreteMetaAction: LANE_LEFT, LANE_RIGHT

# Env reward_speed_range [20, 30] m/s, midpoint 25.
_SPEED_BAND_MID = 25.0

_FAR_GAP_M = 200.0  # treat "no neighbour present" as a large gap


def _ego_speed(next_obs):
    return float(np.asarray(next_obs)[0, 3]) * VX_HALFWIDTH_MPS


def _nearest_gap(next_obs):
    obs = np.asarray(next_obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:
            rx = obs[i, 1] * X_HALFWIDTH_M
            ry = obs[i, 2] * Y_HALFWIDTH_M
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else _FAR_GAP_M


def _crashed(next_obs):
    return float(np.asarray(next_obs)[0, 0]) < 0.5


def _action_id(act):
    return int(np.asarray(act).item()) if np.ndim(act) else int(act)


def performance_metric_safety_clearance(traj):
    """Safety reviewer: prefers larger continuous clearance to other vehicles and no collisions."""
    if not traj:
        return 0.0
    gaps = np.array([_nearest_gap(n) for _, _, n in traj])
    crashed = any(_crashed(n) for _, _, n in traj)
    # Reward the smallest clearance reached (the tightest moment) plus the
    # average clearance, and dock heavily for ending in a collision.
    crash_penalty = 100.0 if crashed else 0.0
    return float(0.5 * gaps.min() + 0.5 * gaps.mean() - crash_penalty)


def performance_metric_ride_comfort(traj):
    """Passenger-comfort stakeholder: prefers smooth driving with few lane changes and gentle speed changes."""
    if not traj:
        return 0.0
    speeds = np.array([_ego_speed(n) for _, _, n in traj])
    jerk = float(np.abs(np.diff(speeds)).mean()) if len(speeds) > 1 else 0.0
    lane_change_rate = float(np.mean([
        1.0 if _action_id(a) in _LANE_CHANGE_ACTIONS else 0.0
        for _, a, _ in traj
    ]))
    return float(-(jerk + 10.0 * lane_change_rate))


def performance_metric_band_speed(traj):
    """Highway operator: prefers speeds near the 25 m/s midpoint of the rewarded band, neither crawling nor speeding."""
    if not traj:
        return 0.0
    speeds = np.array([_ego_speed(n) for _, _, n in traj])
    # Negated mean absolute deviation from the band midpoint.
    return float(-np.abs(speeds - _SPEED_BAND_MID).mean())


def performance_metric_throughput(traj):
    """Traffic-operations stakeholder: prefers brisk, sustained forward progress through the merge."""
    if not traj:
        return 0.0
    speeds = np.array([_ego_speed(n) for _, _, n in traj])
    # Mean speed (correlates positively with the designed reward), but a
    # collision halts throughput entirely.
    if any(_crashed(n) for _, _, n in traj):
        return 0.0
    return float(speeds.mean())
