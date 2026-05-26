"""Candidate performance metrics for the highway `merge-v0` task.

Each function encodes one independent stakeholder's view of what makes a
merge trajectory good. Higher is always better by that stakeholder's
lights (naturally-lower-is-better quantities are negated). Metrics are
self-contained — they compute from the raw `(obs, act, next_obs)` tuples
only, with no env access and no shared helpers.

Denormalisation constants come from highway-env's Kinematics
features_range for merge-v0 (verified empirically):
  ego speed (m/s) = obs[0, 3] * 80
  relative longitudinal gap (m) = obs[i, 1] * 200
  relative lateral offset (m) = obs[i, 2] * 8
A crash on a step is signalled by next_obs[0, 0] < 0.5 (see env_config).
"""

import numpy as np

_VX_HALFWIDTH_MPS = 80.0
_X_HALFWIDTH_M = 200.0
_Y_HALFWIDTH_M = 8.0

# Action ids (DiscreteMetaAction): 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT,
# 3 FASTER, 4 SLOWER.
_LANE_CHANGE_ACTIONS = (0, 2)


def _ego_speed_mps(o):
    return float(np.asarray(o)[0, 3]) * _VX_HALFWIDTH_MPS


def _nearest_gap_m(o, far=200.0):
    o = np.asarray(o)
    gaps = []
    for i in range(1, o.shape[0]):
        if o[i, 0] > 0.5:
            rx = o[i, 1] * _X_HALFWIDTH_M
            ry = o[i, 2] * _Y_HALFWIDTH_M
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else far


def _crashed(o):
    return float(np.asarray(o)[0, 0]) < 0.5


def _action_id(a):
    return int(np.asarray(a).item()) if np.ndim(a) else int(a)


def performance_metric_safety_margin(traj) -> float:
    """Safety reviewer: prefers trajectories whose smallest distance to any neighbour stays large."""
    if not traj:
        return 0.0
    gaps = [_nearest_gap_m(next_obs) for _, _, next_obs in traj]
    # A crash is treated as the worst possible safety outcome.
    if any(_crashed(next_obs) for _, _, next_obs in traj):
        return -1.0  # m, well below any plausible real min-gap
    return float(min(gaps))


def performance_metric_passenger_comfort(traj) -> float:
    """Passenger comfort: prefers smooth speed (low jerk / few abrupt accel-brake cycles)."""
    if len(traj) < 2:
        return 0.0
    speeds = np.asarray([_ego_speed_mps(next_obs) for _, _, next_obs in traj])
    # Jerk proxy: second difference of speed, RMS per step in m/s^2 (1 Hz policy).
    if len(speeds) < 3:
        return 0.0
    second_diff = np.diff(speeds, n=2)
    rms = float(np.sqrt(np.mean(second_diff ** 2)))
    return -rms  # lower jerk is better, so negate


def performance_metric_lane_discipline(traj) -> float:
    """Driving instructor: prefers minimal lane-change churn (only switch lanes when needed)."""
    if not traj:
        return 0.0
    n = len(traj)
    changes = sum(1 for _, act, _ in traj if _action_id(act) in _LANE_CHANGE_ACTIONS)
    return -(changes / n)  # fewer is better, so negate the fraction


def performance_metric_throughput(traj) -> float:
    """Highway operator: prefers covering more longitudinal distance per episode (m)."""
    if not traj:
        return 0.0
    # Step duration is 1 s at the default 1 Hz policy frequency in merge-v0.
    # Integrate forward speed across steps to get distance travelled, while
    # zeroing-out any step after a crash so phantom motion isn't credited.
    dist = 0.0
    for _, _, next_obs in traj:
        if _crashed(next_obs):
            break
        dist += max(0.0, _ego_speed_mps(next_obs))
    return float(dist)


def performance_metric_merge_completion(traj) -> float:
    """Merge supervisor: prefers reaching and holding the mainline lane (lateral merge completed)."""
    if not traj:
        return 0.0
    # The ego's absolute lateral position lives in obs[0, 2] (normalised by
    # Y_HALFWIDTH = 8 m). On merge-v0 the on-ramp sits at a more-positive y
    # than the mainline; smaller absolute |y| means closer to the mainline
    # centreline. Reward the fraction of steps the ego is within ~2 m of
    # the mainline centre (|y_abs_m| <= 2), so a stakeholder that wants the
    # merge actually completed scores higher when it is.
    in_lane = 0
    n = 0
    for _, _, next_obs in traj:
        y_m = float(np.asarray(next_obs)[0, 2]) * _Y_HALFWIDTH_M
        if abs(y_m) <= 2.0:
            in_lane += 1
        n += 1
    return float(in_lane / n) if n else 0.0
