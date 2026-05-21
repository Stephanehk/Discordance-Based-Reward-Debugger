"""Trajectory features for the highway-env `merge-v0` task.

Summarises each rollout in TRUE physical units so a reward designer can
read what happened. Features mix per-step aggregates (mean/min/max ego
speed, nearest-neighbour distance) with trajectory-level summaries
(episode length, crash flag, lane membership, lane-change count). Every
quantity is a raw measured value — no "target" thresholds are baked in
as objectives.

Verified denormalisation facts (against the live simulator):
  - ego forward speed (m/s)            = next_obs[0, 3] * 80.0
  - ego lateral velocity (m/s)         = next_obs[0, 4] * 80.0
  - ego absolute lateral position (m)  = obs[0, 2] * 8.0
  - neighbour relative x (m)           = obs[i, 1] * 200.0   (i >= 1)
  - neighbour relative y (m)           = obs[i, 2] * 8.0     (i >= 1)
  - crash on a step                    = next_obs[0, 0] < 0.5

Lane geometry (verified against the live sim): the ego starts on the
on-ramp / merge lane at y ~= 4.0 m and the mainline sits at y ~= 0.0 m,
so a measured lateral position below ~2.0 m indicates the ego has merged
into the mainline.
"""

import numpy as np

# Verified denormalisation constants (against the live simulator).
_SPEED_MPS = 80.0      # ego forward speed (m/s) = vx_col * 80.0
_LONG_M = 200.0        # relative longitudinal position (m) = col1 * 200.0
_LAT_M = 8.0           # lateral position (m) = col2 * 8.0
_FAR_M = 200.0         # "no neighbour present" -> treat as far (m)
# Verified mainline lane geometry: the left mainline lane sits at y ~= 0 m
# and the right mainline lane at y ~= 4 m (the on-ramp / merge lanes are
# further out at y ~= 8-14.5 m). A measured lateral position below ~2 m
# (the midline between the two mainline lanes) therefore indicates the ego
# is in the left lane; this is used only to label lane membership from a
# measured position, not as a goal.
_LANE_MIDLINE_M = 2.0      # m — boundary between left (y~0) and right (y~4) lanes


def _as_int(action):
    """Coerce a possibly-array discrete action to a plain int."""
    arr = np.asarray(action).reshape(-1)
    if arr.size == 0:
        return 0
    return int(arr[0])


def _crashed_step(next_obs):
    """Whether the env's crash detector fired on this step."""
    return float(next_obs[0, 0]) < 0.5


def _ego_speed_mps(o):
    """True ego forward speed (m/s)."""
    return float(o[0, 3]) * _SPEED_MPS


def _ego_lateral_m(o):
    """Ego absolute lateral position (m)."""
    return float(o[0, 2]) * _LAT_M


def _nearest_neighbour_gap_m(o):
    """Straight-line distance (m) to the nearest present neighbour.

    Considers only present rows (presence > 0.5). Returns _FAR_M (200 m,
    treated as "far") when no neighbour is present.
    """
    best = None
    for r in range(1, o.shape[0]):
        if float(o[r, 0]) <= 0.5:  # padding row
            continue
        dx = float(o[r, 1]) * _LONG_M
        dy = float(o[r, 2]) * _LAT_M
        d = float(np.hypot(dx, dy))
        if best is None or d < best:
            best = d
    return best if best is not None else _FAR_M


class Trajectory2Features:
    feature_names = [
        ("episode_len_steps", "Number of timesteps in the trajectory (caps at 100)."),
        ("crashed", "1.0 if the trajectory ended in a crash, else 0.0."),
        ("mean_speed_mps", "Mean ego forward speed over the trajectory (m/s)."),
        ("min_speed_mps", "Minimum ego forward speed (m/s); low values mean it slowed/stopped."),
        ("max_speed_mps", "Peak ego forward speed reached (m/s)."),
        ("speed_change_mps", "Mean absolute step-to-step change in ego speed (m/s); speed steadiness."),
        ("min_neighbour_gap_m", "Closest straight-line distance to any neighbour over the trajectory (m)."),
        ("mean_neighbour_gap_m", "Mean nearest-neighbour straight-line distance over the trajectory (m)."),
        ("mean_lateral_pos_m", "Mean ego lateral position (m); ~4 = on-ramp lane, ~0 = mainline."),
        ("frac_in_mainline", "Fraction of steps with the ego in the mainline (lateral position < ~2 m)."),
        ("frac_in_left_lane", "Fraction of steps with the ego in the left mainline lane (lateral position near 0 m)."),
        ("num_lane_changes", "Count of LANE_LEFT/LANE_RIGHT actions issued."),
    ]

    def get_traj_features(self, trajectory):
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        speeds = []
        gaps = []
        lats = []
        crashed = 0.0
        lane_changes = 0

        for (obs, act, next_obs) in trajectory:
            o = np.asarray(obs)
            no = np.asarray(next_obs)

            speeds.append(_ego_speed_mps(no))
            gaps.append(_nearest_neighbour_gap_m(o))
            lats.append(_ego_lateral_m(o))

            if _crashed_step(no):
                crashed = 1.0

            if _as_int(act) in (0, 2):  # LANE_LEFT, LANE_RIGHT
                lane_changes += 1

        speeds = np.asarray(speeds, dtype=float)
        gaps = np.asarray(gaps, dtype=float)
        lats = np.asarray(lats, dtype=float)

        if speeds.size >= 2:
            speed_change = float(np.mean(np.abs(np.diff(speeds))))
        else:
            speed_change = 0.0

        frac_in_mainline = float(np.mean(lats < _LANE_MIDLINE_M))
        # Left mainline lane sits at y ~= 0 m; a lateral position below the
        # left/right midline (~2 m) means the ego is in the left lane.
        frac_in_left_lane = float(np.mean(lats < _LANE_MIDLINE_M))

        return [
            float(len(trajectory)),
            crashed,
            float(speeds.mean()),
            float(speeds.min()),
            float(speeds.max()),
            speed_change,
            float(gaps.min()),
            float(gaps.mean()),
            float(lats.mean()),
            frac_in_mainline,
            frac_in_left_lane,
            float(lane_changes),
        ]

    def pretty_print_features(self, features):
        width = max(len(s) for s, _ in self.feature_names)
        for (short, desc), val in zip(self.feature_names, features):
            print(f"  {short:<{width}}  {val:8.2f}   {desc}")
