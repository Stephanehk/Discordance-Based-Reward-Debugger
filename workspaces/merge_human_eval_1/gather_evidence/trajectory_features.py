"""Trajectory features for the highway `merge-v0` task.

These features summarize what happened in a trajectory in the physical
units the reward-design criteria are stated in (m/s for speed, metres for
inter-vehicle distance), so the disagreement between the speed-only
designed reward and other stakeholder views is legible.

Denormalization (verified empirically against the live simulator's
ground-truth vehicle state):
  - ego forward speed (m/s) = next_obs[0, 3] * VX_HALFWIDTH_MPS
  - neighbour relative position (m): x = col1 * X_HALFWIDTH_M,
                                      y = col2 * Y_HALFWIDTH_M
The env's Kinematics observation lists only the closest vehicles *ahead*
of the ego, so the gap features describe the nearest observed leading
vehicle — the same signal the reward designer has access to.
"""

import numpy as np

# highway-env Kinematics features_range for merge-v0 (MAX_SPEED=40):
VX_HALFWIDTH_MPS = 80.0  # m/s — normalised vx in [-1, 1] maps back via *80
X_HALFWIDTH_M = 200.0    # m — normalised longitudinal position half-range
Y_HALFWIDTH_M = 8.0      # m — normalised lateral position half-range

# Action ids (DiscreteMetaAction): 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT,
# 3 FASTER, 4 SLOWER. Lane changes are 0 and 2.
_LANE_CHANGE_ACTIONS = (0, 2)

# Reward-design targets, used only as feature thresholds (not as reward):
_TARGET_SPEED_MPS = 25.0   # m/s — criteria's average-speed target
_MIN_SAFE_GAP_M = 10.0     # m — criteria's minimum inter-vehicle distance
_SPEED_BAND_LO = 20.0      # m/s — lower edge of the acceptable speed band
_SPEED_BAND_HI = 30.0      # m/s — upper edge (env reward_speed_range high)


def _ego_speed_mps(obs):
    """Ego forward speed in m/s from a single observation matrix."""
    return float(np.asarray(obs)[0, 3]) * VX_HALFWIDTH_MPS


def _nearest_gap_m(obs):
    """Distance (m) to the nearest observed leading vehicle in one obs.

    Rows 1..4 hold the closest other vehicles in coordinates relative to
    the ego; col 0 is presence. Returns X_HALFWIDTH_M (treated as "far")
    when no neighbour is present.
    """
    obs = np.asarray(obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:  # present (real, non-padding) neighbour
            rx = obs[i, 1] * X_HALFWIDTH_M  # m, relative longitudinal
            ry = obs[i, 2] * Y_HALFWIDTH_M  # m, relative lateral
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else X_HALFWIDTH_M


def _crashed_step(next_obs):
    """True if the env's crash detector fired on this step (ego presence 0)."""
    return float(np.asarray(next_obs)[0, 0]) < 0.5


class Trajectory2Features:
    feature_names = [
        ("mean_speed", "Mean ego forward speed over the trajectory (m/s)"),
        ("max_speed", "Maximum ego forward speed reached (m/s)"),
        ("frac_speed_over_25", "Fraction of steps with ego speed above the 25 m/s target"),
        ("frac_speed_under_20", "Fraction of steps with ego speed below 20 m/s"),
        ("min_gap", "Smallest distance to the nearest observed leading vehicle (m)"),
        ("mean_gap", "Mean distance to the nearest observed leading vehicle (m)"),
        ("frac_gap_under_10", "Fraction of steps with the nearest vehicle within 10 m"),
        ("frac_lane_change", "Fraction of steps requesting a lane change"),
        ("crashed", "1.0 if the trajectory ended in a collision, else 0.0"),
        ("n_steps", "Number of steps in the trajectory (short = early termination)"),
    ]

    def get_traj_features(self, trajectory):
        if not trajectory:
            return [0.0] * len(self.feature_names)

        speeds = []          # m/s, per step (from next_obs)
        gaps = []            # m, nearest observed leading vehicle per step
        lane_changes = 0     # count of lane-change action requests
        crashed = 0.0
        for obs, act, next_obs in trajectory:
            speeds.append(_ego_speed_mps(next_obs))
            gaps.append(_nearest_gap_m(next_obs))
            a = int(np.asarray(act).item()) if np.ndim(act) else int(act)
            if a in _LANE_CHANGE_ACTIONS:
                lane_changes += 1
            if _crashed_step(next_obs):
                crashed = 1.0

        speeds = np.asarray(speeds)
        gaps = np.asarray(gaps)
        n = len(trajectory)

        return [
            float(speeds.mean()),                                  # mean_speed
            float(speeds.max()),                                   # max_speed
            float((speeds > _TARGET_SPEED_MPS).mean()),            # frac_speed_over_25
            float((speeds < _SPEED_BAND_LO).mean()),               # frac_speed_under_20
            float(gaps.min()),                                     # min_gap
            float(gaps.mean()),                                    # mean_gap
            float((gaps < _MIN_SAFE_GAP_M).mean()),                # frac_gap_under_10
            float(lane_changes / n),                               # frac_lane_change
            crashed,                                               # crashed
            float(n),                                              # n_steps
        ]

    def pretty_print_features(self, features):
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {value:8.2f}   {desc}")
