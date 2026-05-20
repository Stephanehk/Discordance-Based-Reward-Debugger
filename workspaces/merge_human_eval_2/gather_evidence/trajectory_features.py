"""Trajectory features for highway-env merge-v0.

Summarises each rollout in physical units so a reward designer can read what
happened. Features mix per-step aggregates (mean/max speed, min gap, fraction
of steps tailgating) with trajectory-level summaries (episode length, crash,
distance travelled, lane changes). Denormalisation constants are the
empirically verified ones for this simulator.
"""

import numpy as np

# Verified denormalisation constants (against the live simulator).
_SPEED_MPS = 80.0     # ego forward speed (m/s) = next_obs[0, 3] * 80.0
_LONG_M = 200.0       # relative longitudinal position (m) = col1 * 200.0
_LAT_M = 8.0          # lateral position (m) = col2 * 8.0
_LANE_MIDLINE_M = 2.0  # lateral midline between the two lanes (~2 m)


def _crashed_step(next_obs):
    return float(next_obs[0, 0]) < 0.5


def _ego_speed_mps(o):
    return float(o[0, 3]) * _SPEED_MPS


def _nearest_lead_gap_m(o):
    """Longitudinal gap (m) to the nearest observed leading vehicle, or NaN.

    Rows 1..4 hold positions relative to the ego. A leading vehicle has a
    positive relative longitudinal coordinate (ahead of the ego). Returns the
    smallest such forward gap among present vehicles.
    """
    gaps = []
    for r in range(1, o.shape[0]):
        present = float(o[r, 0]) > 0.5
        if not present:
            continue
        dx = float(o[r, 1]) * _LONG_M
        if dx > 0.0:
            gaps.append(dx)
    if not gaps:
        return float("nan")
    return min(gaps)


class Trajectory2Features:
    feature_names = [
        ("mean_speed_mps", "Mean ego forward speed over the trajectory (m/s)."),
        ("max_speed_mps", "Peak ego forward speed reached (m/s)."),
        ("min_speed_mps", "Minimum ego forward speed (m/s); low values mean it slowed/stopped."),
        ("speed_std_mps", "Std-dev of ego forward speed (m/s); a proxy for speed steadiness."),
        ("crashed", "1.0 if the trajectory ended in a crash, else 0.0."),
        ("episode_len_steps", "Number of timesteps in the trajectory (caps at 100)."),
        ("distance_travelled_m", "Approximate forward distance covered (m), summing speed * 1 s steps."),
        ("min_lead_gap_m", "Smallest observed forward gap to a leading vehicle (m); large = NaN-filled with 200."),
        ("mean_lead_gap_m", "Mean observed forward gap to the nearest leading vehicle (m)."),
        ("frac_tailgating", "Fraction of steps with a leading vehicle closer than 10 m."),
        ("frac_above_25mps", "Fraction of steps where ego forward speed exceeded 25 m/s."),
        ("num_lane_changes", "Count of LANE_LEFT/LANE_RIGHT actions issued (a proxy for lateral jerk)."),
    ]

    def get_traj_features(self, trajectory):
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        speeds = []
        gaps = []
        tailgate_steps = 0
        above25_steps = 0
        crashed = 0.0
        lane_changes = 0

        for (obs, act, next_obs) in trajectory:
            no = np.asarray(next_obs)
            spd = _ego_speed_mps(no)
            speeds.append(spd)

            if _crashed_step(no):
                crashed = 1.0

            if spd > 25.0:
                above25_steps += 1

            gap = _nearest_lead_gap_m(no)
            if not np.isnan(gap):
                gaps.append(gap)
                if gap < 10.0:
                    tailgate_steps += 1

            a = int(act) if np.ndim(act) == 0 else int(np.asarray(act).flatten()[0])
            if a in (0, 2):  # LANE_LEFT, LANE_RIGHT
                lane_changes += 1

        n = len(trajectory)
        speeds = np.asarray(speeds, dtype=float)

        mean_speed = float(speeds.mean())
        max_speed = float(speeds.max())
        min_speed = float(speeds.min())
        speed_std = float(speeds.std())
        distance = float(speeds.sum())  # 1-second steps -> m/s * 1 s = m

        # For gaps: fill "no leading vehicle observed" with a large gap (200 m,
        # the longitudinal denorm range) so a clear road reads as a big gap.
        if gaps:
            min_gap = float(min(gaps))
            mean_gap = float(np.mean(gaps))
        else:
            min_gap = _LONG_M
            mean_gap = _LONG_M

        frac_tailgating = tailgate_steps / n
        frac_above25 = above25_steps / n

        return [
            mean_speed,
            max_speed,
            min_speed,
            speed_std,
            crashed,
            float(n),
            distance,
            min_gap,
            mean_gap,
            frac_tailgating,
            frac_above25,
            float(lane_changes),
        ]

    def pretty_print_features(self, features):
        width = max(len(s) for s, _ in self.feature_names)
        for (short, desc), val in zip(self.feature_names, features):
            print(f"  {short:<{width}}  {val:8.2f}   {desc}")
