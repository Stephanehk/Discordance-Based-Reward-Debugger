import numpy as np


class Trajectory2Features:
    """Summarize a merge-v0 highway trajectory into legible driving statistics.

    Trajectory is a list of (obs, act, next_obs) tuples. obs/next_obs are
    Kinematics matrices of shape (5, 5): row 0 is the ego (absolute), rows
    1..4 are the nearest leading vehicles in ego-relative coordinates.
    """

    # Denormalization constants (empirically validated against the simulator).
    SPEED_SCALE = 80.0   # next_obs[0, 3] * 80.0 -> ego forward speed (m/s)
    X_SCALE = 200.0      # neighbour relative x (m)
    Y_SCALE = 8.0        # neighbour relative y (m)
    FAR_GAP = 200.0      # gap used when no neighbour is present (m)

    # Discrete actions: 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER.
    LANE_CHANGE_ACTIONS = (0, 2)

    feature_names: list[tuple[str, str]] = [
        ("mean_speed", "Mean ego forward speed over the trajectory (m/s)."),
        ("max_speed", "Maximum ego forward speed reached (m/s)."),
        ("min_speed", "Minimum ego forward speed reached (m/s)."),
        ("min_gap", "Smallest distance to the nearest observed leading vehicle (m)."),
        ("mean_gap", "Mean distance to the nearest observed leading vehicle (m)."),
        ("lane_change_frac", "Fraction of steps where the ego commanded a lane change."),
        ("faster_frac", "Fraction of steps where the ego commanded FASTER (accelerate)."),
        ("crashed", "1.0 if the trajectory ended in a collision, else 0.0."),
        ("episode_length", "Number of steps in the trajectory."),
    ]

    def _ego_speed(self, frame):
        """Ego forward speed in m/s from a (5, 5) Kinematics frame."""
        return float(frame[0, 3]) * self.SPEED_SCALE

    def _nearest_gap(self, obs):
        """Distance (m) to the nearest present leading vehicle in obs."""
        gaps = []
        for i in range(1, obs.shape[0]):
            if float(obs[i, 0]) > 0.5:  # presence
                x_m = float(obs[i, 1]) * self.X_SCALE
                y_m = float(obs[i, 2]) * self.Y_SCALE
                gaps.append(float(np.hypot(x_m, y_m)))
        if not gaps:
            return self.FAR_GAP
        return min(gaps)

    def get_traj_features(self, trajectory) -> list[float]:
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        speeds = []
        gaps = []
        lane_changes = 0
        faster = 0
        crashed = 0.0

        for obs, act, next_obs in trajectory:
            obs = np.asarray(obs, dtype=float)
            next_obs = np.asarray(next_obs, dtype=float)

            speeds.append(self._ego_speed(next_obs))
            gaps.append(self._nearest_gap(obs))

            a = int(np.asarray(act).flatten()[0]) if np.ndim(act) else int(act)
            if a in self.LANE_CHANGE_ACTIONS:
                lane_changes += 1
            if a == 3:
                faster += 1

            if float(next_obs[0, 0]) < 0.5:  # crash on this step
                crashed = 1.0

        n = len(trajectory)
        speeds = np.asarray(speeds, dtype=float)
        gaps = np.asarray(gaps, dtype=float)

        return [
            float(np.mean(speeds)),
            float(np.max(speeds)),
            float(np.min(speeds)),
            float(np.min(gaps)),
            float(np.mean(gaps)),
            float(lane_changes) / n,
            float(faster) / n,
            float(crashed),
            float(n),
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"{name:<{width}}  {value:8.2f}   {desc}")
