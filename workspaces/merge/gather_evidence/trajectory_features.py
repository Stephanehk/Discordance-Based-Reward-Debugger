"""Trajectory features for highway-env merge-v0.

Summarises what happened in a trajectory along axes that matter for the
specified reward: how fast the ego went, whether/when it got near or
collided with other vehicles, what maneuvers it took, and how far along
the road it made it before the episode ended.

Conventions match env_config.OBS_DESCRIPTION:
  obs[0]    = ego row, absolute normalised coords; obs[0, 0] is overloaded
              to 0.0 on the step where info["crashed"] was True.
  obs[1..4] = nearest other vehicles, positions/velocities relative to ego.
  vx, vy normalised by reward_speed_range (default span ~10 m/s in
  merge-v0), so a normalised value of 1.0 ≈ 10 m/s.

Action ids (Discrete(5)): 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER.
"""

import numpy as np


# Match designed_reward_fn.SAFE_DISTANCE so the "close steps" feature aligns
# with the threshold the reward actually uses.
SAFE_DISTANCE = 0.10
VERY_CLOSE_DISTANCE = 0.05

# Used when no other vehicles are present in the observation; the
# Kinematics view rarely exceeds ~1.0 in normalised relative coords, so
# 1.0 is "comfortably far".
NO_OTHERS_DISTANCE = 1.0


def _min_other_distance(obs) -> float:
    """Smallest normalised distance to any present non-ego vehicle.

    Rows 1..4 give other-vehicle (x, y) *relative to ego*, so their
    Euclidean norm is the displacement distance.
    """
    others = np.asarray(obs[1:5])
    present = others[:, 0] > 0.5
    if not np.any(present):
        return NO_OTHERS_DISTANCE
    rel = others[present, 1:3]
    return float(np.linalg.norm(rel, axis=1).min())


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("mean_speed",
         "Mean ego forward speed across the trajectory (normalised; 1.0 ≈ 10 m/s)."),
        ("max_speed",
         "Max ego forward speed reached in the trajectory (normalised)."),
        ("frac_top_speed_steps",
         "Fraction of steps with ego forward speed at or above 0.35 (near the top of the observed speed range)."),
        ("crashed",
         "1.0 if any step in the trajectory ended in a collision, else 0.0."),
        ("min_distance_to_other",
         "Minimum normalised distance to any other vehicle, over the trajectory."),
        ("mean_min_distance",
         "Mean over steps of the per-step minimum distance to any other vehicle."),
        ("frac_close_steps",
         "Fraction of steps where the closest other vehicle was within the 0.10 safe-distance threshold."),
        ("frac_very_close_steps",
         "Fraction of steps where the closest other vehicle was within 0.05 (near-collision proximity)."),
        ("n_lane_changes",
         "Number of LANE_LEFT or LANE_RIGHT actions taken."),
        ("n_speed_changes",
         "Number of FASTER or SLOWER actions taken."),
        ("episode_length",
         "Number of steps in the trajectory (shorter often means the episode ended early in a crash)."),
        ("final_x_position",
         "Final ego x position (normalised); higher means the ego made it further down the road segment."),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        # Use next_obs throughout — that's where the post-step state and
        # the crash flag (on obs[0, 0]) live.
        next_speeds = np.array([t[2][0, 3] for t in trajectory], dtype=float)
        crashed_any = any(float(t[2][0, 0]) < 0.5 for t in trajectory)
        min_dists = np.array(
            [_min_other_distance(t[2]) for t in trajectory], dtype=float
        )

        actions = [int(t[1]) for t in trajectory]
        n_lane_changes = sum(1 for a in actions if a in (0, 2))
        n_speed_changes = sum(1 for a in actions if a in (3, 4))

        final_x = float(trajectory[-1][2][0, 1])

        return [
            float(next_speeds.mean()),
            float(next_speeds.max()),
            float((next_speeds >= 0.35).mean()),
            1.0 if crashed_any else 0.0,
            float(min_dists.min()),
            float(min_dists.mean()),
            float((min_dists < SAFE_DISTANCE).mean()),
            float((min_dists < VERY_CLOSE_DISTANCE).mean()),
            float(n_lane_changes),
            float(n_speed_changes),
            float(len(trajectory)),
            final_x,
        ]

    def pretty_print_features(self, features) -> None:
        for (name, desc), val in zip(self.feature_names, features):
            print(f"  {name:<25} {val:>8.2f}   {desc}")
