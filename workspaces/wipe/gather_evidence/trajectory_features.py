"""Trajectory features for the robosuite Wipe workspace.

Reads the reward-component tail that ``env_config._WipeRewardEnv``
appends to every observation (the last 10 float channels — see
``env_config.OBS_DESCRIPTION``) and exposes per-trajectory summaries
that make it easy to see whether the agent actually cleaned the table,
where forces and motion went, and whether the arm punished itself by
colliding with the table or hitting joint limits.
"""

from __future__ import annotations

import numpy as np


# Layout of the appended reward-component channels in obs / next_obs.
# Must stay in sync with ``env_config._WipeRewardEnv._reward_components``.
IDX_TOTAL_WIPED        = -10
IDX_NEW_WIPED          = -9
IDX_MEAN_DISTANCE      = -8
IDX_HAS_CONTACT        = -7
IDX_NCON               = -6
IDX_TOTAL_FORCE_EE     = -5
IDX_MEAN_EE_ACC        = -4
IDX_COLLISION_OR_QLIM  = -3
IDX_NUM_MARKERS        = -2
IDX_HORIZON            = -1


# Force threshold above which the env's default reward applies the
# excess-force penalty (DEFAULT_WIPE_CONFIG["pressure_threshold_max"]).
EXCESS_FORCE_THRESHOLD = 60.0


def _stack_next_obs(trajectory) -> np.ndarray:
    """Return next_obs for each step stacked into a (T, obs_dim) array."""
    return np.stack([np.asarray(t[2], dtype=np.float64) for t in trajectory], axis=0)


def _stack_actions(trajectory) -> np.ndarray:
    return np.stack([np.asarray(t[1], dtype=np.float64) for t in trajectory], axis=0)


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("markers_wiped",
         "total dirt markers cleaned by the end of the episode (out of num_markers)."),
        ("frac_complete",
         "fraction of dirt cleaned by end of episode (markers_wiped / num_markers, in [0,1])."),
        ("episode_length",
         "number of control steps the episode actually ran for."),
        ("first_wipe_step",
         "step index of the first dirt marker wiped (= episode_length if none wiped)."),
        ("frac_steps_on_table",
         "fraction of steps where the wiping pad was touching the table."),
        ("frac_steps_collided",
         "fraction of steps where the arm collided with the table or hit a joint limit."),
        ("frac_steps_excess_force",
         "fraction of steps where end-effector force exceeded the safe upper bound."),
        ("mean_force_when_contact",
         "mean end-effector force (N) over steps when the wiping pad was on the table."),
        ("mean_distance_to_dirt",
         "mean distance from the wiping pad to the remaining-dirt centroid, over the episode."),
        ("mean_ee_acceleration",
         "mean magnitude of end-effector acceleration over the episode (motion smoothness)."),
        ("mean_action_magnitude",
         "mean L2 magnitude of the 6-D pose-delta action over the episode (control effort)."),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        if not trajectory:
            return [0.0] * len(self.feature_names)

        S = _stack_next_obs(trajectory)              # (T, D)
        A = _stack_actions(trajectory)               # (T, A)
        T = S.shape[0]

        total_wiped_end = float(S[-1, IDX_TOTAL_WIPED])
        new_wiped_per_step = S[:, IDX_NEW_WIPED]
        num_markers = float(S[-1, IDX_NUM_MARKERS]) if T > 0 else 1.0
        num_markers = num_markers if num_markers > 0 else 1.0

        frac_complete = total_wiped_end / num_markers

        first_wiped = np.where(new_wiped_per_step > 0.5)[0]
        first_wipe_step = float(first_wiped[0]) if first_wiped.size else float(T)

        contact_mask = S[:, IDX_HAS_CONTACT] > 0.5
        frac_on_table = float(contact_mask.mean())

        collided_mask = S[:, IDX_COLLISION_OR_QLIM] > 0.5
        frac_collided = float(collided_mask.mean())

        force = S[:, IDX_TOTAL_FORCE_EE]
        frac_excess_force = float((force > EXCESS_FORCE_THRESHOLD).mean())
        mean_force_when_contact = float(force[contact_mask].mean()) if contact_mask.any() else 0.0

        mean_distance = float(S[:, IDX_MEAN_DISTANCE].mean())
        mean_ee_acc = float(S[:, IDX_MEAN_EE_ACC].mean())
        mean_action_mag = float(np.linalg.norm(A, axis=1).mean()) if A.size else 0.0

        return [
            total_wiped_end,
            frac_complete,
            float(T),
            first_wipe_step,
            frac_on_table,
            frac_collided,
            frac_excess_force,
            mean_force_when_contact,
            mean_distance,
            mean_ee_acc,
            mean_action_mag,
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {float(value):>8.2f}   {desc}")
