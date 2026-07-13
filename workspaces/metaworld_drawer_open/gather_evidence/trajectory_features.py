"""Trajectory features for the Meta-World drawer-open task.

Reward terms in `designed_reward_fn.py` (reward = 5 * (caging + opening),
evaluated fresh at every step and summed over the episode):

  T1 "caging"  — tolerance(|| (handle - hand) * [3,3,1] ||, bounds 0..0.01,
                 margin = same error at episode start). Highest when the hand
                 is at the handle. Purely a *position* term: it says nothing
                 about the fingers, the grip, or whether the drawer moves.
  T2 "opening" — tolerance(|| handle - fully_open_target ||, bounds 0..0.02,
                 margin 0.2 m). Highest when the handle sits at its fully-open
                 pose. Credited per step, so it also pays for *dwelling* near
                 the target rather than only for getting there.

Feature -> term mapping (proxy vs. true outcome kept separate):

  T1 proxy      : mean_hand_to_handle, frac_steps_hand_at_handle
  T1 true outcome: frac_steps_fingers_closed, frac_steps_pulling_handle
                  (the hand can sit on the handle with fingers wide open and
                   the drawer untouched — same T1 credit, no grasp)

  T2 proxy      : mean_handle_to_target, frac_steps_drawer_open
                  (per-step proximity credit; a run that opens early and idles
                   there accrues this without doing anything more)
  T2 true outcome: final_open_frac, max_open_frac, ends_open, steps_to_open,
                  drawer_slipped_back
                  (whether the drawer is actually open, how far, how fast, and
                   whether it stayed open)

  Not in the reward at all: mean_action_magnitude, mean_action_change,
  hand_path_length — control effort, jerkiness, and wandering are free.
"""

import numpy as np

MAX_DIST = 0.2  # meters of handle travel from closed to fully open
SUCCESS_TOL = 0.03  # meters; the task's own "drawer is open" tolerance
NEAR_HANDLE_TOL = 0.03  # meters; hand counts as "at the handle" within this
CLOSED_APERTURE = 0.5  # gripper reading below this = fingers closed (1.0 = wide open)
HANDLE_MOVE_EPS = 1e-3  # meters/step; handle counts as moving above this


def _unpack(next_obs):
    """Pull the physical quantities we need out of one observation."""
    o = np.asarray(next_obs, dtype=np.float64)
    hand = o[0:3]  # end-effector position, meters
    aperture = float(o[3])  # finger separation, 1.0 = wide open
    handle = o[4:7]  # drawer-handle position, meters
    target = o[39:42]  # handle position when fully open, meters
    return hand, aperture, handle, target


def _open_frac(handle, target, start_dist):
    """How far open the drawer is: 0.0 = as shut as it started, 1.0 = fully open."""
    # Remaining handle travel, as a fraction of the travel available at the
    # start of the attempt (the drawer begins shut).
    remaining = float(np.linalg.norm(handle - target))
    return float(np.clip(1.0 - remaining / max(start_dist, 1e-6), 0.0, 1.0))


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("final_open_frac", "how far open the drawer is at the end (0 = shut, 1 = fully open)"),
        ("max_open_frac", "the furthest the drawer was opened at any point in the attempt"),
        ("ends_open", "1 if the drawer is open to within the task's tolerance at the end, else 0"),
        ("steps_to_open", "steps taken to first reach the fully-open position (episode length if never)"),
        ("drawer_slipped_back", "how much of the opening was given back after the furthest point (fraction)"),
        ("frac_steps_drawer_open", "fraction of steps with the drawer at the fully-open position"),
        ("mean_handle_to_target", "average distance from the handle to its fully-open position (m)"),
        ("mean_hand_to_handle", "average distance from the hand to the drawer handle (m)"),
        ("frac_steps_hand_at_handle", "fraction of steps with the hand touching-distance from the handle"),
        ("frac_steps_fingers_closed", "fraction of steps with the fingers closed rather than splayed open"),
        ("frac_steps_pulling_handle", "fraction of steps where the hand is on the handle and the handle is actually moving"),
        ("mean_action_magnitude", "average size of the commanded movement each step (0-1)"),
        ("mean_action_change", "average step-to-step change in the command — how jerky the motion is"),
        ("hand_path_length", "total distance the hand travelled over the attempt (m)"),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        n = len(trajectory)
        if n == 0:
            return [0.0] * len(self.feature_names)

        # Handle-to-target distance at the start of the attempt (drawer shut).
        _, _, handle0, target0 = _unpack(trajectory[0][0])
        start_dist = float(np.linalg.norm(handle0 - target0))

        open_fracs = []
        handle_to_target = []
        hand_to_handle = []
        apertures = []
        pulling = []
        hand_positions = []
        actions = []

        for obs, act, next_obs in trajectory:
            hand, aperture, handle, target = _unpack(next_obs)
            prev_handle = _unpack(obs)[2]

            open_fracs.append(_open_frac(handle, target, start_dist))
            handle_to_target.append(float(np.linalg.norm(handle - target)))
            d_hand = float(np.linalg.norm(handle - hand))
            hand_to_handle.append(d_hand)
            apertures.append(aperture)

            handle_moved = float(np.linalg.norm(handle - prev_handle))
            pulling.append(
                1.0 if (d_hand <= NEAR_HANDLE_TOL and handle_moved > HANDLE_MOVE_EPS) else 0.0
            )

            hand_positions.append(hand)
            actions.append(np.asarray(act, dtype=np.float64).reshape(-1))

        open_fracs = np.asarray(open_fracs)
        handle_to_target = np.asarray(handle_to_target)
        hand_to_handle = np.asarray(hand_to_handle)
        apertures = np.asarray(apertures)
        actions = np.asarray(actions)
        hand_positions = np.asarray(hand_positions)

        # --- T2: is the drawer actually open? ---
        open_steps = handle_to_target <= SUCCESS_TOL
        final_open_frac = float(open_fracs[-1])
        max_open_frac = float(open_fracs.max())
        ends_open = float(open_steps[-1])
        steps_to_open = float(np.argmax(open_steps) + 1) if open_steps.any() else float(n)
        # Opening given back after the high-water mark: pushed shut again / drifted back.
        peak_idx = int(np.argmax(open_fracs))
        slipped = max_open_frac - float(open_fracs[peak_idx:].min())
        drawer_slipped_back = float(np.clip(slipped, 0.0, 1.0))
        frac_steps_drawer_open = float(open_steps.mean())
        mean_handle_to_target = float(handle_to_target.mean())

        # --- T1: hand at the handle, vs. actually gripping and pulling it ---
        mean_hand_to_handle = float(hand_to_handle.mean())
        frac_steps_hand_at_handle = float((hand_to_handle <= NEAR_HANDLE_TOL).mean())
        frac_steps_fingers_closed = float((apertures < CLOSED_APERTURE).mean())
        frac_steps_pulling_handle = float(np.mean(pulling))

        # --- unrewarded: effort, jerk, wandering ---
        mean_action_magnitude = float(np.abs(actions).mean())
        if n > 1:
            mean_action_change = float(np.abs(np.diff(actions, axis=0)).mean())
            hand_path_length = float(
                np.linalg.norm(np.diff(hand_positions, axis=0), axis=1).sum()
            )
        else:
            mean_action_change = 0.0
            hand_path_length = 0.0

        return [
            final_open_frac,
            max_open_frac,
            ends_open,
            steps_to_open,
            drawer_slipped_back,
            frac_steps_drawer_open,
            mean_handle_to_target,
            mean_hand_to_handle,
            frac_steps_hand_at_handle,
            frac_steps_fingers_closed,
            frac_steps_pulling_handle,
            mean_action_magnitude,
            mean_action_change,
            hand_path_length,
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"{name:<{width}}  {value:>8.2f}   {desc}")
