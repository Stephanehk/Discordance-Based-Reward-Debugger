"""Trajectory features for the Meta-World drawer-open task.

Feature -> reward-term mapping
------------------------------
The designed reward sums exactly two per-step *difference* terms, which
telescope over an episode to properties of the endpoints:

  T1  opening term = OPEN_WEIGHT * (openness(next) - openness(obs))
         -> sums to  OPEN_WEIGHT * (openness(final) - openness(initial))
         Since the drawer starts shut, this is essentially the FINAL openness.
  T2  caging term  = CAGE_WEIGHT * (cage(next) - cage(obs))
         -> sums to  CAGE_WEIGHT * (cage(final) - cage(initial)); bounded guidance.

The action is not scored at all.

Features covering T1 (opening):
  - final_openness      : the proxy the opening term actually pays for
  - openness_gain       : final - initial openness (exactly what T1 sums to)
  - max_openness        : peak openness reached (reward only sees the endpoint)
  - openness_backslide  : max - final (opened then let slide/pushed back;
                          invisible to the endpoint reward)
  - drawer_open_success : true outcome -- did the handle actually reach the
                          fully-open target (proxy vs. real success)
  - time_to_first_open  : how quickly it opened (endpoint reward ignores timing)

Features covering T2 (caging):
  - final_hand_at_handle: caging proximity at the end
  - cage_gain           : final - initial caging (exactly what T2 sums to)
  - frac_hand_at_handle : fraction of steps the hand dwells at the handle
                          (reward pays nothing for dwelling)

Features for quantities the reward does NOT score (action / effort):
  - mean_action_magnitude : per-step control effort
  - hand_path_length      : total distance the hand travelled
"""

import numpy as np

MAX_DIST = 0.2  # meters the handle travels from closed to fully open
XY_SCALE = np.array([3.0, 3.0, 1.0])
SUCCESS_REMAINING = 0.03  # handle within 3 cm of fully-open target counts as open
OPEN_THRESH = 0.9  # openness fraction that counts as "open" for timing/dwell


def _openness(o):
    """How far open the drawer is: 0.0 = shut (start), 1.0 = fully open."""
    handle = o[4:7]
    target_pos = o[39:42]
    remaining = float(np.linalg.norm(handle - target_pos))
    return float(np.clip(1.0 - remaining / MAX_DIST, 0.0, 1.0))


def _remaining(o):
    """Meters the handle still is from its fully-open target position."""
    return float(np.linalg.norm(o[4:7] - o[39:42]))


def _hand_at_handle(o):
    """XY-weighted hand-to-handle proximity, 1.0 when essentially touching."""
    gripper = o[0:3]
    handle = o[4:7]
    err = float(np.linalg.norm((handle - gripper) * XY_SCALE))
    # map error 0 -> 1.0, growing error -> 0.0 over the initial-scale span (~0.6 m)
    return float(np.clip(1.0 - err / 0.6, 0.0, 1.0))


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("final_openness", "how far open the drawer ended (0=shut, 1=fully open)"),
        ("openness_gain", "final minus initial openness (what the opening reward sums to)"),
        ("max_openness", "peak openness reached at any point in the attempt"),
        ("openness_backslide", "peak minus final openness (opened then let slide/pushed back)"),
        ("drawer_open_success", "1.0 if the handle actually reached the fully-open target"),
        ("time_to_first_open", "fraction of the episode elapsed before first opening (1.0 = never)"),
        ("final_hand_at_handle", "how close the hand was to the handle at the end (0..1)"),
        ("cage_gain", "final minus initial hand-to-handle proximity (what the guidance reward sums to)"),
        ("frac_hand_at_handle", "fraction of steps the hand dwelt right at the handle"),
        ("mean_action_magnitude", "average size of the arm's per-step movement command"),
        ("hand_path_length", "total distance the hand travelled over the attempt (meters)"),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        if not trajectory:
            return [0.0] * len(self.feature_names)

        n = len(trajectory)
        obs0 = np.asarray(trajectory[0][0], dtype=np.float64)
        obs_last = np.asarray(trajectory[-1][2], dtype=np.float64)

        # Per-step openness / proximity series over the visited states.
        states = [np.asarray(trajectory[0][0], dtype=np.float64)]
        for (_, _, next_obs) in trajectory:
            states.append(np.asarray(next_obs, dtype=np.float64))

        openness_series = np.array([_openness(s) for s in states])
        proximity_series = np.array([_hand_at_handle(s) for s in states])

        final_openness = float(openness_series[-1])
        init_openness = float(openness_series[0])
        openness_gain = final_openness - init_openness
        max_openness = float(openness_series.max())
        openness_backslide = max(0.0, max_openness - final_openness)

        drawer_open_success = 1.0 if _remaining(obs_last) <= SUCCESS_REMAINING else 0.0

        open_steps = np.where(openness_series >= OPEN_THRESH)[0]
        if open_steps.size > 0:
            time_to_first_open = float(open_steps[0]) / max(1, len(openness_series) - 1)
        else:
            time_to_first_open = 1.0

        final_hand_at_handle = float(proximity_series[-1])
        cage_gain = float(proximity_series[-1] - proximity_series[0])
        # "at handle" = proximity essentially saturated
        frac_hand_at_handle = float(np.mean(proximity_series >= 0.95))

        acts = np.array([np.asarray(a, dtype=np.float64) for (_, a, _) in trajectory])
        mean_action_magnitude = float(np.mean(np.linalg.norm(acts, axis=1))) if acts.size else 0.0

        hand_positions = np.array([s[0:3] for s in states])
        hand_path_length = float(np.sum(np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)))

        return [
            final_openness,
            openness_gain,
            max_openness,
            openness_backslide,
            drawer_open_success,
            time_to_first_open,
            final_hand_at_handle,
            cage_gain,
            frac_hand_at_handle,
            mean_action_magnitude,
            hand_path_length,
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {value:8.2f}   {desc}")
