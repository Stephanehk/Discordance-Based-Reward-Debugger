"""Trajectory features for the Meta-World hammer task.

Reward terms in `designed_reward_fn.py`
(reward = (2*reward_grab + 6*reward_in_place) * reward_quat, evaluated fresh at
every step and summed over the episode; overridden to a flat 10 on a step where
the nail is past 9cm AND the shaped reward already exceeds 5):

  T1 "reward_grab"     — gripper caging + squeezing. Pads must bracket the hammer
                         handle in y, the tool-center-point must close in on it in
                         x-z, and the gripper command must be positive. Purely a
                         *pose + command* term: it never checks that the hammer is
                         actually lifted or carried.
  T2 "reward_in_place" — 0.1 * (hammer head is >2cm off the table)
                         + 0.9 * tolerance(||hammer_head - nail_target||, 0..0.02,
                         margin 0.2 m). This is the big one (weight 6). It pays for
                         the hammer head being *near the pose the nail head will
                         occupy once driven in* — not for the nail moving. Credited
                         every step, so parking the head there and idling accrues it.
  T3 "reward_quat"     — multiplier in [0,1] for the hammer lying flat
                         (quaternion near [1,0,0,0]). Scales the whole sum down.
  T4 success override  — nail slide (obs[57]) > 0.09 m gives a flat 10, but only on
                         steps where the shaped sum is already > 5.

Feature -> term mapping (proxy vs. true outcome kept separate):

  T1 proxy       : frac_steps_caging_hammer, frac_steps_gripper_squeezing
                   (the fingers can bracket the handle and squeeze while the hammer
                    never leaves the table — full T1 credit, nothing picked up)
  T1 true outcome: frac_steps_hammer_held, max_hammer_lift

  T2 proxy       : mean_head_to_target, frac_steps_head_at_target
                   (per-step proximity credit; a run that swings the head into the
                    target region and hovers there banks this without ever sinking
                    the nail)
  T2/T4 true out.: final_nail_depth, nail_fully_driven, time_to_drive_nail

  T3             : mean_hammer_flatness

  Cross-cutting  : nail_progress_without_hammer — how much of the nail's travel
                   happened while the hammer head was nowhere near the nail head
                   (i.e. the nail was pushed in by the gripper/arm directly rather
                   than struck with the hammer).

  Not in the reward at all: mean_action_magnitude — control effort is free.
"""

import numpy as np

HEAD_OFFSET = np.array([0.16, 0.06, 0.0])  # meters; hammer head relative to hammer body
NAIL_SUCCESS_DEPTH = 0.09  # meters of nail travel that counts as driven in
CAGE_Y_TOL = 0.03  # meters; pad-to-handle distance in y that counts as bracketing
CAGE_XZ_TOL = 0.03  # meters; tcp-to-handle distance in x-z that counts as closing in
HELD_TOL = 0.07  # meters; tcp-to-hammer distance that counts as the hammer in hand
LIFT_TOL = 0.01  # meters above its starting height for the hammer to count as lifted
AT_TARGET_TOL = 0.05  # meters; hammer head counts as "at the nail target" within this
HAMMER_AT_NAIL_TOL = 0.06  # meters; hammer head counts as "on the nail" within this
NAIL_MOVE_EPS = 1e-4  # meters/step; nail counts as advancing above this


class Trajectory2Features:
    feature_names = [
        ("final_nail_depth", "how far the nail ends up driven into the wall (m)"),
        ("nail_fully_driven", "1 if the nail was driven all the way in, else 0"),
        ("time_to_drive_nail", "fraction of the episode taken to drive the nail in (1.0 = never)"),
        ("mean_head_to_target", "mean distance from the hammer head to the nail's driven-in pose (m)"),
        ("frac_steps_head_at_target", "fraction of steps the hammer head hovers at that pose"),
        ("frac_steps_caging_hammer", "fraction of steps the fingers bracket the hammer handle"),
        ("frac_steps_gripper_squeezing", "fraction of steps the gripper is commanded to squeeze shut"),
        ("frac_steps_hammer_held", "fraction of steps the hammer is actually lifted and in the hand"),
        ("max_hammer_lift", "highest the hammer is raised above where it started (m)"),
        ("mean_hammer_flatness", "how flat the hammer is held, 1 = perfectly level, 0 = fully tipped"),
        ("nail_progress_without_hammer", "nail travel that happened without the hammer head on the nail (m)"),
        ("mean_action_magnitude", "mean size of the arm's per-step movement command"),
    ]

    def get_traj_features(self, trajectory):
        n = len(trajectory)
        if n == 0:
            return [0.0] * len(self.feature_names)

        caging = squeezing = held = at_target = 0
        head_to_target = []
        flatness = []
        action_mag = []
        lifts = []
        nail_off_hammer = 0.0  # meters of nail travel with the hammer head away
        time_to_drive = 1.0  # fraction of the episode; stays 1.0 if never driven

        first = np.asarray(trajectory[0][0], dtype=np.float64)
        prev_nail = float(first[57])
        # The hammer's resting height, read off its actual pose at the first step.
        # (obs[54:57] records where it was *placed*, whose z is 0 rather than the
        # height its body settles at, so it is not a usable baseline for lift.)
        hammer_start_z = float(first[4:7][2])

        for i, (_obs, act, next_obs) in enumerate(trajectory):
            s = np.asarray(next_obs, dtype=np.float64)
            a = np.asarray(act, dtype=np.float64)

            hammer = s[4:7]
            hammer_quat = s[7:11]
            nail_head = s[11:14]
            target = s[39:42]
            tcp = s[45:48]
            left_pad = s[48:51]
            right_pad = s[51:54]
            nail = float(s[57])

            hammer_head = hammer + HEAD_OFFSET

            # --- T1 proxy: pads bracketing the handle, gripper commanded shut ---
            pads_bracket = (
                abs(float(left_pad[1]) - float(hammer[1])) < CAGE_Y_TOL
                and abs(float(right_pad[1]) - float(hammer[1])) < CAGE_Y_TOL
            )
            tcp_close_xz = (
                float(np.linalg.norm(tcp[[0, 2]] - hammer[[0, 2]])) < CAGE_XZ_TOL
            )
            caging += int(pads_bracket and tcp_close_xz)
            squeezing += int(float(a[-1]) > 0.0)

            # --- T1 true outcome: hammer actually off the table and in the hand ---
            lift = float(hammer[2]) - hammer_start_z  # meters above its start height
            lifts.append(lift)
            held += int(
                float(np.linalg.norm(tcp - hammer)) < HELD_TOL and lift > LIFT_TOL
            )

            # --- T2 proxy: hammer head near the nail's driven-in pose ---
            d_head_target = float(np.linalg.norm(hammer_head - target))  # meters
            head_to_target.append(d_head_target)
            at_target += int(d_head_target < AT_TARGET_TOL)

            # --- T3: how flat the hammer is being held ---
            quat_err = float(np.linalg.norm(hammer_quat - np.array([1.0, 0.0, 0.0, 0.0])))
            flatness.append(max(1.0 - quat_err / 0.4, 0.0))

            # --- cross-cutting: nail advanced while the hammer head was elsewhere ---
            d_nail = nail - prev_nail  # meters of nail travel this step
            if d_nail > NAIL_MOVE_EPS:
                hammer_on_nail = (
                    float(np.linalg.norm(hammer_head - nail_head)) < HAMMER_AT_NAIL_TOL
                )
                if not hammer_on_nail:
                    nail_off_hammer += d_nail
            prev_nail = nail

            # --- T4: when the nail first reaches full depth ---
            if nail > NAIL_SUCCESS_DEPTH and time_to_drive == 1.0:
                time_to_drive = (i + 1) / n

            action_mag.append(float(np.linalg.norm(a[:3])))

        final_nail = float(np.asarray(trajectory[-1][2], dtype=np.float64)[57])

        return [
            final_nail,
            float(final_nail > NAIL_SUCCESS_DEPTH),
            time_to_drive,
            float(np.mean(head_to_target)),
            at_target / n,
            caging / n,
            squeezing / n,
            held / n,
            float(max(lifts)),
            float(np.mean(flatness)),
            nail_off_hammer,
            float(np.mean(action_mag)),
        ]

    def pretty_print_features(self, features):
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {value:8.2f}   {desc}")
