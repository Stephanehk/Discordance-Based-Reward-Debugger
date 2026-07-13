"""Trajectory features for PyFlyt/Rocket-Landing-v4.

These summarize what happened in a rollout so a reward designer can see, at a
glance, how far the rocket got toward a soft upright landing on the pad and how
strongly each term of the designed reward fired.

Feature -> reward-term mapping (designed_reward_fn.py reproduces PyFlyt's dense
reward; terms listed there):
  - stay-over-pad  (+0.3/(horiz_dist+0.1))  and lateral-progress (+10*dh):
        frac_hover_over_pad (PROXY: sitting over the pad without landing)
        vs final_horiz_dist (TRUE outcome: actually over the pad at the end).
  - descend/vertical-progress (+0.2*dh) and decelerate (+4*decel near ground):
        mean_descent_rate (PROXY: descending / slowing) vs
        final_height + touched_pad (TRUE outcome: reached the ground).
  - pad-contact bonus (+5 - 0.3*|vspeed|):
        touched_pad (PROXY: any contact triggers the +5) vs
        impact_vspeed (TRUE outcome: how hard the touchdown was — a crash and a
        soft landing both trip the +5, so these are exposed separately).
  - clean-touchdown bonus (+3 if upright & near-zero velocity):
        clean_landing (TRUE outcome the +3 is meant to credit).
  - spin penalty (-|yaw rate|):        mean_spin.
  - tilt penalty  (-||roll,pitch||):   mean_tilt (sustained) and max_tilt (peak),
        so "briefly tilted" is visibly distinct from "tilted throughout".
  - engine effort (no reward term, but the physical cost of the maneuver):
        engine_on_frac.
"""

import numpy as np


def _quat_to_rotmat(q):
    """Body->world rotation matrix from a PyBullet (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _roll_pitch(q):
    """Roll and pitch (rad) from a PyBullet (x, y, z, w) quaternion."""
    x, y, z, w = q
    sarg = np.clip(-2.0 * (x * z - w * y), -1.0, 1.0)
    pitch = np.arcsin(sarg)
    roll = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    return float(roll), float(pitch)


def _ground_vertical_speed(state):
    """World-frame vertical velocity (m/s), positive up: (R @ body_lin_vel)[2]."""
    rot = _quat_to_rotmat(state[3:7])
    return float(rot[2, :] @ state[7:10])


class Trajectory2Features:
    feature_names = [
        ("final_horiz_dist", "horizontal distance from pad center at the last step (m)"),
        ("final_height", "height above the pad at the last step (m)"),
        ("touched_pad", "1.0 if the rocket ever contacted the landing pad, else 0.0"),
        ("impact_vspeed", "downward speed at first pad contact (m/s); 0 if never touched"),
        ("clean_landing", "1.0 if it ended in an upright, near-zero-velocity touchdown"),
        ("frac_hover_over_pad", "fraction of steps hovering high (>5 m) but within 2 m of the pad"),
        ("mean_descent_rate", "mean downward speed over the trajectory (m/s, positive = descending)"),
        ("mean_tilt", "mean tilt from upright over the trajectory (rad)"),
        ("max_tilt", "peak tilt from upright reached at any step (rad)"),
        ("mean_spin", "mean absolute yaw rate over the trajectory (rad/s)"),
        ("engine_on_frac", "fraction of steps with the main engine ignited"),
    ]

    def get_traj_features(self, trajectory):
        n = len(trajectory)
        if n == 0:
            return [0.0] * len(self.feature_names)

        horiz_dists = []
        descent_rates = []
        tilts = []
        spins = []
        engine_on = 0
        hover_over_pad = 0

        touched_pad = 0.0
        impact_vspeed = 0.0
        clean_landing = 0.0
        contact_seen = False

        for obs, act, next_obs in trajectory:
            prev = np.asarray(obs, dtype=np.float64)
            curr = np.asarray(next_obs, dtype=np.float64)

            horiz = float(np.linalg.norm(curr[10:12]))
            height = float(curr[12])
            horiz_dists.append(horiz)

            gv = _ground_vertical_speed(curr)
            descent_rates.append(-gv)  # positive when moving down

            roll, pitch = _roll_pitch(curr[3:7])
            tilts.append(float(np.hypot(roll, pitch)))
            spins.append(abs(float(curr[2])))  # yaw-rate component of angular velocity

            if curr[24] >= 0.5:  # main-booster ignition state
                engine_on += 1
            if horiz < 2.0 and height > 5.0:
                hover_over_pad += 1

            # First pad contact: record the softness and whether it was a clean landing.
            if not contact_seen and curr[29] >= 0.5:
                contact_seen = True
                touched_pad = 1.0
                impact_vspeed = abs(gv)
                prev_ang_vel = prev[0:3]
                prev_lin_vel = prev[7:10]
                fatal = (
                    np.linalg.norm(prev_ang_vel) > 0.35
                    or np.linalg.norm(prev_lin_vel) > 1.0
                )
                curr_tilt = float(np.hypot(roll, pitch))
                landed = (
                    np.linalg.norm(prev_ang_vel) < 0.02
                    and np.linalg.norm(prev_lin_vel) < 0.02
                    and curr_tilt < 0.1
                )
                if landed and not fatal:
                    clean_landing = 1.0

        last = np.asarray(trajectory[-1][2], dtype=np.float64)
        final_horiz_dist = float(np.linalg.norm(last[10:12]))
        final_height = float(last[12])

        return [
            final_horiz_dist,
            final_height,
            touched_pad,
            impact_vspeed,
            clean_landing,
            hover_over_pad / n,
            float(np.mean(descent_rates)),
            float(np.mean(tilts)),
            float(np.max(tilts)),
            float(np.mean(spins)),
            engine_on / n,
        ]

    def pretty_print_features(self, features):
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {value:8.2f}   {desc}")
