"""Round 1 data-collection policies for the rocket-landing reward.

Each policy targets a term / exploit hypothesis from
.repair_history/_reward_analysis.md. Together they probe the reward's flaws:
a genuine landing attempt (the maximizer), plus decoupling probes that drive a
rewarded proxy high while the true outcome (a soft upright landing) never
happens.

Action layout: [finlet_x, finlet_y, finlet_roll, ignition, throttle, gimbal_x, gimbal_y].
"""

import numpy as np


def _quat_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _roll_pitch(q):
    x, y, z, w = q
    sarg = np.clip(-2.0 * (x * z - w * y), -1.0, 1.0)
    pitch = float(np.arcsin(sarg))
    roll = float(np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)))
    return roll, pitch


def _world_vspeed(obs):
    """World-frame vertical velocity (m/s), positive up."""
    rot = _quat_to_rotmat(obs[3:7])
    return float(rot[2, :] @ obs[7:10])


def _clip(a, space):
    return np.clip(np.asarray(a, dtype=np.float64), space.low, space.high)


class DataCollectionPolicyRewardMaximizer:
    """Maximizes the designed reward: decelerate near the ground, stay over the
    pad, and keep upright for a soft touchdown."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy, h = obs[10], obs[11], obs[12]
        vz = _world_vspeed(obs)
        roll, pitch = _roll_pitch(obs[3:7])

        # Want a gentle descent profile: slower the lower we are.
        target_descent = -min(2.0 + 0.08 * h, 40.0)  # m/s (negative = down)
        # Ignite to arrest the fall once we're getting low or falling fast.
        ignite = 1.0 if (h < 150.0 or vz < -35.0) else 0.0
        throttle = np.clip((target_descent - vz) * 0.05, 0.0, 1.0) if ignite else 0.0

        # Gimbal thrust to null horizontal offset (drive toward the pad).
        gimbal_x = np.clip(-0.05 * ox, -1.0, 1.0)
        gimbal_y = np.clip(-0.05 * oy, -1.0, 1.0)
        # Grid fins to keep upright.
        finlet_x = np.clip(-1.5 * pitch, -1.0, 1.0)
        finlet_y = np.clip(-1.5 * roll, -1.0, 1.0)

        return _clip([finlet_x, finlet_y, 0.0, ignite, throttle, gimbal_x, gimbal_y],
                     self.action_space)


class DataCollectionPolicyHoverOverPad:
    """H1 (T2 stay-over-pad): burns thrust to slow the fall and sit horizontally
    over the pad, farming the stay-over-pad term without ever landing."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy = obs[10], obs[11]
        roll, pitch = _roll_pitch(obs[3:7])
        # Full-throttle burn to resist the descent as long as fuel lasts.
        ignite, throttle = 1.0, 1.0
        # Steer hard to stay directly over the pad.
        gimbal_x = np.clip(-0.15 * ox, -1.0, 1.0)
        gimbal_y = np.clip(-0.15 * oy, -1.0, 1.0)
        finlet_x = np.clip(-1.5 * pitch, -1.0, 1.0)
        finlet_y = np.clip(-1.5 * roll, -1.0, 1.0)
        return _clip([finlet_x, finlet_y, 0.0, ignite, throttle, gimbal_x, gimbal_y],
                     self.action_space)


class DataCollectionPolicyNosedivePlummet:
    """H3/H2 (T4 altitude-loss + T8 flat contact bonus): engine off, plummet
    straight down and slam the pad — fast descent and a hard impact still pay."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy = obs[10], obs[11]
        roll, pitch = _roll_pitch(obs[3:7])
        # No engine — never decelerate. Fins only to keep it pointed down and
        # nudge toward the pad so it actually contacts.
        finlet_x = np.clip(-1.0 * pitch - 0.02 * ox, -1.0, 1.0)
        finlet_y = np.clip(-1.0 * roll - 0.02 * oy, -1.0, 1.0)
        return _clip([finlet_x, finlet_y, 0.0, 0.0, 0.0, 0.0, 0.0],
                     self.action_space)


class DataCollectionPolicyTiltedTranslate:
    """H4 (T3 lateral progress vs T7 tilt): tilts hard to translate toward the
    pad, trading large attitude error for horizontal approach."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy = obs[10], obs[11]
        # Command large fin/gimbal deflections in the direction that leans the
        # rocket toward the pad — deliberately not correcting the resulting tilt.
        sx = -np.sign(ox) if abs(ox) > 0.5 else 0.0
        sy = -np.sign(oy) if abs(oy) > 0.5 else 0.0
        ignite, throttle = 1.0, 0.7  # thrust so the lean produces translation
        return _clip([sx, sy, 0.0, ignite, throttle, 0.6 * sx, 0.6 * sy],
                     self.action_space)


class DataCollectionPolicyThrashControls:
    """H6 (missing jerk/fuel penalty) + state coverage: thrashes every actuator
    at full magnitude, an un-penalized jerky, fuel-wasting descent."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.rng = np.random.default_rng(12345)
        self.t = 0

    def act(self, obs):
        self.t += 1
        low, high = self.action_space.low, self.action_space.high
        # Fresh full-range random action every step -> maximal jerk.
        a = self.rng.uniform(low, high)
        # Flicker the engine on/off to waste fuel erratically.
        a[3] = 1.0 if (self.t % 2 == 0) else 0.0
        return _clip(a, self.action_space)


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyHoverOverPad,
    DataCollectionPolicyNosedivePlummet,
    DataCollectionPolicyTiltedTranslate,
    DataCollectionPolicyThrashControls,
]
