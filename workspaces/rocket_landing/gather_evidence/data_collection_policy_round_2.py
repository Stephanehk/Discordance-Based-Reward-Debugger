"""Round 2 data-collection policies.

Round 1 showed the designed reward's favorites are the fast plummet-crashes
(they minimise per-step tilt/loiter penalties and rack up the altitude-loss
term), while nothing ever reaches the pad-contact bonus. This round sharpens
that: a genuinely soft, engine-controlled descent (the good behaviour the reward
under-credits), the reward maximizer under the current reward, and a best-effort
centered plummet aimed at the pad (the crash-on-pad probe, H2).

Action layout: [finlet_x, finlet_y, finlet_roll, ignition, throttle, gimbal_x, gimbal_y].
"""

import numpy as np


def _rmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _wvel(o):
    """World-frame velocity (m/s); [2] is vertical, positive up."""
    return _rmat(o[3:7]) @ o[7:10]


def _roll_pitch(q):
    x, y, z, w = q
    s = np.clip(-2.0 * (x * z - w * y), -1.0, 1.0)
    return (float(np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))),
            float(np.arcsin(s)))


def _clip(a, space):
    return np.clip(np.asarray(a, dtype=np.float64), space.low, space.high)


class DataCollectionPolicyRewardMaximizer:
    """Maximizes the designed reward: decelerate near the ground, drive toward
    the pad, and stay upright for as soft a touchdown as it can manage."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy, h = obs[10], obs[11], obs[12]
        wv = _wvel(obs)
        roll, pitch = _roll_pitch(obs[3:7])
        target_descent = -min(2.0 + 0.08 * h, 40.0)
        ignite = 1.0 if (h < 150.0 or wv[2] < -35.0) else 0.0
        throttle = np.clip((target_descent - wv[2]) * 0.05, 0.0, 1.0) if ignite else 0.0
        gimbal_x = np.clip(-0.05 * ox, -1.0, 1.0)
        gimbal_y = np.clip(-0.05 * oy, -1.0, 1.0)
        finlet_x = np.clip(-1.5 * pitch, -1.0, 1.0)
        finlet_y = np.clip(-1.5 * roll, -1.0, 1.0)
        return _clip([finlet_x, finlet_y, 0.0, ignite, throttle, gimbal_x, gimbal_y],
                     self.action_space)


class DataCollectionPolicySoftControlledDescent:
    """The good behaviour the reward under-credits: an engine-controlled, gentle
    descent (~5 m/s touchdown) — soft and low-impact, yet the shaping terms score
    it far below a fast crash. Contrasts a careful landing against a plummet."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy, h = obs[10], obs[11], obs[12]
        wv = _wvel(obs)
        # Aerodynamic finlet steering to bleed off horizontal velocity gently.
        finlet_x = np.clip(0.05 * ox + 0.2 * wv[0], -1.0, 1.0)
        finlet_y = np.clip(0.05 * oy + 0.2 * wv[1], -1.0, 1.0)
        # Engine feedback holds a slow descent profile -> soft touchdown.
        target_descent = -min(3.0 + 0.06 * h, 30.0)
        ignite = 1.0 if (h < 120.0 or wv[2] < -35.0) else 0.0
        throttle = np.clip((target_descent - wv[2]) * 0.06, 0.0, 1.0) if ignite else 0.0
        return _clip([finlet_x, finlet_y, 0.0, ignite, throttle, 0.0, 0.0],
                     self.action_space)


class DataCollectionPolicyCenteredPlummet:
    """H2 (T8 flat contact bonus): a near-ballistic fast fall with only light
    centering — the regime that stays closest to the pad — aiming to slam the pad
    at high speed, probing whether a hard crash onto it still pays."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        ox, oy = obs[10], obs[11]
        wv = _wvel(obs)
        # Very light finlet centering — enough to nudge toward the pad without
        # tumbling (aggressive steering was shown to increase drift). No engine.
        finlet_x = np.clip(0.01 * ox + 0.03 * wv[0], -1.0, 1.0)
        finlet_y = np.clip(0.01 * oy + 0.03 * wv[1], -1.0, 1.0)
        return _clip([finlet_x, finlet_y, 0.0, 0.0, 0.0, 0.0, 0.0],
                     self.action_space)


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicySoftControlledDescent,
    DataCollectionPolicyCenteredPlummet,
]
