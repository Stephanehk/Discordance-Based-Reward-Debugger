"""Round 3 data-collection policies (reward unchanged since rounds 1-2).

Rounds 1-2 showed the designed reward favours fast engine-off plummets and never
reaches a pad landing. This round introduces the one clearly untried behaviour:
a suicide burn — free-fall fast, then a hard late engine burn that arrests the
descent to a genuinely soft near-ground speed. Two thresholds span a softness
gradient (~1 m/s vs ~5 m/s final descent), directly probing the reward's
near-ground deceleration term (T5) and the softness question (H5): does the
reward actually prefer a soft arrival over a hard crash, given both fall fast?

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


class _SuicideBurn:
    """Free-fall (engine off) staying upright, then full-throttle burn below a
    height threshold to arrest the descent to a soft near-ground speed."""

    burn_height = 70.0

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        h = obs[12]
        roll, pitch = _roll_pitch(obs[3:7])
        # Keep upright throughout so the late burn arrests vertical speed cleanly.
        finlet_x = np.clip(-1.0 * pitch, -1.0, 1.0)
        finlet_y = np.clip(-1.0 * roll, -1.0, 1.0)
        if h > self.burn_height:
            ignite, throttle = 0.0, 0.0
        else:
            ignite, throttle = 1.0, 1.0
        return _clip([finlet_x, finlet_y, 0.0, ignite, throttle, 0.0, 0.0],
                     self.action_space)


class DataCollectionPolicySuicideBurnSoft(_SuicideBurn):
    """H5/T5: fast fall then an early-enough hard burn (high threshold) that
    arrests the descent to ~1 m/s — a genuinely soft near-ground arrival that
    still fell fast for most of the descent."""

    burn_height = 80.0


class DataCollectionPolicySuicideBurnLate(_SuicideBurn):
    """H5/T5: fast fall then a very late hard burn (low threshold) that only
    partly arrests the descent (~5 m/s) — an intermediate impact, spanning the
    softness gradient between a full soft landing and a pure plummet."""

    burn_height = 35.0


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicySuicideBurnSoft,
    DataCollectionPolicySuicideBurnLate,
]
