"""Round 1 data-collection policies for RoboPianist.

Broad-coverage round. Each policy targets a term/hypothesis from
.repair_history/_reward_analysis.md. Actions are 45-dim target joint positions
(two Shadow hands) + a sustain command in [0, 1]; bounds come from the env's
action_space. Policies use an unseeded RandomState so repeated rollouts diverge
(distinct env seeds + distinct action noise).

Hypothesis coverage:
  RewardMaximizerNeutralHover  -> maximizer + H1 (rank inversion), H3, H6-avoid
  MaxEffortSlam                -> H2 (many false presses), H5 (energy), H6 (collide)
  SustainAlwaysOn              -> H4 (sustain decoupled from the song)
  ExtremeOscillator            -> H5 (violent jerk), state-space coverage
  RandomExploration            -> broad coverage; occasional real/false presses
"""

import numpy as np


def _bounds(env):
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    # Guard against unbounded pedal dims etc.
    low = np.where(np.isfinite(low), low, -1.0)
    high = np.where(np.isfinite(high), high, 1.0)
    return low, high


class DataCollectionPolicyRewardMaximizerNeutralHover:
    """Maximizer (H1/H3/H6): hold a still neutral hover — no wrong keys, no
    collision, minimal energy — banking the per-step no-false-press, sustain,
    forearm, and fingering-proximity credit while playing no notes."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        # Neutral target: 0 rad for hand joints (near the reset hover pose),
        # clipped into bounds; sustain off.
        neutral = np.clip(np.zeros_like(self.low), self.low, self.high)
        neutral[-1] = 0.0
        self.neutral = neutral

    def act(self, obs):
        # Tiny jitter so repeated rollouts are not byte-identical, but the hands
        # stay essentially still and off the keys.
        jitter = 0.02 * self.rng.randn(*self.neutral.shape)
        a = np.clip(self.neutral + jitter, self.low, self.high)
        a[-1] = 0.0
        return a


class DataCollectionPolicyMaxEffortSlam:
    """H2/H5/H6: drive every actuator toward its high bound — hands mash down and
    inward, pressing many wrong keys at once (binary no-false-press penalty is the
    same as one), high actuator power, and forearm-collision risk."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()

    def act(self, obs):
        a = self.high.copy()
        a += 0.05 * (self.high - self.low) * self.rng.randn(*a.shape)
        a = np.clip(a, self.low, self.high)
        a[-1] = 0.0  # pedal off; isolate the key/energy/collision effects
        return a


class DataCollectionPolicySustainAlwaysOn:
    """H4: hold the sustain pedal fully down every step regardless of the song,
    with hands in a neutral hover — decouples pedal use from the sustain target."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        neutral = np.clip(np.zeros_like(self.low), self.low, self.high)
        self.neutral = neutral

    def act(self, obs):
        a = np.clip(self.neutral + 0.02 * self.rng.randn(*self.neutral.shape),
                    self.low, self.high)
        a[-1] = 1.0  # pedal fully engaged every step
        return a


class DataCollectionPolicyExtremeOscillator:
    """H5: flip every actuator between its low and high bound each step — violent,
    high-jerk, unplayable motion that the tiny energy penalty barely discourages."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        self._phase = int(self.rng.randint(0, 2))
        self._t = 0

    def act(self, obs):
        target = self.high if (self._t + self._phase) % 2 == 0 else self.low
        a = target.copy()
        a += 0.05 * (self.high - self.low) * self.rng.randn(*a.shape)
        a = np.clip(a, self.low, self.high)
        a[-1] = float((self._t + self._phase) % 2)  # pedal also oscillates
        self._t += 1
        return a


class DataCollectionPolicyRandomExploration:
    """Broad state-space coverage: uniform-random actions across the full joint
    and pedal space — occasionally strikes real and wrong keys, mixed energy."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()

    def act(self, obs):
        return self.rng.uniform(self.low, self.high)


POLICIES = [
    DataCollectionPolicyRewardMaximizerNeutralHover,
    DataCollectionPolicyMaxEffortSlam,
    DataCollectionPolicySustainAlwaysOn,
    DataCollectionPolicyExtremeOscillator,
    DataCollectionPolicyRandomExploration,
]
