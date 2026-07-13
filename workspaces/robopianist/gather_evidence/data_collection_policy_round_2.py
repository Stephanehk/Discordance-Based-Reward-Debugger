"""Round 2 data-collection policies for RoboPianist — fresh evidence, reward unchanged.

Round 1 (data_collection_policy_round_1.py) established that a still hover, a
static mash, an oscillator, a sustain-holder, and pure-random control all fail
to play the song (musical F1 ~0). To surface NEW disagreements this round tries
to actually *engage the keys in time with the music*, and adds targeted probes
for the two hypotheses round 1 left under-instantiated:

  H2 (no-false-press binary): single-hand pressing depresses fewer keys at once,
      moving toward a "few wrong keys" trajectory to contrast with round 1's mash.
  H3 (fingering proximity proxy): hover the hands just above the keys (near the
      press threshold) so fingertips sit close to their targets without sounding
      keys, probing whether the fingering term can be farmed by proximity alone.

The required maximizer under the current (unchanged) reward is re-included.

All presses use the calibrated fact that moving the hand actuators ~0.30 of the
way from their low to high bound depresses the keys under the hands, while ~0.15
leaves the hands hovering just above them. Timed policies read the song's
per-step key target from the tail of the observation (goal_current[:88]).
"""

import numpy as np

_AUG_DIM = 339


def _bounds(env):
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    low = np.where(np.isfinite(low), low, -1.0)
    high = np.where(np.isfinite(high), high, 1.0)
    return low, high


def _goal_keys(obs):
    """The 88 per-step key targets from the tail of the observation."""
    return np.asarray(obs, dtype=np.float64)[-_AUG_DIM:][0:88]


class DataCollectionPolicyRewardMaximizerNeutralHover:
    """Maximizer (H1): still neutral hover — banks the per-step no-false-press,
    sustain, and forearm credit at near-zero energy while playing no notes; the
    best hand-coded behavior under the current reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        neutral = np.clip(np.zeros_like(self.low), self.low, self.high)
        neutral[-1] = 0.0
        self.neutral = neutral

    def act(self, obs):
        a = np.clip(self.neutral + 0.02 * self.rng.randn(*self.neutral.shape),
                    self.low, self.high)
        a[-1] = 0.0
        return a


class DataCollectionPolicyTimedPressWhenRequired:
    """NEW: press the keys under both hands only on steps the song actually calls
    for a note, and lift during rests. Concentrates presses at the right *times*
    -- the first policy that genuinely attempts the piece -- to test whether the
    reward credits actual (if imperfect) playing over doing nothing."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        self.neutral = np.clip(np.zeros_like(self.low), self.low, self.high)
        self.span = self.high - self.low
        self._press = 0.30    # calibrated: depresses the keys under the hands

    def act(self, obs):
        required = bool((_goal_keys(obs) > 0.5).any())
        frac = self._press if required else 0.0
        a = self.neutral + frac * self.span
        a += 0.02 * self.span * self.rng.randn(*a.shape)
        a = np.clip(a, self.low, self.high)
        a[-1] = 0.0
        return a


class DataCollectionPolicyRightHandTimedPress:
    """NEW / H2: press only the RIGHT hand (the melody hand) when the song calls
    for a note, leaving the left hand in a neutral hover. Depresses fewer keys at
    once than a two-hand mash -- a step toward the 'few wrong keys' trajectory
    round 1 lacked -- and drives the two hands maximally out of balance."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        self.neutral = np.clip(np.zeros_like(self.low), self.low, self.high)
        self.span = self.high - self.low
        self._press = 0.30

    def act(self, obs):
        required = bool((_goal_keys(obs) > 0.5).any())
        a = self.neutral + 0.02 * self.span * self.rng.randn(*self.neutral.shape)
        if required:
            a[0:22] = self.neutral[0:22] + self._press * self.span[0:22]  # right hand only
        a = np.clip(a, self.low, self.high)
        a[-1] = 0.0
        return a


class DataCollectionPolicyHoverJustAboveKeys:
    """H3 (fingering proximity proxy): hold the hands just above the keys (below
    the press threshold) so fingertips sit close to their target keys without
    sounding them -- probes whether the fingering term rewards proximity while
    activation and musical accuracy stay at zero."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.low, self.high = _bounds(env)
        self.rng = np.random.RandomState()
        self.neutral = np.clip(np.zeros_like(self.low), self.low, self.high)
        self.span = self.high - self.low
        self._hover = 0.15    # calibrated: hands descend near, but do not press

    def act(self, obs):
        a = self.neutral + self._hover * self.span
        a += 0.015 * self.span * self.rng.randn(*a.shape)
        a = np.clip(a, self.low, self.high)
        a[-1] = 0.0
        return a


POLICIES = [
    DataCollectionPolicyRewardMaximizerNeutralHover,
    DataCollectionPolicyTimedPressWhenRequired,
    DataCollectionPolicyRightHandTimedPress,
    DataCollectionPolicyHoverJustAboveKeys,
]
