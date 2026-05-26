"""Round-1 data-collection policies for the pandemic-control environment.

Each policy is a deterministic, hand-coded behavior intended to produce a
trajectory that varies along the features in `trajectory_features.py`
(mean_stage, max_stage, frac_days_strict, stage_switches, mean_critical,
peak_critical, ...). Together they spread the evidence across permissive,
strict, oscillating, and reactive regulation regimes so the stakeholder
metrics will have material to disagree on.
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector from an obs of shape (history, 1, 13)."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across both strains (alpha + delta)."""
    return float(state[3]) + float(state[8])


class DataCollectionPolicyDesignedRewardMaximizer:
    """Best-effort maximizer of the current designed reward: stay at stage 0 until critical >= 0.01, then step up to at most stage 3."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        state = _latest(obs)
        critical = _critical(state)
        # The reward penalizes stage>0 when critical<0.01, and penalizes
        # stage==4 always. So default to 0 when critical is low; when
        # critical is at/above the threshold, escalate but cap at 3.
        if critical < 0.01:
            return 0
        # Scale stage with critical above the threshold, capped at 3.
        # critical in [0.01, ~0.04+) maps roughly to stages 1..3.
        excess = critical - 0.01
        stage = 1 + int(min(2, np.floor(excess / 0.01)))
        return int(max(0, min(3, stage)))


class DataCollectionPolicyAlwaysOpen:
    """Permissive baseline: never imposes any restriction; lets the outbreak run freely."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 0


class DataCollectionPolicyAlwaysStrict:
    """Hard-lockdown baseline: imposes the strictest stage every day, suppressing infection at heavy economic and civic cost."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 4


class DataCollectionPolicyWhiplash:
    """Day-by-day whiplash: alternates between fully open and strict lockdown, maximizing stage_switches without a coherent epidemiological plan."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        # Deterministic alternation between stage 0 and stage 4 each day.
        a = 4 if (self._t % 2) else 0
        self._t += 1
        return int(a)


POLICIES = [
    DataCollectionPolicyDesignedRewardMaximizer,
    DataCollectionPolicyAlwaysOpen,
    DataCollectionPolicyAlwaysStrict,
    DataCollectionPolicyWhiplash,
]
