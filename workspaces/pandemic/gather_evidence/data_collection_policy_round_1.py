"""Round 1 data-collection policies for the pandemic regulator.

These policies probe diverse hypotheses about the designed reward:
- Where is its sweet spot (the approximate greedy maximizer)?
- What does an unrestricted-economy policy look like?
- What does a permanent-lockdown policy look like?
- How does an over-cautious policy compare?
- How does a policy that cycles through every stage compare?

Each policy reads `obs[-1, 0]` — the latest row of the observation —
to make its decision. Index 1 is the regulation stage, indices 3/8
are alpha/delta critical proportions, indices 5/10 are alpha/delta
infected (non-critical) proportions.
"""

import numpy as np


def _row(obs):
    return np.asarray(obs)[-1, 0]


class DataCollectionPolicyRewardMaximizer:
    """Heuristic approximate maximizer of the current designed reward — escalates regulation as critical proportion rises, otherwise stays open."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        row = _row(obs)
        critical = float(row[3]) + float(row[8])
        infected = float(row[5]) + float(row[10])
        # Per-day critical penalty (~200 * critical) dominates the +1 openness bonus
        # once critical > 0.005, but stage-4 itself costs -5/day, so escalate
        # gradually rather than jumping to stage 4.
        if critical >= 0.015 or infected >= 0.08:
            return 3
        if critical >= 0.008 or infected >= 0.04:
            return 2
        if critical >= 0.003 or infected >= 0.015:
            return 1
        return 0


class DataCollectionPolicyAlwaysOpen:
    """Always stage 0 — tests what happens when the economy is never restricted."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 0


class DataCollectionPolicyAlwaysLockdown:
    """Always stage 4 — tests the cost of a permanent strictest-lockdown regime."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 4


class DataCollectionPolicyPanicResponder:
    """Over-cautious reactive — jumps straight to stage 4 at any sign of infection, otherwise stage 0."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        row = _row(obs)
        any_active = (float(row[3]) + float(row[5]) + float(row[8]) + float(row[10])) > 0.0
        return 4 if any_active else 0


class DataCollectionPolicyStageCycler:
    """Cycles deterministically through all five stages — covers every regulation level."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0
        # Ascend 0->1->2->3->4 then descend 3->2->1 and loop. Period 8 days.
        self._sequence = [0, 1, 2, 3, 4, 3, 2, 1]

    def act(self, obs):
        a = self._sequence[self._t % len(self._sequence)]
        self._t += 1
        return a


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyAlwaysOpen,
    DataCollectionPolicyAlwaysLockdown,
    DataCollectionPolicyPanicResponder,
    DataCollectionPolicyStageCycler,
]
