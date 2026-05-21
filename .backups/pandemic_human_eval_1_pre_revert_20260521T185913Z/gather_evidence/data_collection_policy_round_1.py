"""Round-1 data-collection policies for the pandemic-control environment.

Each policy chooses a regulation stage (an integer in 0..4) every simulated
day. The four policies below span deliberately different regimes along the two
axes the later steps care about — how strict the regulation stage is, and how
the policy reacts (or not) to the observed outbreak — so that the designed
reward and the independent stakeholder metrics have many chances to disagree.

The latest length-13 state vector is read as np.asarray(obs)[-1, 0]:
  [1] = current stage; [3]+[8] = combined critical-case proportion;
  [5]+[10] = combined currently-infected proportion. (Index [2] is a meaningless
legacy flag and is never read.)
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector from an obs of shape (history,1,13)."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across both strains (alpha + delta)."""
    return float(state[3] + state[8])


class DataCollectionPolicyMaximizeDesignedReward:
    """Maximizes the current designed reward: always stage 0 (collect the +1 bonus, avoid the stage-4 penalty)."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        # The reward is -100*dead - 10*(stage==4) + 1*(stage==0). Deaths stay
        # ~0, so the death term contributes ~0 regardless. The only levers are
        # the +1 stage-0 bonus and the -10 stage-4 penalty; sitting at stage 0
        # every day collects the bonus on every step and never incurs the
        # penalty, which is the maximal achievable per-step reward.
        return 0


class DataCollectionPolicyProportionalResponse:
    """Reacts to the observed critical-case level, raising the stage proportionally as the outbreak worsens."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        # Critical-proportion thresholds at which to escalate to stages 1..4.
        # Critical cases are a small share of the population, so the cut points
        # are small; below the first cut we stay fully open.
        self._cuts = (0.002, 0.005, 0.01, 0.02)

    def act(self, obs):
        crit = _critical(_latest(obs))
        # Escalate one stage per threshold crossed; cap at 4.
        stage = int(np.sum([crit >= c for c in self._cuts]))
        return min(stage, 4)


class DataCollectionPolicyAlwaysStrict:
    """Holds the strictest lockdown (stage 4) the entire episode, regardless of the outbreak."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        # Constant heavy regulation: suppresses transmission hard but pays the
        # full closure cost (and the designed reward's stage-4 penalty) every
        # day, ignoring how mild or severe the disease actually is.
        return 4


class DataCollectionPolicyOscillating:
    """Rapidly switches stages on a fixed cycle, churning the regulation regardless of the outbreak."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        # Deterministic high-churn cycle over the full stage range; this
        # produces large day-to-day swings independent of disease state.
        self._cycle = (0, 3, 1, 4, 2)
        self._t = 0

    def act(self, obs):
        stage = self._cycle[self._t % len(self._cycle)]
        self._t += 1
        return stage


POLICIES = [
    DataCollectionPolicyMaximizeDesignedReward,
    DataCollectionPolicyProportionalResponse,
    DataCollectionPolicyAlwaysStrict,
    DataCollectionPolicyOscillating,
]
