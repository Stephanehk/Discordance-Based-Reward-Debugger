"""Round 1 data-collection policies for the Pandemic Simulator.

Broad-coverage round. Each policy chooses a daily regulation stage (0=open
.. 4=full lockdown). Together they exercise every term of the designed
reward (deaths, critical-case load, lockdown level, day-to-day fluctuation)
and span the stakeholder perspectives the detector will compare against.

The observation's latest row obs[-1, 0] is the length-13 state vector
(see env_config.OBS_DESCRIPTION): [1] current stage,
alpha=[3]critical [5]infected, delta=[8]critical [10]infected.
"""

import numpy as np


def _row(obs):
    return np.asarray(obs)[-1, 0]


class DataCollectionPolicyRewardMaximizer:
    """Approximately maximizes the designed reward: suppress when severe, relax when safe.

    Reactive controller that raises restrictions as infections/critical load
    climb and relaxes them once the outbreak recedes, stepping one stage at a
    time to keep the fluctuation penalty small. Stands in for the behavior the
    current designed_reward_fn most rewards.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        row = _row(obs)
        cur_stage = int(round(float(row[1])))
        infected = float(row[5]) + float(row[10])
        critical = float(row[3]) + float(row[8])
        if critical > 0.01:
            target = 4
        elif infected > 0.05:
            target = 3
        elif infected > 0.02:
            target = 2
        elif infected > 0.005:
            target = 1
        else:
            target = 0
        if target > cur_stage:
            return min(cur_stage + 1, 4)
        if target < cur_stage:
            return max(cur_stage - 1, 0)
        return cur_stage


class DataCollectionPolicyLaissezFaire:
    """No restrictions ever (stage 0) — best for the business stakeholder, lets the outbreak run.

    Stresses the deaths and critical-case terms of the designed reward while
    scoring maximally on the economy metric.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 0


class DataCollectionPolicyPermanentLockdown:
    """Full lockdown ever (stage 4) — best for health stakeholders, worst for the economy.

    Stresses the lockdown-level (stage) term of the designed reward while
    minimizing infections and deaths.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 4


class DataCollectionPolicyOscillator:
    """Whipsaws between fully open and full lockdown every day — maximal fluctuation.

    Stresses the day-to-day stage-change (smoothness) term of the designed
    reward and the policy-stability stakeholder.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        self._t += 1
        return 4 if (self._t % 2 == 0) else 0


class DataCollectionPolicyModerateConstant:
    """Holds a steady moderate stage (2) regardless of conditions — stable middle ground.

    A low-fluctuation baseline that neither fully suppresses nor fully
    reopens; useful contrast against the reactive and extreme policies.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 2


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyLaissezFaire,
    DataCollectionPolicyPermanentLockdown,
    DataCollectionPolicyOscillator,
    DataCollectionPolicyModerateConstant,
]
