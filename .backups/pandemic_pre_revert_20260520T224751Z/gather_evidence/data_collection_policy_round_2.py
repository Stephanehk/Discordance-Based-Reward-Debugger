"""Round 2 data-collection policies for the Pandemic Simulator.

Round 1 covered the extremes (always-open, always-locked, daily whiplash,
constant-moderate) and a step-toward-target reactive controller. Two
disagreements remained against the *updated* reward (which now adds a convex
high-stage surcharge): the daily-oscillation policy still out-scores stable
policies despite its fluctuation, and stable strong-suppression policies
weren't well sampled. This round adds STABLE suppressors and a graded
controller so the fluctuation axis can be probed at fixed suppression, plus
a maximizer retuned to the current reward.

Observation latest row obs[-1, 0] (see env_config.OBS_DESCRIPTION):
[1] current stage, alpha=[3]critical [5]infected,
delta=[8]critical [10]infected.
"""

import numpy as np


def _row(obs):
    return np.asarray(obs)[-1, 0]


class DataCollectionPolicyRewardMaximizer:
    """Approximately maximizes the *current* reward: suppress with moderate stages, avoid full lockdown.

    Since the updated reward now penalizes high lockdown levels convexly,
    this controller targets moderate stages (2-3) to keep the outbreak under
    control and only escalates to full lockdown when critical load is high,
    stepping one stage at a time to limit fluctuation.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        row = _row(obs)
        cur_stage = int(round(float(row[1])))
        infected = float(row[5]) + float(row[10])
        critical = float(row[3]) + float(row[8])
        if critical > 0.015:
            target = 4
        elif infected > 0.06:
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


class DataCollectionPolicyStableStrongSuppression:
    """Holds a steady strong stage (3) the whole time — strong suppression with zero fluctuation.

    A stable counterpart to round 1's daily oscillator: comparable or better
    outbreak suppression but no day-to-day swings, so it isolates how the
    reward values policy stability versus the new high-stage surcharge.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 3


class DataCollectionPolicyGradedController:
    """Sets the stage directly from the current infection level — graded, can jump between levels.

    Maps the observed active-infection fraction to a stage in one shot
    (no smoothing), producing an intermediate amount of fluctuation tied to
    the outbreak. Probes the region between the smooth reactive controller
    and the daily oscillator on the fluctuation axis.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        row = _row(obs)
        infected = float(row[5]) + float(row[10])
        critical = float(row[3]) + float(row[8])
        if critical > 0.015:
            return 4
        if infected > 0.08:
            return 4
        if infected > 0.04:
            return 3
        if infected > 0.015:
            return 2
        if infected > 0.004:
            return 1
        return 0


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyStableStrongSuppression,
    DataCollectionPolicyGradedController,
]
