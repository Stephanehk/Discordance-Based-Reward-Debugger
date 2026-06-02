"""Round 5 data-collection policies for the robosuite Wipe workspace.

The reward has just been edited a second time: UNIT_WIPED_REWARD went
from 75 to 600. The wipe-bonus now dwarfs every other reward component
(reaching, contact, force-penalty, smoothness), so the proximity
Goodhart is expected to be dead and the dominant question is now:
*at this very high wipe weight, does the reward still distinguish
gentle from aggressive cleaning when both clear similar dirt counts?*

Round 5 keeps the required maximizer and adds two contrastive
policies — one that maximises raw throughput at the cost of force
exposure, and one that maximises gentleness at the cost of speed.
"""

from __future__ import annotations

import numpy as np


class DataCollectionPolicyDesignedRewardMaximizer:
    """Hand-coded wiping heuristic: descend, then sweep in a snake pattern.

    Re-included to satisfy the per-round maximizer requirement under
    the current reward (UNIT_WIPED_REWARD = 600).
    """

    DESCEND_STEPS = 25
    SWEEP_HALF = 60
    Y_HALF = 8

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            a[2] = -1.0
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.SWEEP_HALF)
            a[0] = 1.0 if phase < self.SWEEP_HALF else -1.0
            a[2] = -0.4
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.3
            else:
                a[1] = -0.3
        self._t += 1
        return a


class DataCollectionPolicyAggressiveSweep:
    """Full-force descent with a fast sweep — maximizes throughput at any cost.

    Drives the wiper into the table at full magnitude, then sweeps
    sideways with maximum-amplitude commands. Trades episode length
    (the env will early-terminate on excess force) for raw markers
    cleared. Under the new reward weighting, an aggressive policy may
    score higher than a careful one *if* it can wipe more markers
    before truncation. Probes that boundary.
    """

    DESCEND_STEPS = 20
    SWEEP_HALF = 40
    Y_HALF = 6

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            a[2] = -1.0
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.SWEEP_HALF)
            a[0] = 1.0 if phase < self.SWEEP_HALF else -1.0
            a[2] = -0.85    # heavier than Maximizer (-0.4), lighter than full -1
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.5
            else:
                a[1] = -0.5
        self._t += 1
        return a


class DataCollectionPolicyPatientWiper:
    """Slow, gentle wiper that trades episode length for force budget.

    Half-magnitude descent and a slow sweep that keeps the wiper on
    the table at minimal force. The bet is that under the new reward,
    if this can complete a comparable number of wipes to the Maximizer
    while never triggering the force penalty, the maintenance
    perspective will still distinguish it sharply — even though the
    reward edit may have collapsed that distinction.
    """

    DESCEND_STEPS = 60
    SWEEP_HALF = 100
    Y_HALF = 14

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            a[2] = -0.4
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.SWEEP_HALF)
            a[0] = 0.35 if phase < self.SWEEP_HALF else -0.35
            a[2] = -0.12    # very light sustained pressure
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.10
            else:
                a[1] = -0.10
        self._t += 1
        return a


POLICIES = [
    DataCollectionPolicyDesignedRewardMaximizer,
    DataCollectionPolicyAggressiveSweep,
    DataCollectionPolicyPatientWiper,
]
