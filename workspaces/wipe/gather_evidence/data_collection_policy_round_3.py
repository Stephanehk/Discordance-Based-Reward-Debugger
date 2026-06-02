"""Round 3 data-collection policies for the robosuite Wipe workspace.

Round 2 surfaced a fresh Goodhart (pulsed proximity-without-progress
trajectories outscoring real cleaning), which the user addressed by
raising the unit-wiped bonus from 50 to 75. Two disagreement axes
remain after that edit:

  - Force-vs-progress at high force: HeavyPress still beats gentler
    wipers when its raw-wipe count is enough larger.
  - Proximity-without-progress on the *reaching* reward: trajectories
    that hover near the dirt centroid (without contacting it) still
    accumulate more reward than a far-away hover, even at zero wipes.

Round 3 includes the required maximizer (now under the U=75 reward)
plus two policies that target those remaining axes head-on.
"""

from __future__ import annotations

import numpy as np


class DataCollectionPolicyDesignedRewardMaximizer:
    """Hand-coded wiping heuristic: descend, then sweep in a snake pattern.

    Re-included to satisfy the per-round maximizer requirement under
    the updated designed reward (UNIT_WIPED_REWARD raised to 75).
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


class DataCollectionPolicyProximityHover:
    """Hovers just above the dirt without ever making contact.

    Designed to maximize the *reaching* reward (the distance-multiplier
    × (1 − tanh(d)) term) while never wiping a marker and never
    exposing the table to force. Stops the descent just shy of the
    table top, then oscillates vertically with tiny amplitude — keeps
    mean_distance very small without ever triggering wipe_contact, the
    excess-force penalty, or a collision. Probes whether the residual
    proximity-Goodhart can still beat genuine policies once the wipe
    bonus is heavier.
    """

    DESCEND_STEPS = 18      # short of the Maximizer's 25 — stop above the table
    OSC_HALF = 5

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            a[2] = -0.7
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.OSC_HALF)
            # Small symmetric vertical oscillation — net no descent.
            a[2] = -0.15 if phase < self.OSC_HALF else +0.15
            # Slow lateral drift to traverse the dirt field while staying clear.
            a[0] = 0.1 * np.sin(t * 0.05)
            a[1] = 0.1 * np.cos(t * 0.05)
        self._t += 1
        return a


class DataCollectionPolicyMediumPress:
    """Intermediate-force wiper between Maximizer and HeavyPress.

    Probes the force-vs-progress axis at a force-budget the prior
    rounds did not visit (somewhere between the Maximizer's ~70 N and
    HeavyPress's ~210 N). The descent is committed (a[2] = −1) but the
    sustained sweep uses a moderate downward command (a[2] = −0.65 vs
    Maximizer's −0.4 and HeavyPress's constant −1.0). Lets the updated
    reward be tested against a third magnitude on the same axis.
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
            a[0] = 0.8 if phase < self.SWEEP_HALF else -0.8
            a[2] = -0.65    # heavier than Maximizer (-0.4), lighter than HeavyPress (-1)
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.25
            else:
                a[1] = -0.25
        self._t += 1
        return a


POLICIES = [
    DataCollectionPolicyDesignedRewardMaximizer,
    DataCollectionPolicyProximityHover,
    DataCollectionPolicyMediumPress,
]
