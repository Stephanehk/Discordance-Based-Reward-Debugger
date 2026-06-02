"""Round 2 data-collection policies for the robosuite Wipe workspace.

Round 1 covered four corners of the action-strategy space (full-bore
heuristic wiper, hammered-press exploit, contact-free hover, and
uniform-random exploration). The disagreements that round produced
collapsed onto two dominant axes (force-vs-progress and any-progress-
vs-do-nothing), which means the *intermediate* regions of policy
space — gentler wipers, pulsed contact, lateral drift without
contact — are under-sampled. Round 2 targets those gaps.

The reward function has not been edited since round 1, so the required
designed-reward maximizer is re-included verbatim (its outputs already
exist on disk from round 1, but seeding differences will surface fresh
variance).
"""

from __future__ import annotations

import numpy as np


class DataCollectionPolicyDesignedRewardMaximizer:
    """Hand-coded wiping heuristic: descend, then sweep in a snake pattern.

    Re-included to satisfy the per-round maximizer requirement under the
    *unchanged* designed reward; lets fresh seeds widen the per-policy
    distribution of outcomes already seen in round 1.
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


class DataCollectionPolicyGentleWiper:
    """Reduced-magnitude variant of the wiper — softer descent and sweep.

    Probes the corner of the reward space where the agent attempts the
    task but with much smaller per-step force commands. Should expose
    whether the force-vs-progress trade-off shifts when *neither* term
    is pushed to its extreme — i.e. is there a regime where the reward
    rewards both gentle force AND incremental progress, or does the
    distance-multiplier dominate at low contact pressure?
    """

    DESCEND_STEPS = 40    # slower descent → less aggressive contact
    SWEEP_HALF = 80
    Y_HALF = 10

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            a[2] = -0.5     # half the maximizer's descent commitment
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.SWEEP_HALF)
            a[0] = 0.5 if phase < self.SWEEP_HALF else -0.5
            a[2] = -0.15     # light maintained pressure
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.15
            else:
                a[1] = -0.15
        self._t += 1
        return a


class DataCollectionPolicyPulseWiper:
    """Pulsed contact: alternates between brief press-and-sweep and lift phases.

    Probes whether the reward credits 'bursts' of productive contact
    separated by lift-off periods, or whether sustained contact is
    over-rewarded. Five steps of press + sweep, then five steps of
    upward retreat, repeating — keeps the sustained excess-force
    counter low while still attempting wipes.
    """

    DESCEND_STEPS = 25
    PRESS_LEN = 5
    LIFT_LEN = 5

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
            phase = (t - self.DESCEND_STEPS) % (self.PRESS_LEN + self.LIFT_LEN)
            if phase < self.PRESS_LEN:
                # press into the table and sweep horizontally
                a[2] = -0.7
                a[0] = 1.0 if ((t - self.DESCEND_STEPS) // (self.PRESS_LEN + self.LIFT_LEN)) % 2 == 0 else -1.0
            else:
                # lift off (no contact)
                a[2] = +0.7
        self._t += 1
        return a


class DataCollectionPolicyLateralDrift:
    """Slow lateral drift at the starting height — never makes contact.

    Hover round-1 stayed roughly in place; this policy drifts laterally
    across the workspace at constant elevation, so the wiper traverses
    the airspace above different parts of the dirt field. Probes whether
    the reaching reward (1 − tanh(d)) actually fires from above the
    table or only when contact is imminent — useful for diagnosing
    whether the reward rewards approach without engagement.
    """

    DRIFT_PERIOD = 80   # steps before reversing direction

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        phase = (t // self.DRIFT_PERIOD) % 4
        if phase == 0:
            a[0] = +0.5     # drift right
        elif phase == 1:
            a[1] = +0.5     # drift forward
        elif phase == 2:
            a[0] = -0.5     # drift left
        else:
            a[1] = -0.5     # drift back
        a[2] = +0.05        # tiny upward bias to stay clear of the table
        self._t += 1
        return a


POLICIES = [
    DataCollectionPolicyDesignedRewardMaximizer,
    DataCollectionPolicyGentleWiper,
    DataCollectionPolicyPulseWiper,
    DataCollectionPolicyLateralDrift,
]
