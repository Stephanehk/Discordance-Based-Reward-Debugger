"""Round 4 data-collection policies for the robosuite Wipe workspace.

After three earlier rounds and one reward edit (UNIT_WIPED_REWARD raised
from 50 to 75), two disagreement axes remain:

  - **Proximity Goodhart.** Trajectories that stay near the dirt
    centroid without ever wiping a marker still outscore real (low-
    yield) cleaning. This round pushes the Goodhart harder with a more
    extreme proximity-hover variant.
  - **Force-vs-progress at the low end.** No policy has yet probed the
    *ultra-low* force regime — sweeping with a downward command of only
    -0.05 to -0.1. Round 1's GentleWiper used -0.15; this round dials it
    further down to expose whether any cleaning is possible at near-
    minimal force.

Includes the required maximizer under the unchanged reward.
"""

from __future__ import annotations

import numpy as np


class DataCollectionPolicyDesignedRewardMaximizer:
    """Hand-coded wiping heuristic: descend, then sweep in a snake pattern.

    Re-included to satisfy the per-round maximizer requirement under
    the current reward (UNIT_WIPED_REWARD = 75).
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


class DataCollectionPolicyExtremeProximity:
    """Pure proximity Goodhart: stops the descent even earlier, then holds.

    Aims to spend as many steps as possible at minimal distance from
    the dirt centroid with absolutely no contact and no horizontal
    motion. Should produce an even cleaner Goodhart than the prior
    ProximityHover (which still oscillated vertically and drifted
    laterally) — a trajectory that wipes zero markers, never touches
    the table, never moves, and yet accumulates ~10–15 reward purely
    from the reaching component over 500 steps.
    """

    DESCEND_STEPS = 15      # shorter descent — stop further above the table

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            a[2] = -0.5
        # After descent: hold perfectly still (all-zero action).
        self._t += 1
        return a


class DataCollectionPolicyUltraGentleSweep:
    """Sweeps at near-minimal downward pressure (a[2] = −0.08).

    Probes the bottom edge of the force-vs-progress axis: can the agent
    actually clean any dirt when the downward command is barely
    perceptible? If yes, the reward should credit this gentle outcome
    handsomely under the maintenance/safety perspectives. If no, the
    reaching-reward stream is what dominates the per-step return for
    almost-no-contact policies.
    """

    DESCEND_STEPS = 40
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
            a[2] = -0.25
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.SWEEP_HALF)
            a[0] = 0.4 if phase < self.SWEEP_HALF else -0.4
            a[2] = -0.08    # near-minimal sustained pressure
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.10
            else:
                a[1] = -0.10
        self._t += 1
        return a


POLICIES = [
    DataCollectionPolicyDesignedRewardMaximizer,
    DataCollectionPolicyExtremeProximity,
    DataCollectionPolicyUltraGentleSweep,
]
