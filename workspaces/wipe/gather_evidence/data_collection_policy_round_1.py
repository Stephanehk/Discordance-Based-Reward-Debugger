"""Round 1 data-collection policies for the robosuite Wipe workspace.

Round 1 is a broad first pass: one heuristic that genuinely tries to
maximize the env's default shaped reward (= the designed reward), plus
three contrastive hand-coded policies that stress particular reward
terms or the unrewarded margins of the env's behaviour. Together they
should populate every reward component with at least one trajectory
that fires it.

Action layout (env_config.ACTION_DESCRIPTION):
  a[0..2] = (dx, dy, dz) end-effector position delta in [-1, 1]
  a[3..5] = (drx, dry, drz) end-effector axis-angle rotation delta in [-1, 1]
"""

from __future__ import annotations

import numpy as np


class DataCollectionPolicyDesignedRewardMaximizer:
    """Hand-coded wiping heuristic: descend onto the table, then sweep in a
    snake pattern. This is the round's required reward-maximizing policy —
    a heuristic that tries to score well under the env's default shaped
    reward by actually making contact and clearing dirt.
    """

    DESCEND_STEPS = 25     # control ticks spent pressing down to make contact
    SWEEP_HALF = 60        # half-period of the lateral sweep (steps)
    Y_HALF = 8             # how often to bump in y while sweeping in x

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._t = 0

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        t = self._t
        if t < self.DESCEND_STEPS:
            # Drive end-effector straight down toward the table top.
            a[2] = -1.0
        else:
            phase = (t - self.DESCEND_STEPS) % (2 * self.SWEEP_HALF)
            # Sweep in x: positive for the first half, negative for the second.
            a[0] = 1.0 if phase < self.SWEEP_HALF else -1.0
            # Maintain downward pressure so the wiper stays on the table.
            a[2] = -0.4
            # Every Y_HALF steps, nudge in y to shift the wipe line.
            if ((t - self.DESCEND_STEPS) // self.Y_HALF) % 2 == 0:
                a[1] = 0.3
            else:
                a[1] = -0.3
        self._t += 1
        return a


class DataCollectionPolicyHeavyPress:
    """Pushes the arm straight down at full magnitude for the whole episode.

    Stresses the excess-force penalty and the arm-collision / joint-limit
    branch of the reward — both of which only fire when the agent over-
    commits vertically. Likely to score very negatively under the designed
    reward; useful for surfacing whether those penalties are correctly
    calibrated.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        a = np.zeros(self.action_space.shape, dtype=np.float32)
        a[2] = -1.0  # maximum downward command every step
        return a


class DataCollectionPolicyHover:
    """Keep the arm hovering above the table with tiny jitter.

    Never makes contact, never wipes, never collides. Acts as a do-nothing
    baseline that exposes how much reward the agent gets purely from the
    reaching-reward (1 - tanh(...)) and zero-action penalties — a useful
    contrast against policies that actually engage the table.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._rng = np.random.default_rng(7)

    def act(self, obs):
        # Small upward bias so the arm stays clear of the table, plus
        # gentle jitter to expose any per-step reward fluctuations.
        a = self._rng.uniform(-0.05, 0.05, size=self.action_space.shape).astype(np.float32)
        a[2] = 0.1
        return a


class DataCollectionPolicyRandomExplore:
    """Uniform-random actions across the full 6-D pose-delta range.

    Provides broad coverage of the state and action space without any
    task-oriented bias; serves as a foil to the designed-reward-
    maximizing heuristic and surfaces unintended reward gradients in
    state regions a goal-directed policy would never visit.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._rng = np.random.default_rng(123)

    def act(self, obs):
        low, high = self.action_space.low, self.action_space.high
        return self._rng.uniform(low, high).astype(np.float32)


POLICIES = [
    DataCollectionPolicyDesignedRewardMaximizer,
    DataCollectionPolicyHeavyPress,
    DataCollectionPolicyHover,
    DataCollectionPolicyRandomExplore,
]
