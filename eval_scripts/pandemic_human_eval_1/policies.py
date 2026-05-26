"""Evaluation data-collection policies for the ``pandemic_human_eval_1`` env.

These ten policies span constant-stage extremes, the chaos baseline, and a
range of reactive stakeholder views over the regulation stage (0..4). They
exercise rewards on every axis we expect to matter: the *level* of
intervention (always-off vs always-on vs middle vs reactive), the
*responsiveness* to critical-case load, the *smoothness* of the trajectory
(steady vs volatile), and structural preferences such as skipping a stage
or refusing to relax. Together they should yield trajectories that *tease
apart* candidate rewards by surfacing where they order these behaviours
differently.

Conforms to the workspace policy invariant (``__init__(self, env)`` +
``act(self, obs) -> action``); the module-level ``POLICIES`` list is the
canonical export.

Action: a discrete integer in {0, 1, 2, 3, 4} naming the regulation stage
to impose (0 = no restrictions, 4 = strictest lockdown).

Observation accessors (latest row obs[-1, 0], length 13):
    stage   = obs[-1, 0, 1]   (int 0..4)
    critical proportion = obs[-1, 0, 3] + obs[-1, 0, 8]
"""

import numpy as np


def _row(obs):
    """Latest length-13 state vector."""
    return np.asarray(obs)[-1, 0]


def _current_stage(obs):
    return int(round(float(_row(obs)[1])))


def _critical(obs):
    r = _row(obs)
    return float(r[3]) + float(r[8])


class DataCollectionPolicyNoRestrictions:
    """Always stage 0 — ignores the outbreak entirely."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return 0


class DataCollectionPolicyAlwaysLockdown:
    """Always stage 4 — strictest lockdown regardless of outbreak severity."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return 4


class DataCollectionPolicyMiddleGround:
    """Always stage 2 — moderate restrictions, no adaptation."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return 2


class DataCollectionPolicyUniformRandom:
    """Uniform random over the five stages — chaotic baseline, very volatile."""

    def __init__(self, env, seed=0):
        self.action_space = env.action_space
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(0, 5))


class DataCollectionPolicyProportionalReactive:
    """Smoothly ramps the stage with the critical-case proportion (uses all five stages)."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        c = _critical(obs)
        if c < 0.005:
            return 0
        if c < 0.01:
            return 1
        if c < 0.02:
            return 2
        if c < 0.04:
            return 3
        return 4


class DataCollectionPolicyConservativeReactive:
    """Errs on the side of restriction — escalates earlier and ratchets up faster."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        c = _critical(obs)
        if c < 0.002:
            return 1
        if c < 0.01:
            return 2
        if c < 0.02:
            return 3
        return 4


class DataCollectionPolicyLenientReactive:
    """Keeps stages low — delays escalation and never goes beyond stage 3."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        c = _critical(obs)
        if c < 0.02:
            return 0
        if c < 0.04:
            return 1
        if c < 0.06:
            return 2
        return 3


class DataCollectionPolicyVolatileFlipper:
    """Alternates between stage 0 and stage 4 every step — extreme jitter."""

    def __init__(self, env):
        self.action_space = env.action_space
        self.t = 0

    def act(self, obs):
        self.t += 1
        return 4 if (self.t % 2 == 0) else 0


class DataCollectionPolicySkipStage2:
    """Reactive but never uses stage 2 — represents the 'stage 2 is ineffective' view."""

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        c = _critical(obs)
        if c < 0.005:
            return 0
        if c < 0.015:
            return 1
        if c < 0.04:
            return 3
        return 4


class DataCollectionPolicyMonotonicEscalator:
    """Reactive but never lowers the stage — once at k, stays >= k."""

    def __init__(self, env):
        self.action_space = env.action_space
        self.floor = 0

    def act(self, obs):
        c = _critical(obs)
        if c < 0.005:
            target = 0
        elif c < 0.01:
            target = 1
        elif c < 0.02:
            target = 2
        elif c < 0.04:
            target = 3
        else:
            target = 4
        self.floor = max(self.floor, target)
        return self.floor


POLICIES = [
    DataCollectionPolicyNoRestrictions,
    DataCollectionPolicyAlwaysLockdown,
    DataCollectionPolicyMiddleGround,
    DataCollectionPolicyUniformRandom,
    DataCollectionPolicyProportionalReactive,
    DataCollectionPolicyConservativeReactive,
    DataCollectionPolicyLenientReactive,
    DataCollectionPolicyVolatileFlipper,
    DataCollectionPolicySkipStage2,
    DataCollectionPolicyMonotonicEscalator,
]
