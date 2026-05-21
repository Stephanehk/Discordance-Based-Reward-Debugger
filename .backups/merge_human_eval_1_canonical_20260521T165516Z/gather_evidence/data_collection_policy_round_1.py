"""Round 1 data-collection policies for the highway `merge-v0` task.

The current designed reward has two terms only: a large crash penalty and
a reward proportional to ego forward speed (monotonic — faster is always
better up to the env clip). It does NOT credit keeping distance and does
NOT target ~25 m/s. These policies span behaviours that should let the
speed-only reward disagree with the safety, eco, comfort and target-speed
stakeholder views, and include the required maximizer of the current
reward.

Action ids (DiscreteMetaAction): 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT,
3 FASTER, 4 SLOWER.
"""

import numpy as np

# Denormalisation (verified against the live simulator):
X_HALFWIDTH_M = 200.0  # m — normalised longitudinal position half-range
Y_HALFWIDTH_M = 8.0    # m — normalised lateral position half-range
VX_HALFWIDTH_MPS = 80.0  # m/s — normalised vx half-range

LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


def _nearest_gap(obs):
    """Distance (m) to the nearest observed leading vehicle, or large if none."""
    obs = np.asarray(obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:
            rx = obs[i, 1] * X_HALFWIDTH_M
            ry = obs[i, 2] * Y_HALFWIDTH_M
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else X_HALFWIDTH_M


def _ego_speed(obs):
    return float(np.asarray(obs)[0, 3]) * VX_HALFWIDTH_MPS


class DataCollectionPolicyRewardMaximizer:
    """Maximizes the current designed reward: pushes for top speed, only easing off to avoid an imminent crash."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        # Speed is the sole reward (crash is the only thing to avoid), so
        # go FASTER unless a leading vehicle is close enough to risk the
        # -100 crash penalty, in which case ease off just enough.
        gap = _nearest_gap(obs)
        if gap < 8.0:
            return SLOWER
        return FASTER


class DataCollectionPolicySafeFollower:
    """Keeps a generous gap and a moderate ~25 m/s cruise — high on the safety and target-speed views, lower designed reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        gap = _nearest_gap(obs)
        speed = _ego_speed(obs)
        if gap < 25.0:
            return SLOWER          # back off to preserve a comfortable gap
        if speed < 24.0:
            return FASTER          # build back up toward the ~25 m/s target
        if speed > 27.0:
            return SLOWER          # don't overshoot the target
        return IDLE


class DataCollectionPolicyAggressiveTailgater:
    """Always accelerates and closes on the leader — high designed reward via raw speed, poor on the safety view, prone to crashing."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return FASTER              # ignores gaps entirely


class DataCollectionPolicySlowCruiser:
    """Crawls well below the target — very safe gaps and low energy, but far from the 25 m/s target and low designed reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        speed = _ego_speed(obs)
        if speed > 14.0:
            return SLOWER
        return IDLE


class DataCollectionPolicyLaneWeaver:
    """Repeatedly weaves between lanes while moving briskly — stresses the ride-comfort view at little cost to the designed reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.t = 0

    def act(self, obs):
        self.t += 1
        # Alternate lane-change requests every few steps; accelerate between.
        phase = self.t % 4
        if phase == 0:
            return LANE_LEFT
        if phase == 2:
            return LANE_RIGHT
        return FASTER


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicySafeFollower,
    DataCollectionPolicyAggressiveTailgater,
    DataCollectionPolicySlowCruiser,
    DataCollectionPolicyLaneWeaver,
]
