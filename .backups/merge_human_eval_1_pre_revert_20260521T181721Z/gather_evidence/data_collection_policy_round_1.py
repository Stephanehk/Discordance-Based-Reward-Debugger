"""Round 1 data-collection policies for the highway `merge-v0` task.

Round 1 is broad-coverage: spread the ego across distinct driving regimes
(target-speed cruiser, max-speed aggressor, timid crawler, restless
lane-changer, fully passive driver) so later analysis can compare how the
designed reward and the stakeholder metrics rank very different styles of
merging.

Observation/denormalisation facts used here (verified in env_config.py):
  - ego forward speed (m/s) = obs[0, 3] * 80
  - neighbour relative position (m): x = obs[i, 1] * 200, y = obs[i, 2] * 8
  - rows 1..4 are the nearest neighbours in ego-relative coords; col 0 is
    presence (1.0 real, 0.0 padding).
Actions Discrete(5): 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER.
"""

import numpy as np

VX_HALFWIDTH_MPS = 80.0  # m/s — ego speed = obs[0, 3] * 80
X_HALFWIDTH_M = 200.0    # m — neighbour relative longitudinal half-range
Y_HALFWIDTH_M = 8.0      # m — neighbour relative lateral half-range

# Action ids (DiscreteMetaAction).
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4

_FAR_GAP_M = 200.0       # gap reported when no neighbour is present


def _ego_speed_mps(obs):
    """Ego forward speed in m/s from one observation matrix."""
    return float(np.asarray(obs)[0, 3]) * VX_HALFWIDTH_MPS


def _nearest_gap_m(obs):
    """Distance (m) to the nearest present neighbour; _FAR_GAP_M if none."""
    obs = np.asarray(obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:  # present, non-padding neighbour
            rx = obs[i, 1] * X_HALFWIDTH_M
            ry = obs[i, 2] * Y_HALFWIDTH_M
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else _FAR_GAP_M


class DataCollectionPolicyTargetSpeedCruiser:
    """Holds ~25 m/s while keeping >=10 m from the nearest vehicle; the
    reference behaviour that approximately maximises the designed reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.target_mps = 25.0
        self.speed_tol = 1.5    # m/s deadband around target
        self.min_gap_m = 10.0   # desired clearance to nearest vehicle

    def act(self, obs):
        speed = _ego_speed_mps(obs)
        gap = _nearest_gap_m(obs)
        # If too close to the nearest vehicle, try to create space: change
        # lane if a neighbour is essentially ahead/alongside, else ease off.
        if gap < self.min_gap_m:
            obs = np.asarray(obs)
            # Sign of nearest neighbour's lateral offset tells which side it
            # is on; steer toward the emptier side.
            nearest_ry = 0.0
            best = self.min_gap_m
            for i in range(1, obs.shape[0]):
                if obs[i, 0] > 0.5:
                    rx = obs[i, 1] * X_HALFWIDTH_M
                    ry = obs[i, 2] * Y_HALFWIDTH_M
                    d = float(np.hypot(rx, ry))
                    if d < best:
                        best = d
                        nearest_ry = ry
            if nearest_ry >= 0:   # neighbour to the right/ahead -> go left
                return LANE_LEFT
            return LANE_RIGHT
        # Otherwise regulate speed toward the 25 m/s target.
        if speed > self.target_mps + self.speed_tol:
            return SLOWER
        if speed < self.target_mps - self.speed_tol:
            return FASTER
        return IDLE


class DataCollectionPolicyMaxSpeedAggressor:
    """Always accelerates and rarely yields — probes high-speed,
    small-gap merging behaviour the designed reward may mis-rank."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return FASTER


class DataCollectionPolicyTimidCrawler:
    """Drives as slowly as possible, braking persistently — probes the
    very-slow regime well below the target speed."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return SLOWER


class DataCollectionPolicyRestlessLaneChanger:
    """Oscillates between left and right lane changes with brief cruising
    — probes frequent lateral manoeuvring and the near-miss states it
    creates."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.t = 0
        # Repeating pattern: change left, hold, change right, hold.
        self.pattern = [LANE_LEFT, IDLE, LANE_RIGHT, IDLE]

    def act(self, obs):
        a = self.pattern[self.t % len(self.pattern)]
        self.t += 1
        return a


class DataCollectionPolicyPassiveIdle:
    """Never intervenes (IDLE every step) — probes the do-nothing baseline,
    coasting at whatever speed the env initialises and merges into."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return IDLE


POLICIES = [
    DataCollectionPolicyTargetSpeedCruiser,
    DataCollectionPolicyMaxSpeedAggressor,
    DataCollectionPolicyTimidCrawler,
    DataCollectionPolicyRestlessLaneChanger,
    DataCollectionPolicyPassiveIdle,
]
