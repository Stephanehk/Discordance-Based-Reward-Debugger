"""Round 1 data-collection policies for highway-env merge-v0.

These hand-coded heuristics produce diverse merge behaviours so that the
designed reward (penalise crashes, otherwise reward raw forward speed) can be
contrasted against stakeholder metrics for safety, comfort, eco/efficiency and
throughput. Designed purely from generic highway-driving strategies.
"""

import numpy as np

_SPEED_MPS = 80.0
_LONG_M = 200.0

# Action ids (DiscreteMetaAction).
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


def _ego_speed_mps(obs):
    return float(np.asarray(obs)[0, 3]) * _SPEED_MPS


def _nearest_lead_gap_m(obs):
    o = np.asarray(obs)
    gaps = []
    for r in range(1, o.shape[0]):
        if float(o[r, 0]) <= 0.5:
            continue
        dx = float(o[r, 1]) * _LONG_M
        if dx > 0.0:
            gaps.append(dx)
    return min(gaps) if gaps else _LONG_M


class DataCollectionPolicySpeedMaximizer:
    """Maximises the designed reward: always pushes for top speed, easing off only to avoid an imminent crash."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        gap = _nearest_lead_gap_m(obs)
        # Only back off when a leading vehicle is dangerously close.
        if gap < 8.0:
            return SLOWER
        return FASTER


class DataCollectionPolicyTailgater:
    """High designed reward, low safety metric: keeps top speed and rides close behind leaders without slowing."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        gap = _nearest_lead_gap_m(obs)
        # Only the very last instant before contact triggers a brake; otherwise
        # accelerate even when tailgating closely.
        if gap < 4.0:
            return SLOWER
        return FASTER


class DataCollectionPolicyCautiousCruiser:
    """Low designed reward, high safety/comfort/eco: holds a moderate speed and a wide following gap."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.target_mps = 22.0

    def act(self, obs):
        spd = _ego_speed_mps(obs)
        gap = _nearest_lead_gap_m(obs)
        if gap < 20.0:
            return SLOWER
        if spd < self.target_mps:
            return FASTER
        if spd > self.target_mps + 3.0:
            return SLOWER
        return IDLE


class DataCollectionPolicyLaneWeaver:
    """Probes the comfort metric: chases speed while changing lanes frequently, generating lateral jerk the reward ignores."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.t = 0

    def act(self, obs):
        self.t += 1
        gap = _nearest_lead_gap_m(obs)
        if gap < 8.0:
            return SLOWER
        # Alternate lane changes interleaved with acceleration.
        if self.t % 3 == 0:
            return LANE_LEFT
        if self.t % 3 == 1:
            return LANE_RIGHT
        return FASTER


class DataCollectionPolicyStopAndGo:
    """Covers unexplored low-speed states: oscillates speed up and down, hurting comfort/eco while exposing reward behaviour at low speed."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.t = 0

    def act(self, obs):
        self.t += 1
        gap = _nearest_lead_gap_m(obs)
        if gap < 8.0:
            return SLOWER
        # Sawtooth: accelerate for a few steps, then brake hard.
        return FASTER if (self.t // 3) % 2 == 0 else SLOWER


POLICIES = [
    DataCollectionPolicySpeedMaximizer,
    DataCollectionPolicyTailgater,
    DataCollectionPolicyCautiousCruiser,
    DataCollectionPolicyLaneWeaver,
    DataCollectionPolicyStopAndGo,
]
