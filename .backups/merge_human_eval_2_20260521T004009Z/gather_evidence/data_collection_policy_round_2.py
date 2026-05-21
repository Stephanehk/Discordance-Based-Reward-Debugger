"""Round 2 data-collection policies for highway-env merge-v0 (focused).

The reward was revised since round 1: it now peaks at a band of ego speeds
near the middle of the range, adds a small left-lane bonus, and still
penalises crashes. These three heuristics target that revised reward
specifically. One policy maximises the revised reward directly; the other
two hold the same reward-favoured speed/lane band while deliberately
trading away safety (tight following gap) or comfort (needless speed and
lane oscillation) — quantities the revised reward still does not credit.
Designed purely from the env semantics and the reward's mechanics.
"""

import numpy as np

_SPEED_MPS = 80.0
_LONG_M = 200.0
_LAT_M = 8.0

# Action ids (DiscreteMetaAction).
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4

# Reward-favoured operating point read off the revised reward: speed term
# peaks at ~25 m/s, left-lane bonus when lateral < ~2 m.
_TARGET_MPS = 25.0
_LEFT_LANE_M = 2.0


def _np(obs):
    return np.asarray(obs, dtype=float)


def _ego_speed_mps(obs):
    return float(_np(obs)[0, 3]) * _SPEED_MPS


def _ego_lateral_m(obs):
    return float(_np(obs)[0, 2]) * _LAT_M


def _nearest_lead_gap_m(obs):
    """Longitudinal gap (m) to the nearest present neighbour ahead, or large if none."""
    o = _np(obs)
    gaps = []
    for r in range(1, o.shape[0]):
        if float(o[r, 0]) <= 0.5:  # padding / absent row
            continue
        dx = float(o[r, 1]) * _LONG_M
        if dx > 0.0:  # only vehicles ahead of the ego
            gaps.append(dx)
    return min(gaps) if gaps else _LONG_M


def _speed_action_toward(spd, target, tol=1.5):
    """Pick FASTER/SLOWER/IDLE to drive ego speed toward `target` m/s."""
    if spd < target - tol:
        return FASTER
    if spd > target + tol:
        return SLOWER
    return IDLE


class DataCollectionPolicyRewardMaximizer:
    """Maximises the revised reward: holds ~25 m/s in the left lane, easing only to avoid an imminent crash."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        spd = _ego_speed_mps(obs)
        lat = _ego_lateral_m(obs)
        gap = _nearest_lead_gap_m(obs)
        # Safety floor: back off only when a leader is dangerously close.
        if gap < 8.0:
            return SLOWER
        # Prefer reaching the left lane (lateral < ~2 m) for the lane bonus.
        if lat >= _LEFT_LANE_M:
            return LANE_LEFT
        # Already in the left lane: hold the reward-peak speed band.
        return _speed_action_toward(spd, _TARGET_MPS)


class DataCollectionPolicyTightFollower:
    """High reward, low safety: holds ~25 m/s in the left lane while riding very close behind a leader."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        spd = _ego_speed_mps(obs)
        lat = _ego_lateral_m(obs)
        gap = _nearest_lead_gap_m(obs)
        # Only the very last instant before contact triggers a brake; the
        # reward gives no credit for buffer, so a tiny gap is held on purpose.
        if gap < 3.0:
            return SLOWER
        # Reach/keep the left lane for the lane bonus.
        if lat >= _LEFT_LANE_M:
            return LANE_LEFT
        # Hold the reward-peak speed band right behind the leader.
        return _speed_action_toward(spd, _TARGET_MPS)


class DataCollectionPolicyComfortBreaker:
    """Comparable reward, low comfort: keeps speed near the reward band but oscillates speed and lane choice needlessly."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.t = 0

    def act(self, obs):
        self.t += 1
        spd = _ego_speed_mps(obs)
        gap = _nearest_lead_gap_m(obs)
        if gap < 8.0:
            return SLOWER
        # Cycle of length 4: jitter speed around the band (overshoot up then
        # down) and inject needless lane changes, staying near the rewarded
        # speed on average while wrecking comfort the reward ignores.
        phase = self.t % 4
        if phase == 0:
            return LANE_LEFT
        if phase == 1:
            # Push above the band so the subsequent brake brings it back.
            return FASTER
        if phase == 2:
            return LANE_RIGHT
        # phase == 3: pull speed back toward the band.
        return SLOWER if spd > _TARGET_MPS else FASTER


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyTightFollower,
    DataCollectionPolicyComfortBreaker,
]
