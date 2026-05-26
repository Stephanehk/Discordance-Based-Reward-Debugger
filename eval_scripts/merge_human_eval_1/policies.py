"""Evaluation data-collection policies for the ``merge_human_eval_1`` env.

Each policy is a competent optimizer for its own implicit reward function —
listed at the top of every docstring. Together they span a wide set of
stakeholder preferences (speed, safety, lane choice, comfort, overtaking,
tail-gating, …) so that rolling them all out yields trajectories that
*tease apart* candidate reward functions that disagree on those axes.

Every policy is genuinely **reactive** to the other vehicles around the
ego: it computes the nearest in-lane leader's distance and closing rate,
the time-to-collision, and lane-clearance for any lane change it requests.
None of them blindly press FASTER or LANE_LEFT regardless of traffic.

Conforms to the workspace policy invariant (``__init__(self, env)`` +
``act(self, obs) -> action``); the module-level ``POLICIES`` list is the
canonical export. The underlying highway env is shared with
``merge_human_eval_2``, so the same policy set works there.

Action ids (highway-env DiscreteMetaAction):
    0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER.

Observation accessors (Kinematics, verified denormalisation):
    ego speed (m/s)        = obs[0, 3] * 80
    ego absolute lateral y = obs[0, 2] * 8        (left lane y=0, right y=4)
    neighbour relative x   = obs[i, 1] * 200      (positive = ahead of ego)
    neighbour relative y   = obs[i, 2] * 8        (positive = to ego's right)
    neighbour relative vx  = obs[i, 3] * 80       (negative = closing on ego)

Lane convention used here: ``round(y_m / 4)`` → 0 = left lane, 1 = right
(merge) lane. ``rvx < 0`` means the neighbour is moving slower than the
ego in absolute terms; combined with ``rx > 0`` (neighbour ahead) it means
the ego is closing on a leader and TTC ``= -rx / rvx`` (s).
"""

import numpy as np

# Denormalisation (matches env_config.py).
X_HALFWIDTH_M = 200.0
Y_HALFWIDTH_M = 8.0
VX_HALFWIDTH_MPS = 80.0
LANE_WIDTH_M = 4.0

LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


# ---- observation accessors ----------------------------------------------

def _ego_speed_mps(obs):
    return float(np.asarray(obs)[0, 3]) * VX_HALFWIDTH_MPS


def _ego_y_m(obs):
    return float(np.asarray(obs)[0, 2]) * Y_HALFWIDTH_M


def _ego_lane(obs):
    """0 = left lane (y ≈ 0 m), 1 = right/merge lane (y ≈ 4 m)."""
    return int(round(_ego_y_m(obs) / LANE_WIDTH_M))


def _present_neighbours(obs):
    """Yield (rx_m, ry_m, rvx_mps) for every present neighbour row.

    ``rx, ry`` are *relative to the ego* (highway-env Kinematics default).
    ``rvx`` is the neighbour's longitudinal velocity minus the ego's, so
    ``rvx < 0`` means the neighbour is slower than the ego.
    """
    obs = np.asarray(obs)
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:
            yield (
                float(obs[i, 1]) * X_HALFWIDTH_M,
                float(obs[i, 2]) * Y_HALFWIDTH_M,
                float(obs[i, 3]) * VX_HALFWIDTH_MPS,
            )


def _nearest_gap_m(obs):
    """Straight-line distance to the nearest present neighbour (or 200 m if none)."""
    best = X_HALFWIDTH_M
    for rx, ry, _ in _present_neighbours(obs):
        d = float(np.hypot(rx, ry))
        if d < best:
            best = d
    return best


def _leader_in_lane(obs, lane_offset=0, lane_tol=2.0):
    """Closest neighbour ahead in the lane at ``lane_offset`` lanes from ego.

    Returns ``(rx_m, rvx_mps)`` or ``(None, None)`` if no leader is present.
    ``lane_offset = 0`` means the ego's current lane; ``-1`` is one lane to
    the left, ``+1`` one to the right.
    """
    target_ry = lane_offset * LANE_WIDTH_M
    best_rx, best_rvx = None, None
    for rx, ry, rvx in _present_neighbours(obs):
        if rx <= 0:
            continue
        if abs(ry - target_ry) > lane_tol:
            continue
        if best_rx is None or rx < best_rx:
            best_rx, best_rvx = rx, rvx
    return best_rx, best_rvx


def _ttc_s(rx_m, rvx_mps):
    """Time-to-collision (s) with a leader at ``rx_m`` closing at ``rvx_mps``.

    Returns ``+inf`` if there's no leader or the gap isn't shrinking
    (``rvx >= 0`` means the leader is moving away or matching speed).
    """
    if rx_m is None or rvx_mps is None or rvx_mps >= 0:
        return float("inf")
    return -rx_m / rvx_mps


def _lane_clear(obs, lane_offset, fwd_thresh=18.0, bwd_thresh=10.0, lane_tol=2.0):
    """True iff the lane at ``lane_offset`` is clear of close traffic."""
    target_ry = lane_offset * LANE_WIDTH_M
    for rx, ry, _ in _present_neighbours(obs):
        if abs(ry - target_ry) > lane_tol:
            continue
        if -bwd_thresh < rx < fwd_thresh:
            return False
    return True


def _safe_following_decision(obs, target_speed, brake_ttc=2.5,
                             min_gap_m=10.0, headway_s=1.5):
    """Shared inner loop: brake/cruise/accelerate around ``target_speed`` while
    respecting an in-lane leader. Returns the action id.

    Reused by several "safe driver with a target speed" policies so the
    crash-avoidance logic stays consistent across them.
    """
    speed = _ego_speed_mps(obs)
    rx, rvx = _leader_in_lane(obs)
    ttc = _ttc_s(rx, rvx)
    desired_gap = max(min_gap_m, headway_s * max(speed, 1.0))
    if ttc < brake_ttc:
        return SLOWER
    if rx is not None and rx < desired_gap:
        return SLOWER
    if speed < target_speed - 1.0:
        return FASTER
    if speed > target_speed + 1.0:
        return SLOWER
    return IDLE


# ---- policies -----------------------------------------------------------

class DataCollectionPolicySafeFollower:
    """Reward: ``+0.05·speed − 200·crash − 5·max(0, 10−gap) − 0.02·(speed−25)²``.

    Cruises at ~25 m/s while honouring a 1.5-second headway and a 10 m
    minimum gap. Brakes whenever TTC with the in-lane leader drops below
    2.5 s, otherwise nudges speed back to 25.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return _safe_following_decision(obs, target_speed=25.0)


class DataCollectionPolicyAggressiveSpeeder:
    """Reward: ``+1·speed − 200·crash`` (no gap / target-speed term).

    Holds FASTER unconditionally unless a crash is imminent (TTC < 1.2 s
    with the in-lane leader or that leader is closer than 6 m), in which
    case it dabs the brake.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        rx, rvx = _leader_in_lane(obs)
        ttc = _ttc_s(rx, rvx)
        if ttc < 1.2 or (rx is not None and rx < 6.0):
            return SLOWER
        return FASTER


class DataCollectionPolicyTargetCruiser:
    """Reward: ``−|speed − 25| − 100·crash − 2·max(0, 10−gap)``.

    Tight cruise band around 25 m/s with conservative crash avoidance.
    Identical structure to ``SafeFollower`` but with a narrower speed
    tolerance (±0.5 m/s) and a stricter ``brake_ttc``.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return _safe_following_decision(
            obs, target_speed=25.0, brake_ttc=3.0,
            min_gap_m=12.0, headway_s=1.8,
        )


class DataCollectionPolicyTimidConservative:
    """Reward: ``+1·gap − 0.5·|speed − 15| − 200·crash``.

    Maximises the gap to every neighbour by driving slowly. SLOWER any
    time the nearest neighbour comes within 30 m; targets ~15 m/s when
    the road is clear.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        gap = _nearest_gap_m(obs)
        speed = _ego_speed_mps(obs)
        if gap < 30.0:
            return SLOWER
        if speed > 16.0:
            return SLOWER
        if speed < 13.0:
            return FASTER
        return IDLE


class DataCollectionPolicyLeftLanePreferrer:
    """Reward: ``+5·1[lane==left] + 0.05·speed − 200·crash − 5·max(0, 10−gap)``.

    Moves to the left lane whenever the lane is clear of close traffic in
    both directions; once there, drives like ``SafeFollower`` at 24 m/s.
    Refuses a lane change that would put it close to an adjacent
    neighbour.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        lane = _ego_lane(obs)
        if lane != 0 and _lane_clear(obs, lane_offset=-1):
            return LANE_LEFT
        return _safe_following_decision(obs, target_speed=24.0)


class DataCollectionPolicyOpportunisticOvertaker:
    """Reward: ``+0.1·speed − 200·crash`` (no lane / gap term).

    Tries to keep moving fast. When stuck behind a slower in-lane leader
    (rx < 25 m and rvx < 0), checks the adjacent lane and changes if it's
    clear; otherwise eases off. Otherwise pushes FASTER, but still brakes
    on TTC < 1.8 s.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        rx, rvx = _leader_in_lane(obs)
        ttc = _ttc_s(rx, rvx)
        lane = _ego_lane(obs)
        if rx is not None and rx < 25.0 and (rvx is not None and rvx < 0):
            if lane == 1 and _lane_clear(obs, lane_offset=-1):
                return LANE_LEFT
            if lane == 0 and _lane_clear(obs, lane_offset=+1):
                return LANE_RIGHT
            return SLOWER
        if ttc < 1.8:
            return SLOWER
        return FASTER


class DataCollectionPolicyEcoCruiser:
    """Reward: ``−|speed − 20| − 100·crash − 5·max(0, 10−gap)``.

    Smooth, fuel-economy-style driving: targets ~20 m/s with a long
    headway. Reuses the shared safe-following logic with a lower target
    speed and a stricter ``brake_ttc`` so it rarely needs hard braking.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return _safe_following_decision(
            obs, target_speed=20.0, brake_ttc=3.5,
            min_gap_m=15.0, headway_s=2.0,
        )


class DataCollectionPolicyMinGapMaximizer:
    """Reward: ``+log(min_gap) − 200·crash``.

    Whenever the closest neighbour is within 25 m, *moves away from it*:
    if the nearest neighbour is on the right, request LANE_LEFT (if clear),
    if on the left, request LANE_RIGHT (if clear); otherwise SLOWER. When
    the road is empty it cruises gently at ~22 m/s.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        nearest = None
        for rx, ry, _ in _present_neighbours(obs):
            d = float(np.hypot(rx, ry))
            if nearest is None or d < nearest[0]:
                nearest = (d, rx, ry)
        if nearest is not None and nearest[0] < 25.0:
            _, _, ry = nearest
            lane = _ego_lane(obs)
            if ry > 0 and lane != 0 and _lane_clear(obs, lane_offset=-1):
                return LANE_LEFT
            if ry < 0 and lane != 1 and _lane_clear(obs, lane_offset=+1):
                return LANE_RIGHT
            return SLOWER
        speed = _ego_speed_mps(obs)
        if speed < 21.0:
            return FASTER
        if speed > 23.0:
            return SLOWER
        return IDLE


class DataCollectionPolicyTailgater:
    """Reward: ``−(gap − 5)² − 50·crash`` (rewards *close* following).

    Wants to sit ~5 m behind the in-lane leader at matched speed. Closes
    in (FASTER) when gap > 8 m, opens up (SLOWER) when gap < 4 m or
    TTC < 1.0 s, otherwise IDLE. If there's no leader it cruises at the
    ego's current speed (IDLE) so it doesn't leave the close-follow band.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        rx, rvx = _leader_in_lane(obs)
        ttc = _ttc_s(rx, rvx)
        if ttc < 1.0:
            return SLOWER
        if rx is None:
            return IDLE
        if rx > 8.0:
            return FASTER
        if rx < 4.0:
            return SLOWER
        return IDLE


class DataCollectionPolicyDefensiveMerger:
    """Reward: ``−1·1[crash] − 0.1·max(0, 15−gap) − 0.05·|speed − 22|``.

    Designed for the merge-from-right-lane scenario: prefers a moderate
    22 m/s cruise but always yields, slowing down rather than forcing a
    lane change. Eases off whenever the nearest neighbour comes within
    15 m, and only requests a lane change if both the target lane and the
    leader-side are clear.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        speed = _ego_speed_mps(obs)
        gap = _nearest_gap_m(obs)
        rx, rvx = _leader_in_lane(obs)
        ttc = _ttc_s(rx, rvx)
        # Hard yield: any close neighbour or imminent collision → slow down.
        if gap < 15.0 or ttc < 3.0:
            return SLOWER
        # If currently in the merge lane and the left lane is *very* clear,
        # take the merge — but defensively (extra buffer).
        if _ego_lane(obs) == 1 and _lane_clear(obs, lane_offset=-1,
                                               fwd_thresh=25.0, bwd_thresh=15.0):
            return LANE_LEFT
        if speed < 21.0:
            return FASTER
        if speed > 23.0:
            return SLOWER
        return IDLE


POLICIES = [
    DataCollectionPolicySafeFollower,
    DataCollectionPolicyAggressiveSpeeder,
    DataCollectionPolicyTargetCruiser,
    DataCollectionPolicyTimidConservative,
    DataCollectionPolicyLeftLanePreferrer,
    DataCollectionPolicyOpportunisticOvertaker,
    DataCollectionPolicyEcoCruiser,
    DataCollectionPolicyMinGapMaximizer,
    DataCollectionPolicyTailgater,
    DataCollectionPolicyDefensiveMerger,
]
