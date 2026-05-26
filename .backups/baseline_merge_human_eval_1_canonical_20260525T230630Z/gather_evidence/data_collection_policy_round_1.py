"""Round-1 data collection policy: a hand-coded maximizer of the designed reward.

The designed reward pays the ego's normalised forward speed (scaled by 30) every
non-crash step and a fixed -100 on a crash step. Expected return is therefore
maximized by driving as fast as possible while never colliding. This controller
defaults to FASTER, but slows (or, if blocked, changes lanes toward clear space)
when a vehicle is close ahead in the ego's lane, since a crash is far costlier
than the speed it would buy.
"""

import math

import numpy as np


# Action ids (DiscreteMetaAction).
LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4

# Denormalization constants (verified facts).
X_SCALE = 200.0  # relative longitudinal position -> metres
Y_SCALE = 8.0    # relative lateral position -> metres

# Tuning thresholds (metres), from general driving sense.
SLOW_GAP = 25.0       # back off if a leading car is closer than this
LANE_CHANGE_GAP = 18.0  # very close: prefer a lane change to clear space if possible
LANE_HALF_WIDTH = 2.0   # |y_m| under this counts as "in the ego's lane"


class DataCollectionPolicyRewardMaximizer:
    """Maximizes the expected sum of the designed reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._last_lane_change_step = -10
        self._step = 0

    def act(self, obs):
        self._step += 1
        obs = np.asarray(obs, dtype=float)

        # Find the nearest leading vehicle in roughly the ego's lane.
        nearest_lead_x = None
        for i in range(1, obs.shape[0]):
            if obs[i, 0] <= 0.5:  # padding row, no real vehicle
                continue
            x_m = obs[i, 1] * X_SCALE
            y_m = obs[i, 2] * Y_SCALE
            if x_m <= 0.0:  # behind the ego, ignore for forward-collision purposes
                continue
            if abs(y_m) > LANE_HALF_WIDTH:  # different lane
                continue
            if nearest_lead_x is None or x_m < nearest_lead_x:
                nearest_lead_x = x_m

        # No close obstacle ahead: push speed up.
        if nearest_lead_x is None or nearest_lead_x >= SLOW_GAP:
            return FASTER

        # Something close ahead. If it's very close, try to escape into a clear
        # lane (without spamming lane changes); otherwise just slow down.
        if nearest_lead_x < LANE_CHANGE_GAP and (self._step - self._last_lane_change_step) > 3:
            lane_action = self._clear_lane_change(obs)
            if lane_action is not None:
                self._last_lane_change_step = self._step
                return lane_action

        return SLOWER

    def _clear_lane_change(self, obs):
        """Return LANE_LEFT/LANE_RIGHT if a neighbouring lane looks clear, else None."""
        # Aggregate occupancy in the lane to the left (y_m < 0) and right (y_m > 0).
        left_blocked = False
        right_blocked = False
        for i in range(1, obs.shape[0]):
            if obs[i, 0] <= 0.5:
                continue
            x_m = obs[i, 1] * X_SCALE
            y_m = obs[i, 2] * Y_SCALE
            # Only consider vehicles near the ego longitudinally (a hazard for a merge).
            if abs(x_m) > SLOW_GAP:
                continue
            # An adjacent lane is one lane-width away laterally.
            if -3.0 * LANE_HALF_WIDTH < y_m < -LANE_HALF_WIDTH * 0.5:
                left_blocked = True
            elif LANE_HALF_WIDTH * 0.5 < y_m < 3.0 * LANE_HALF_WIDTH:
                right_blocked = True

        if not left_blocked:
            return LANE_LEFT
        if not right_blocked:
            return LANE_RIGHT
        return None


POLICIES = [DataCollectionPolicyRewardMaximizer]
