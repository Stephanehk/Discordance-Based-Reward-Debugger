"""Round 2 data-collection policies for metaworld_hammer.

The reward is unchanged, so the decomposition in `.repair_history/_reward_analysis.md`
still holds. Round 1 established that the reward pays for holding the hammer near
the nail's driven-in pose rather than for the nail going in. Round 2 pushes the
three consequences round 1 only approached:

  * Round 1's under-drive probe stopped the nail at 0.05 m, still 0.05 m short of
    the target — outside the reward's 0.02 m tolerance band. Parking it *just under*
    the 0.09 m success line puts the hammer head inside that band, where the shaped
    reward is near its 8/step ceiling. If that beats the reward maximizer, the
    reward strictly prefers a job left unfinished (H2).
  * Nothing penalises violence, so a run can succeed AND thrash at full command
    magnitude for the rest of the episode at no cost (H6).
  * The success override only asks that the nail be deep and the hammer be held
    near it — not that the hammer did the work. The nail can be pushed in by hand
    and the hammer picked up afterwards purely to collect the override (H5/H8).

Observation slices used: hand = obs[0:3], hammer = obs[4:7], nail head = obs[11:14],
target = obs[39:42], tcp = obs[45:48], nail slide depth = obs[57].
"""

import numpy as np

from gather_evidence.data_collection_policy_round_1 import (
    GRIP_CLOSE,
    GRIP_OPEN,
    STRIKE_POS,
    SUCCESS_DEPTH,
    _act4,
    _BasePolicy,
    _move,
)


class DataCollectionPolicyRewardMaximizer2(_BasePolicy):
    """Maximizes the current designed reward: drive the nail in as early as possible, then keep the hammer pressed against it so the per-step success override keeps re-firing."""

    SEED = 11

    def _step(self, o):
        return self._drive(o, goal=STRIKE_POS)


class DataCollectionPolicyPoisedAtSuccessLine(_BasePolicy):
    """H2 (reward_in_place vs the success override): stops the nail a hair short of the success line and holds the hammer there, so the hammer head sits inside the reward's tolerance band and collects a near-maximal per-step payout for an episode in which the task is never completed."""

    SEED = 12
    DEPTH = 0.085  # meters: just under the 0.09 m that counts as driven in

    def _new_episode(self):
        self._hold = None

    def _step(self, o):
        hand, nail = o[0:3], float(o[57])
        if self._hold is not None:
            # Hold station: the hammer head stays parked just short of the target.
            return _act4(_move(hand, self._hold, self._p), GRIP_CLOSE)
        if nail >= self.DEPTH:
            self._hold = hand.copy()
            return _act4(np.zeros(3), GRIP_CLOSE)
        # Close the gap fast, then creep so the nail can be stopped precisely
        # short of the line rather than blowing through it.
        head_to_nail = float(np.linalg.norm((o[4:7] + np.array([0.16, 0.06, 0.0])) - o[11:14]))
        gap = self.DEPTH - nail  # meters of travel still wanted
        if head_to_nail > 0.05:
            y_rate = 1.0
        elif gap > 0.02:
            y_rate = 0.30
        else:
            y_rate = 0.06  # final approach: inch up to the line without crossing it
        return self._drive(o, y_rate=y_rate)


class DataCollectionPolicySucceedThenThrash(_BasePolicy):
    """H6 (missing penalty): drives the nail fully in and then, still holding the hammer against it, thrashes at maximum command magnitude for the rest of the episode — the reward keeps paying the success override while the arm does violent, pointless work."""

    SEED = 13

    def _new_episode(self):
        self._done = False

    def _step(self, o):
        hand, nail = o[0:3], float(o[57])
        if nail > SUCCESS_DEPTH + 0.005:
            self._done = True
        if not self._done:
            return self._drive(o, goal=STRIKE_POS)
        # Nail is in. Stay jammed against it — that keeps the override's gate
        # satisfied — but shake the arm as hard as the action range allows.
        base = _move(hand, STRIKE_POS + self._jit, self._p)
        thrash = self._rng.uniform(-1.0, 1.0, size=3)
        return _act4(np.clip(base + thrash, -1.0, 1.0), GRIP_CLOSE)


class DataCollectionPolicyHandDriveThenPose(_BasePolicy):
    """H5/H8 (override gate): pushes the nail into the wall with the bare gripper, then goes back for the hammer and holds it against the already-driven nail — collecting the success override for a nail the hammer never struck."""

    SEED = 14

    def _new_episode(self):
        self._driven = False
        self._release_until = None  # steps spent un-clenching before fetching the hammer

    def _step(self, o):
        hand, nail_head, nail = o[0:3], o[11:14], float(o[57])
        if nail > SUCCESS_DEPTH + 0.005 and not self._driven:
            self._driven = True
            self._release_until = self._t + 12
        if not self._driven:
            # Phase 1: shove the nail in with the fingers pinched shut. No hammer.
            approach = nail_head + np.array([-0.02, -0.06, 0.0]) + self._jit
            if float(np.linalg.norm(hand - approach)) > 0.03:
                return _act4(_move(hand, approach, self._p), GRIP_CLOSE)
            return _act4(_move(hand, nail_head + np.array([-0.02, 0.12, 0.0]), self._p), GRIP_CLOSE)
        if self._t < self._release_until:
            # Back off and open the fingers first. Approaching the hammer still
            # clenched would just pin it to the table instead of grasping it.
            return _act4(_move(hand, hand + np.array([0.0, -0.10, 0.06]), self._p), GRIP_OPEN)
        # Phase 2: the nail is already in. Fetch the hammer and pose with it
        # against the driven nail so the shaped sum clears the override's gate.
        return self._drive(o, goal=STRIKE_POS)


POLICIES = [
    DataCollectionPolicyRewardMaximizer2,
    DataCollectionPolicyPoisedAtSuccessLine,
    DataCollectionPolicySucceedThenThrash,
    DataCollectionPolicyHandDriveThenPose,
]
