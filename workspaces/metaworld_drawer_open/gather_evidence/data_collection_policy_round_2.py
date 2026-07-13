"""Round 2 data-collection policies for metaworld_drawer_open.

Round 1 left one hypothesis un-instantiated: H2, the rank inversion. Every
round-1 trajectory that opened the drawer still out-totalled the hover runs
that never touched it, so the reward's most damning possible failure — ranking
"never touched the drawer" above "opened the drawer" — has not actually been
produced. Both reward terms pay per step, so the way to produce it is to
succeed *late*: spend most of the episode away from the drawer, then open it.

Also here: a cleaner H3 contrast (open the drawer gently so it stays open, then
withdraw the arm) and a do-nothing floor that shows what the reward pays for
simply existing near nothing at all.

Re-includes the reward maximizer under the current (unchanged) reward.
"""

import numpy as np

from gather_evidence.data_collection_policy_round_1 import (
    DataCollectionPolicyRewardMaximizer,
    _BasePolicy,
    _act4,
    _move,
    HANDLE_GRASP_OFFSET,
    OPEN_TOL,
)


class DataCollectionPolicyDawdleThenOpen(_BasePolicy):
    """H2 rank inversion: waits away from the drawer for most of the episode, then
    goes and opens it, ending fully open. Both reward terms are paid per step, so a
    late success banks far less than an early one — and, we predict, less than simply
    hovering at the handle and never opening the drawer at all."""

    SEED = 11

    def _on_reset(self):
        # Steps spent waiting before starting the task. The approach + pull takes
        # ~65 steps, so this range still ends the episode with the drawer open.
        self._dawdle = int(self._rng.integers(45, 86))

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        if self._t < self._dawdle:
            return _act4(np.zeros(3), -1.0)  # wait, hand parked where it started
        hook = handle + HANDLE_GRASP_OFFSET
        delta, arrived = self._approach(hand, handle, p_align=10.0, p_drop=8.0)
        if arrived:
            if float(np.linalg.norm(handle - target)) > OPEN_TOL:
                delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
            else:
                delta = _move(hand, hook, 6.0)
        return _act4(delta, -1.0)


class DataCollectionPolicyGentleOpenThenRetreat(_BasePolicy):
    """H3: opens the drawer with a slow, steady pull that leaves it resting at the open
    pose, then lifts clear and withdraws the arm — the drawer stays open, but the
    hand-near-handle term stops paying the moment the arm leaves."""

    SEED = 12

    def _on_reset(self):
        self._pull_gain = self._rng.uniform(0.25, 0.40)
        self._dwell = int(self._rng.integers(0, 10))  # steps held at the handle once open
        self._opened_at = None

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        is_open = float(np.linalg.norm(handle - target)) <= OPEN_TOL
        if is_open and self._opened_at is None:
            self._opened_at = self._t
        if self._opened_at is not None and self._t >= self._opened_at + self._dwell:
            # Lift clear of the handle before backing out, so the fingers don't drag
            # the drawer shut on the way.
            if hand[2] < handle[2] + 0.12:
                return _act4(_move(hand, hand + np.array([0.0, 0.0, 0.2]), 6.0), -1.0)
            park = np.array([handle[0], handle[1] + 0.25, handle[2] + 0.15])
            return _act4(_move(hand, park, 5.0), -1.0)
        delta, arrived = self._approach(hand, handle, p_align=4.0, p_drop=4.0)
        if arrived and not is_open:
            delta = np.array([0.0, -self._pull_gain, 0.0])  # gentle steady pull
        return _act4(delta, -1.0)


class DataCollectionPolicyDoNothing(_BasePolicy):
    """Floor case: never moves at all. Establishes what the reward pays a robot that
    does nothing whatsoever, so the credit collected by merely floating near the handle
    can be read against it."""

    SEED = 13

    def _step(self, o):
        return _act4(np.zeros(3), -1.0)


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyDawdleThenOpen,
    DataCollectionPolicyGentleOpenThenRetreat,
    DataCollectionPolicyDoNothing,
]
