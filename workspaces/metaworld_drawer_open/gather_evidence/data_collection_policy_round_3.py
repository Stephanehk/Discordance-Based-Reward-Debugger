"""Round 3 data-collection policies for metaworld_drawer_open.

The reward is unchanged, so this round goes after angles rounds 1-2 never
produced:

  - the exchange rate the reward implies between *how far the drawer is open*
    and *whether the hand is still on the handle* (half-open-and-camp vs.
    fully-open-and-leave),
  - H8, the un-penalized over-run: keep wrenching the handle once the drawer is
    already open, including off its axis,
  - unstructured state-space coverage (a pure random walk), which no prior round
    included — every policy so far was a purposeful script.

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


class DataCollectionPolicyHalfOpenThenCamp(_BasePolicy):
    """Trades outcome against presence: pulls the drawer only part-way open, then parks
    on the handle for the rest of the episode. Tests the exchange rate the reward implies
    between finishing the job and keeping the hand where the credit is paid."""

    SEED = 21

    def _on_reset(self):
        # Fraction of the full travel to pull before stopping and camping.
        self._stop_frac = self._rng.uniform(0.35, 0.65)
        self._start_dist = None

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        if self._start_dist is None:
            self._start_dist = float(np.linalg.norm(handle - target))
        hook = handle + HANDLE_GRASP_OFFSET
        remaining = float(np.linalg.norm(handle - target))
        opened_frac = 1.0 - remaining / max(self._start_dist, 1e-6)
        delta, arrived = self._approach(hand, handle)
        if arrived:
            if opened_frac < self._stop_frac:
                delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
            else:
                delta = _move(hand, hook, 6.0)  # camp on the handle, job half done
        return _act4(delta, -1.0)


class DataCollectionPolicyWrenchPastOpen(_BasePolicy):
    """H8: once the drawer is open, keeps hauling on the handle at full force and drags it
    off its axis. The opening term saturates near the target and nothing penalizes over-run,
    so wrenching the handle around costs the reward nothing."""

    SEED = 22

    def _on_reset(self):
        self._lateral = self._rng.choice([-1.0, 1.0]) * self._rng.uniform(0.5, 1.0)

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        delta, arrived = self._approach(hand, handle, p_align=10.0, p_drop=8.0)
        if arrived:
            if float(np.linalg.norm(handle - target)) > OPEN_TOL:
                delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
            else:
                # Open already: keep pulling, and haul the handle sideways as well.
                delta = np.array([self._lateral, -1.0, 0.0])
        return _act4(delta, -1.0)


class DataCollectionPolicyRandomWalk(_BasePolicy):
    """Unstructured coverage: samples actions uniformly at random. Every prior policy was a
    purposeful script, so the reward's behavior across the rest of the state space is
    untested."""

    SEED = 23

    def _step(self, o):
        return np.append(self._rng.uniform(-1.0, 1.0, size=3), self._rng.uniform(-1.0, 1.0))


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyHalfOpenThenCamp,
    DataCollectionPolicyWrenchPastOpen,
    DataCollectionPolicyRandomWalk,
]
