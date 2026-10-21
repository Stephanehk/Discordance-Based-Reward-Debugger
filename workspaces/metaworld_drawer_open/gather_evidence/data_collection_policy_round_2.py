"""Round 2 data-collection policies for metaworld_drawer_open.

Focused round targeting H2 (the telescoping endpoint reward ignores timing and
dwell — only the final openness is read). Round 1's dawdle-then-open policy
failed to actually open the drawer, so the clean H2 contrast is still missing:
two trajectories that both end *fully open* (so both collect ~full opening
credit) but differ sharply in *when* the drawer opens and how long it stays
open.

  OpenEarlyAndHold : opens fully within the first ~20 steps and stays clamped
                     open to the end (fast, high sustained-open).
  OpenOnlyAtEnd    : H2 probe — stays lined up above the handle without opening
                     it for most of the episode, then drops and yanks it fully
                     open only in the final stretch (full opening credit, but
                     open for just a few steps and slow to get there).

The required current-reward maximizer is OpenEarlyAndHold (fully open as early
as possible, then held). Observation slices as in round 1.
"""

import numpy as np

HANDLE_GRASP_OFFSET = np.array([0.0, 0.0, -0.02])
OPEN_TOL = 0.03
FULL_TOL = 0.005  # meters; pull until essentially fully open (openness ~ 1.0)


def _move(cur, to, p):
    return np.clip(p * (np.asarray(to) - np.asarray(cur)), -1.0, 1.0)


def _act4(delta_xyz, grip):
    a = np.zeros(4, dtype=np.float32)
    a[:3] = delta_xyz
    a[3] = grip
    return a


class _BasePolicy:
    """Shared episode bookkeeping: detect resets, re-seed, expose obs slices."""

    SEED = 0

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._episode = -1
        self._t = 0
        self._base_seed = 1000 * self.SEED + 7919 * int(getattr(env, "rd_seed", 0) or 0)
        self._rng = np.random.default_rng(self._base_seed)
        self._handle_y0 = None

    def act(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        handle = o[4:7]
        shut = float(np.linalg.norm(handle - o[39:42])) > 0.15
        if shut and np.array_equal(o[0:18], o[18:36]):
            self._episode += 1
            self._t = 0
            self._rng = np.random.default_rng(self._base_seed + self._episode)
            self._handle_y0 = float(handle[1])
            self._on_reset()
        a = self._step(o)
        self._t += 1
        return np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)

    def _on_reset(self):
        pass

    def _step(self, o):
        raise NotImplementedError

    def _approach(self, hand, handle, p_align=8.0, p_drop=8.0, hover_h=0.2):
        hook = handle + HANDLE_GRASP_OFFSET
        if np.linalg.norm(hand[:2] - hook[:2]) > 0.04:
            return _move(hand, hook + np.array([0.0, 0.0, hover_h]), p_align), False
        if abs(hand[2] - hook[2]) > 0.03:
            return _move(hand, hook, p_drop), False
        return _move(hand, hook, p_drop), True


class DataCollectionPolicyOpenEarlyAndHold(_BasePolicy):
    """Maximizes the current designed reward: reach the handle and yank the drawer
    all the way open as early as possible, then keep a steady light pull so it stays
    fully open (openness ~ 1) and clamped to the end. Fast and high sustained-open."""

    SEED = 1

    def _on_reset(self):
        self._p_align = 28.0 + self._rng.uniform(-4.0, 4.0)

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        remaining = float(np.linalg.norm(handle - target))
        delta, arrived = self._approach(hand, handle, p_align=self._p_align, p_drop=28.0)
        if arrived:
            if remaining > FULL_TOL:
                delta = _move(hand, hook + np.array([0.0, -0.08, 0.0]), 50.0)  # yank fully open
            else:
                delta = _move(hand, hook, 6.0) + np.array([0.0, -0.05, 0.0])  # hold fully open
        return _act4(delta, -1.0)


class DataCollectionPolicyOpenOnlyAtEnd(_BasePolicy):
    """H2 (T1 endpoint-only): keeps the hand hovering lined up a little above the
    handle without opening the drawer for most of the episode, then drops on and
    yanks it open only in the back stretch of the episode. Ends open (full opening
    credit, indistinguishable to the reward from opening immediately) yet the drawer
    reaches the goal late and is open for only a fraction of the episode."""

    SEED = 2

    def _on_reset(self):
        # Horizon is 150 and a full pull takes ~55-60 steps, so start the pull late
        # but early enough that it still reaches (near-)full open by the end.
        self._open_start = int(self._rng.integers(78, 88))

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        remaining = float(np.linalg.norm(handle - target))
        if self._t < self._open_start:
            # Loiter lined up in XY but held clear above the handle so the drawer
            # stays shut; small jitter keeps it from drifting into contact.
            hover = hook + np.array([0.0, 0.03, 0.12])
            jit = self._rng.uniform(-0.03, 0.03, size=3)
            return _act4(_move(hand, hover, 4.0) + jit, -1.0)
        delta, arrived = self._approach(hand, handle, p_align=30.0, p_drop=30.0)
        if arrived and remaining > FULL_TOL:
            delta = _move(hand, hook + np.array([0.0, -0.08, 0.0]), 50.0)
        return _act4(delta, -1.0)


POLICIES = [
    DataCollectionPolicyOpenEarlyAndHold,
    DataCollectionPolicyOpenOnlyAtEnd,
]
