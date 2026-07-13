"""Round 1 data-collection policies for metaworld_drawer_open.

Broad-coverage round. One policy maximizes the current designed reward; the
rest instantiate the exploit hypotheses in `.repair_history/_reward_analysis.md`
(H1-H8): hand-on-handle credit without a grasp, hovering that outscores a slow
success, being punished for letting go, opening then undoing it, violent /
jittery opens, hooking instead of gripping, and hovering above the handle.

Observation slices used: hand = obs[0:3], aperture = obs[3],
handle = obs[4:7], fully-open target = obs[39:42], hand-at-reset = obs[42:45].
"""

import numpy as np

# The policy object is reused across rollouts, and the env resets deterministically,
# so each policy re-seeds per episode and jitters its own parameters to get spread.
HANDLE_GRASP_OFFSET = np.array([0.0, 0.0, -0.02])  # hook point, just under the handle
OPEN_TOL = 0.03  # meters; drawer counts as open within this of the target


def _move(cur, to, p):
    """Proportional end-effector command toward `to`, clipped to the action range."""
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
        # The serial runner reuses one policy object across a round's episodes, so
        # the episode counter alone varies the parameters. The parallel runner
        # builds a fresh policy per episode, so the counter always reads 0 there —
        # it hands the env a per-episode seed instead, which we fold in.
        self._base_seed = 1000 * self.SEED + 7919 * int(getattr(env, "rd_seed", 0) or 0)
        self._rng = np.random.default_rng(self._base_seed)
        self._handle_y0 = None

    def act(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        hand, handle = o[0:3], o[4:7]
        # At an episode start the previous-step block is a bitwise copy of the
        # current one, and the drawer is still shut. Once the sim is running the
        # two blocks never coincide exactly again.
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

    # --- shared approach helper -------------------------------------------------
    def _approach(self, hand, handle, p_align=6.0, p_drop=6.0, hover_h=0.2):
        """Return a command that lines the hand up over the handle, then drops onto it.

        Returns (delta_xyz, arrived) — `arrived` once the hand is at the hook point.
        """
        hook = handle + HANDLE_GRASP_OFFSET
        if np.linalg.norm(hand[:2] - hook[:2]) > 0.04:
            return _move(hand, hook + np.array([0.0, 0.0, hover_h]), p_align), False
        if abs(hand[2] - hook[2]) > 0.03:
            return _move(hand, hook, p_drop), False
        return _move(hand, hook, p_drop), True


class DataCollectionPolicyRewardMaximizer(_BasePolicy):
    """Maximizes the current designed reward: get the hand onto the handle and the
    drawer fully open as early as possible, then stay clamped on the handle so both
    reward terms are saturated for the rest of the episode."""

    SEED = 1

    def _on_reset(self):
        self._p_align = 8.0 + self._rng.uniform(-2.0, 4.0)
        self._pull_depth = 0.06 + self._rng.uniform(-0.01, 0.02)

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        delta, arrived = self._approach(hand, handle, p_align=self._p_align, p_drop=8.0)
        if arrived:
            if float(np.linalg.norm(handle - target)) > OPEN_TOL:
                # Still pulling: push hard past the handle to drag the drawer open.
                delta = _move(hand, hook + np.array([0.0, -self._pull_depth, 0.0]), 50.0)
            else:
                # Open: hold the hand on the handle to keep the caging term saturated.
                delta = _move(hand, hook, 6.0)
        return _act4(delta, -1.0)


class DataCollectionPolicyHoverOnHandle(_BasePolicy):
    """H1/H2: parks the hand on the handle with the fingers splayed and never pulls —
    drives the caging term to its maximum for the whole episode while the drawer never
    moves, and tests whether that out-totals trajectories that actually open it."""

    SEED = 2

    def _on_reset(self):
        self._p_align = 6.0 + self._rng.uniform(-2.0, 3.0)
        self._standoff = self._rng.uniform(0.0, 0.012)  # meters held in front of the handle

    def _step(self, o):
        hand, handle = o[0:3], o[4:7]
        park = handle + HANDLE_GRASP_OFFSET + np.array([0.0, self._standoff, 0.0])
        delta, arrived = self._approach(hand, handle, p_align=self._p_align)
        if arrived:
            delta = _move(hand, park, 5.0)
            # Back off if contact has started to drag the drawer: stay a hoverer.
            if float(handle[1]) < self._handle_y0 - 0.004:
                delta = _move(hand, park + np.array([0.0, 0.02, 0.0]), 5.0)
        return _act4(delta, -1.0)


class DataCollectionPolicyHoverAboveHandle(_BasePolicy):
    """H7: floats a few centimetres directly above the handle and never descends —
    the caging term weights height error only a third as much as sideways error, so
    this collects most of that credit while never touching the drawer at all."""

    SEED = 3

    def _on_reset(self):
        self._height = self._rng.uniform(0.05, 0.09)  # meters above the handle

    def _step(self, o):
        hand, handle = o[0:3], o[4:7]
        park = handle + np.array([0.0, 0.0, self._height])
        return _act4(_move(hand, park, 6.0), -1.0)


class DataCollectionPolicyOpenThenRetreat(_BasePolicy):
    """H3: opens the drawer, then withdraws the arm to its starting pose and leaves the
    workspace clear — the drawer ends open, but the caging term stops paying the moment
    the hand lets go."""

    SEED = 4

    def _on_reset(self):
        self._dwell = int(self._rng.integers(0, 12))  # steps held on the handle once open
        self._opened_at = None

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        is_open = float(np.linalg.norm(handle - target)) <= OPEN_TOL
        if is_open and self._opened_at is None:
            self._opened_at = self._t
        if self._opened_at is not None and self._t >= self._opened_at + self._dwell:
            # Lift clear of the handle first — the fingers sit behind it, so backing
            # straight out would drag the drawer shut again — then withdraw and hold.
            if hand[2] < handle[2] + 0.12:
                return _act4(_move(hand, hand + np.array([0.0, 0.0, 0.2]), 6.0), -1.0)
            park = np.array([handle[0], handle[1] + 0.25, handle[2] + 0.15])
            return _act4(_move(hand, park, 5.0), -1.0)
        delta, arrived = self._approach(hand, handle)
        if arrived and not is_open:
            delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
        return _act4(delta, -1.0)


class DataCollectionPolicyOpenThenClose(_BasePolicy):
    """H4: opens the drawer, dwells at the open pose to bank the per-step opening credit,
    then shoves the drawer shut again before the episode ends — the reward never checks
    the final state, so the outcome is undone but the credit is kept."""

    SEED = 5

    def _on_reset(self):
        self._dwell = int(self._rng.integers(20, 55))  # steps spent sitting at the open pose
        self._opened_at = None

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        is_open = float(np.linalg.norm(handle - target)) <= OPEN_TOL
        if is_open and self._opened_at is None:
            self._opened_at = self._t
        if self._opened_at is not None and self._t >= self._opened_at + self._dwell:
            # Push the drawer back shut.
            return _act4(_move(hand, hook + np.array([0.0, 0.12, 0.0]), 20.0), -1.0)
        delta, arrived = self._approach(hand, handle)
        if arrived and not is_open:
            delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
        return _act4(delta, -1.0)


class DataCollectionPolicySlamOpen(_BasePolicy):
    """H5/H8: rushes the handle and yanks the drawer open at full command magnitude with
    chattering, jittery motion, and keeps wrenching it past the fully-open pose — nothing
    in the reward reads the action, the handle speed, or the jerk, so violence is free."""

    SEED = 6

    def _on_reset(self):
        self._jitter = self._rng.uniform(0.2, 0.6)

    def _step(self, o):
        hand, handle = o[0:3], o[4:7]
        hook = handle + HANDLE_GRASP_OFFSET
        noise = self._rng.uniform(-self._jitter, self._jitter, size=3)
        delta, arrived = self._approach(hand, handle, p_align=30.0, p_drop=30.0)
        if arrived:
            # Keep yanking in -y at full magnitude, well past the open pose.
            delta = np.array([0.0, -1.0, 0.0]) + noise
        else:
            delta = delta + 0.3 * noise
        grip = 1.0 if (self._t // 2) % 2 == 0 else -1.0  # chatter the fingers
        return _act4(delta, grip)


class DataCollectionPolicyGripAndOpenGently(_BasePolicy):
    """H6: closes the fingers around the handle and draws the drawer open slowly and
    smoothly — the proper way to do it, but the reward reads only positions, so this
    earns nothing extra for gripping and is penalized for being slow."""

    SEED = 7

    def _on_reset(self):
        self._grip_steps = int(self._rng.integers(6, 14))  # steps spent closing the fingers
        self._pull_gain = self._rng.uniform(0.25, 0.40)  # gentle pull command magnitude
        self._closed_at = None

    def _step(self, o):
        hand, handle = o[0:3], o[4:7]
        hook = handle + HANDLE_GRASP_OFFSET
        delta, arrived = self._approach(hand, handle, p_align=4.0, p_drop=4.0)
        if not arrived:
            return _act4(delta, -1.0)
        if self._closed_at is None:
            self._closed_at = self._t
        if self._t < self._closed_at + self._grip_steps:
            # Hold still on the handle and close the fingers around it.
            return _act4(_move(hand, hook, 3.0), 1.0)
        # Gentle, steady pull with the fingers kept closed.
        return _act4(np.array([0.0, -self._pull_gain, 0.0]), 1.0)


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyHoverOnHandle,
    DataCollectionPolicyHoverAboveHandle,
    DataCollectionPolicyOpenThenRetreat,
    DataCollectionPolicyOpenThenClose,
    DataCollectionPolicySlamOpen,
    DataCollectionPolicyGripAndOpenGently,
]
