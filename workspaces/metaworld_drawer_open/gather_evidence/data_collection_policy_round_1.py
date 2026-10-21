"""Round 1 data-collection policies for metaworld_drawer_open.

Broad-coverage round for the current *telescoping endpoint* reward
(`return ~= 10 * openness(final) + 0.5 * (cage(final) - cage(initial))`). One
policy maximizes the current designed reward; the rest instantiate the exploit
hypotheses in `.repair_history/_reward_analysis.md`:

  H1 — effort/smoothness are free: SlamOpenViolent (ends open, violent) vs
       GentleGripOpen (ends open, smooth/slow).
  H2 — timing/dwell are free: DawdleThenOpenLate (fully open only at the end).
  H3 — release/creep vs. task achieved: OpenThenRetreat (opens, then lets go).
  H4 — no grasp required: ShoveOpenNoGrasp (pushes the drawer open, never grips).

Observation slices: hand = obs[0:3], aperture = obs[3], handle = obs[4:7],
fully-open target = obs[39:42], hand-at-reset = obs[42:45].
"""

import numpy as np

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
        handle = o[4:7]
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

    def _approach(self, hand, handle, p_align=6.0, p_drop=6.0, hover_h=0.2):
        """Line the hand up over the handle, then drop onto it.

        Returns (delta_xyz, arrived) — `arrived` once the hand is at the hook point.
        """
        hook = handle + HANDLE_GRASP_OFFSET
        if np.linalg.norm(hand[:2] - hook[:2]) > 0.04:
            return _move(hand, hook + np.array([0.0, 0.0, hover_h]), p_align), False
        if abs(hand[2] - hook[2]) > 0.03:
            return _move(hand, hook, p_drop), False
        return _move(hand, hook, p_drop), True


class DataCollectionPolicyRewardMaximizer(_BasePolicy):
    """Maximizes the current designed reward: line up on the handle and yank the
    drawer fully open as early as possible, then stay clamped on the handle so the
    final openness is 1 and the caging change is fully banked."""

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
                delta = _move(hand, hook + np.array([0.0, -self._pull_depth, 0.0]), 50.0)
            else:
                delta = _move(hand, hook, 6.0)  # hold clamped on the handle
        return _act4(delta, -1.0)


class DataCollectionPolicySlamOpenViolent(_BasePolicy):
    """H1 (T1/action): rushes the handle and yanks the drawer open at full command
    magnitude with heavy jitter and finger chatter, wrenching well past the open
    pose. Ends fully open (full opening credit) while the motion is maximally
    violent — nothing in the reward reads the action, speed, or jerk."""

    SEED = 2

    def _on_reset(self):
        self._jitter = self._rng.uniform(0.3, 0.6)

    def _step(self, o):
        hand, handle = o[0:3], o[4:7]
        noise = self._rng.uniform(-self._jitter, self._jitter, size=3)
        delta, arrived = self._approach(hand, handle, p_align=30.0, p_drop=30.0)
        if arrived:
            delta = np.array([0.0, -1.0, 0.0]) + noise  # yank in -y at full magnitude
        else:
            delta = delta + 0.3 * noise
        grip = 1.0 if (self._t // 2) % 2 == 0 else -1.0  # chatter the fingers
        return _act4(delta, grip)


class DataCollectionPolicyGentleGripOpen(_BasePolicy):
    """H1 contrast (T1/action): closes the fingers around the handle and draws the
    drawer open slowly and smoothly. The 'proper' way to do it — but the reward
    reads only positions, so it earns nothing extra for the grasp or the smoothness
    and may not fully finish within the horizon at this gentle pace."""

    SEED = 3

    def _on_reset(self):
        self._grip_steps = int(self._rng.integers(6, 14))
        self._pull_gain = self._rng.uniform(0.20, 0.35)  # gentle pull magnitude
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
            return _act4(_move(hand, hook, 3.0), 1.0)  # close fingers, hold still
        return _act4(np.array([0.0, -self._pull_gain, 0.0]), 1.0)  # steady gentle pull


class DataCollectionPolicyDawdleThenOpenLate(_BasePolicy):
    """H2 (T1 endpoint-only): loiters away from the handle for most of the episode,
    then rushes in and yanks the drawer fully open only in the final stretch. Ends
    fully open (full opening credit, same as opening immediately) yet the drawer is
    open for only a handful of steps — timing and dwell are invisible to the reward."""

    SEED = 4

    def _on_reset(self):
        # Horizon is 150; start the opening push late enough to dawdle, early
        # enough to still reach full open by the end.
        self._open_start = int(self._rng.integers(95, 112))
        self._wander_dir = self._rng.uniform(-1.0, 1.0, size=3)
        self._wander_dir[2] = abs(self._wander_dir[2])  # bias upward, stay clear

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        if self._t < self._open_start:
            # Drift around above/away from the handle without touching it.
            noise = self._rng.uniform(-0.2, 0.2, size=3)
            return _act4(np.clip(0.3 * self._wander_dir + noise, -1.0, 1.0), -1.0)
        delta, arrived = self._approach(hand, handle, p_align=30.0, p_drop=30.0)
        if arrived and float(np.linalg.norm(handle - target)) > OPEN_TOL:
            delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
        return _act4(delta, -1.0)


class DataCollectionPolicyOpenThenRetreat(_BasePolicy):
    """H3 (T1 release/creep): opens the drawer, dwells briefly, then withdraws the
    arm to a clear pose and holds it there. The released drawer creeps back a few
    cm, so the final openness — and thus the reward — is docked relative to staying
    clamped, even though the drawer was fully opened."""

    SEED = 5

    def _on_reset(self):
        self._dwell = int(self._rng.integers(0, 12))
        self._opened_at = None

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        is_open = float(np.linalg.norm(handle - target)) <= OPEN_TOL
        if is_open and self._opened_at is None:
            self._opened_at = self._t
        if self._opened_at is not None and self._t >= self._opened_at + self._dwell:
            # Lift clear first (fingers sit behind the handle; backing straight out
            # would drag it shut), then withdraw and hold clear.
            if hand[2] < handle[2] + 0.12:
                return _act4(_move(hand, hand + np.array([0.0, 0.0, 0.2]), 6.0), -1.0)
            park = np.array([handle[0], handle[1] + 0.25, handle[2] + 0.15])
            return _act4(_move(hand, park, 5.0), -1.0)
        delta, arrived = self._approach(hand, handle)
        if arrived and not is_open:
            delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
        return _act4(delta, -1.0)


class DataCollectionPolicyShoveOpenNoGrasp(_BasePolicy):
    """H4 (T1 proxy, no grasp): approaches with the fingers held open and shoves the
    drawer open by pushing the handle from behind — never closing a grasp. Ends
    open (full opening credit) though the handle was never actually gripped; the
    bounded caging term cannot require a real grasp."""

    SEED = 6

    def _step(self, o):
        hand, handle, target = o[0:3], o[4:7], o[39:42]
        hook = handle + HANDLE_GRASP_OFFSET
        delta, arrived = self._approach(hand, handle, p_align=8.0, p_drop=8.0)
        if arrived and float(np.linalg.norm(handle - target)) > OPEN_TOL:
            delta = _move(hand, hook + np.array([0.0, -0.06, 0.0]), 50.0)
        return _act4(delta, -1.0)  # gripper held open the whole time


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicySlamOpenViolent,
    DataCollectionPolicyGentleGripOpen,
    DataCollectionPolicyDawdleThenOpenLate,
    DataCollectionPolicyOpenThenRetreat,
    DataCollectionPolicyShoveOpenNoGrasp,
]
