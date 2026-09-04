"""Round 1 data-collection policies for metaworld_hammer.

Broad-coverage round. One policy maximizes the current designed reward; the rest
instantiate the exploit hypotheses in `.repair_history/_reward_analysis.md`
(H1-H8): caging the handle without ever lifting it, under-driving the nail and
holding, holding against a nail that never moves, succeeding then putting the
tool down, sinking the nail by hand with no hammer, slamming with a loose grip,
and carrying the hammer tilted.

All the hammer-wielding policies share one reliable pick-up-and-strike
controller (`_BasePolicy._drive`): approach the handle from above, close the
fingers, swing into the strike plane, then push along +y. Policies differ in
what they do once the hammer is in hand — that difference is the hypothesis.

Observation slices used: hand = obs[0:3], aperture = obs[3], hammer = obs[4:7],
hammer quat = obs[7:11], nail head = obs[11:14], target = obs[39:42],
tcp = obs[45:48], nail slide depth = obs[57].
"""

import numpy as np

# Hand pose at which a held hammer's head meets the nail; pushing further in +y
# drives the nail into the wall.
STRIKE_POS = np.array([0.05, 0.71, 0.16])  # meters
GRASP_OFFSET = np.array([-0.04, 0.0, -0.01])  # meters; grasp point on the handle
HOME_POS = np.array([0.0, 0.4, 0.2])  # meters; the arm's neutral pose at reset
SUCCESS_DEPTH = 0.09  # meters of nail travel that counts as fully driven in
GRIP_CLOSE = 0.8  # gripper command that holds the hammer
GRIP_OPEN = -1.0  # gripper command that releases it


def _move(cur, to, p):
    """Proportional end-effector command toward `to`, clipped to the action range."""
    return np.clip(p * (np.asarray(to) - np.asarray(cur)), -1.0, 1.0)


def _act4(delta_xyz, grip):
    a = np.zeros(4, dtype=np.float32)
    a[:3] = np.clip(delta_xyz, -1.0, 1.0)
    a[3] = grip
    return a


class _BasePolicy:
    """Shared episode bookkeeping and the pick-up-and-strike controller."""

    SEED = 0
    GRASP = GRASP_OFFSET  # subclasses may grip the handle somewhere else

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

    def _new_episode(self):
        """Override to reset per-episode state."""

    def act(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        # At an episode start the previous-step block is a bitwise copy of the
        # current one and the nail has not moved. Once the sim is running the two
        # blocks never coincide exactly again.
        if float(o[57]) < 1e-9 and np.array_equal(o[0:18], o[18:36]):
            self._episode += 1
            self._t = 0
            self._rng = np.random.default_rng(self._base_seed + 131 * self._episode)
            self._p = float(self._rng.uniform(9.0, 11.0))  # proportional gain
            self._jit = self._rng.normal(0.0, 0.005, size=3)  # meters of goal jitter
            self._new_episode()
        a = self._step(o)
        self._t += 1
        return np.asarray(a, dtype=np.float32)

    def _drive(self, o, goal=None, y_rate=1.0):
        """Pick the hammer up and push it into the nail.

        `y_rate` caps the forward push, so a policy can creep in slowly enough to
        stop the nail at a chosen depth. Returns a 4-vector action.
        """
        hand, hammer = o[0:3], o[4:7]
        grasp = hammer + self.GRASP
        far_xy = float(np.linalg.norm(hand[:2] - grasp[:2])) > 0.04
        dz = abs(float(hand[2] - grasp[2]))
        # Keep the fingers open until they are actually down around the handle.
        # Closing early clamps them shut above the hammer, and the descent then
        # just pins it to the table instead of enveloping the handle.
        approach_grip = 0.0 if (far_xy or dz > 0.1) else GRIP_CLOSE
        if far_xy:
            # Off in the horizontal plane — hover above the handle.
            return _act4(_move(hand, grasp + np.array([0.0, 0.0, 0.1]), self._p), approach_grip)
        if dz > 0.05 and float(grasp[2]) < 0.03:
            # Lined up above the handle and it is still on the table — drop onto
            # it. The height guard matters: once the hammer has come off the
            # table, stop pressing down or the arm just pins it there.
            return _act4(_move(hand, grasp + np.array([0.0, 0.0, 0.03]), self._p), approach_grip)
        g = (STRIKE_POS if goal is None else goal) + self._jit
        if float(np.linalg.norm(hand[[0, 2]] - g[[0, 2]])) > 0.02:
            # Hammer in hand — swing into the strike plane without advancing yet.
            return _act4(_move(hand, np.array([g[0], hand[1], g[2]]), self._p), GRIP_CLOSE)
        delta = _move(hand, g, self._p)
        delta[1] = min(float(delta[1]), y_rate)  # cap the forward push
        return _act4(delta, GRIP_CLOSE)


class DataCollectionPolicyRewardMaximizer(_BasePolicy):
    """Maximizes the current designed reward: drive the nail in as early as possible, then keep the hammer pressed hard against it so the per-step success override keeps re-firing."""

    SEED = 1

    def _step(self, o):
        # Settle exactly on the strike pose and stay there: the hammer remains
        # loaded against the nail, so the shaped sum stays above the override's
        # gate and the 10/step keeps paying for the rest of the episode.
        return self._drive(o, goal=STRIKE_POS)


class DataCollectionPolicyCageWithoutLifting(_BasePolicy):
    """H1 (reward_grab): brackets the fingers around the hammer handle and squeezes for the whole episode without ever lifting it — drives the grasp proxy high while the hammer is never wielded."""

    SEED = 2

    def _step(self, o):
        hand, hammer = o[0:3], o[4:7]
        # Settle the pads around the handle at its own resting height and squeeze.
        grasp = hammer + np.array([-0.04, 0.0, 0.0]) + np.array([self._jit[0], self._jit[1], 0.0])
        if float(np.linalg.norm(hand[:2] - grasp[:2])) > 0.04:
            return _act4(_move(hand, grasp + np.array([0.0, 0.0, 0.08]), self._p), GRIP_OPEN)
        delta = _move(hand, grasp, self._p)
        delta[2] = min(float(delta[2]), 0.0)  # never command a lift — the hammer stays put
        return _act4(delta, GRIP_CLOSE)


class _DriveToDepthAndHold(_BasePolicy):
    """Push the nail in until it reaches DEPTH, then freeze the arm and hold there.

    Approaches at full speed until the hammer head is nearly touching the nail,
    then creeps, so the nail can be stopped at a chosen depth without overshooting.
    """

    DEPTH = 0.05  # meters of nail travel to stop at
    CREEP = 0.10  # capped forward push once the hammer head is on the nail

    def _new_episode(self):
        self._hold = None  # hand pose to freeze at, once the depth is reached

    def _step(self, o):
        hand, nail = o[0:3], float(o[57])
        if self._hold is not None:
            # Hold station: keep the hammer head parked exactly where it stopped.
            return _act4(_move(hand, self._hold, self._p), GRIP_CLOSE)
        if nail >= self.DEPTH:
            self._hold = hand.copy()
            return _act4(np.zeros(3), GRIP_CLOSE)
        # Close the gap quickly, then ease off so the nail advances in small
        # increments and we can stop it exactly where we want.
        head_to_nail = float(np.linalg.norm((o[4:7] + np.array([0.16, 0.06, 0.0])) - o[11:14]))
        y_rate = 1.0 if head_to_nail > 0.05 else self.CREEP
        return self._drive(o, y_rate=y_rate)


class DataCollectionPolicyUnderDriveAndHold(_DriveToDepthAndHold):
    """H2 (reward_in_place vs the success override): drives the nail only halfway — deliberately stopping short of the success line — then holds the hammer pressed there, farming the dense per-step proximity credit instead of finishing the task."""

    SEED = 3
    DEPTH = 0.05
    CREEP = 0.10


class DataCollectionPolicyHoldAgainstUndrivenNail(_DriveToDepthAndHold):
    """H3 (reward_in_place): carries the hammer head up against the nail and holds it there without the nail ever moving — banks dense proximity credit across an episode of zero task progress."""

    SEED = 4
    DEPTH = 0.003
    CREEP = 0.02


class DataCollectionPolicySucceedThenDisengage(_BasePolicy):
    """H4 (success-override gate): drives the nail fully in, then releases the hammer and retracts the arm to its neutral pose — the tidy finish that the re-gated override stops paying for."""

    SEED = 5

    def _new_episode(self):
        self._done = False

    def _step(self, o):
        hand, nail = o[0:3], float(o[57])
        if nail > SUCCESS_DEPTH + 0.005:
            self._done = True
        if self._done:
            # Task complete: let go of the hammer and go back to rest.
            return _act4(_move(hand, HOME_POS, self._p), GRIP_OPEN)
        return self._drive(o)


class DataCollectionPolicyNailByHandNoHammer(_BasePolicy):
    """H5 (all terms): ignores the hammer entirely and pushes the nail into the wall with the closed gripper — the outcome the user asked for, reached by a route the reward pays nothing for."""

    SEED = 6

    def _step(self, o):
        hand, nail_head = o[0:3], o[11:14]
        # Line up on the nail head with the fingers pinched shut, then push in +y.
        approach = nail_head + np.array([-0.02, -0.06, 0.0]) + self._jit
        if float(np.linalg.norm(hand - approach)) > 0.03:
            return _act4(_move(hand, approach, self._p), GRIP_CLOSE)
        return _act4(_move(hand, nail_head + np.array([-0.02, 0.12, 0.0]), self._p), GRIP_CLOSE)


class DataCollectionPolicyViolentLooseSlam(_BasePolicy):
    """H6/H8 (missing penalty + override gate): grabs the hammer and hurls it at the nail with maximum-magnitude jittery strokes, opening the fingers on impact — nothing penalizes the violence, and a nail sunk with a loose grip can miss the override's gate."""

    SEED = 7

    def _new_episode(self):
        self._armed = False  # True once the hammer is in hand and we can swing

    def _step(self, o):
        hand, hammer = o[0:3], o[4:7]
        if not self._armed:
            # Pick the hammer up normally first.
            if float(hammer[2]) > 0.05 and float(np.linalg.norm(o[45:48] - hammer)) < 0.07:
                self._armed = True
            return self._drive(o, goal=STRIKE_POS + np.array([0.0, -0.12, 0.10]))
        # Wind up behind the nail, then whip forward at saturated command
        # magnitude, releasing the grip mid-strike.
        phase = self._t % 24
        jitter = self._rng.normal(0.0, 0.25, size=3)
        if phase < 14:
            windup = STRIKE_POS + np.array([0.0, -0.12, 0.10]) + self._jit
            return _act4(_move(hand, windup, self._p) + jitter, GRIP_CLOSE)
        swing = STRIKE_POS + np.array([0.0, 0.10, -0.02])
        drive = _move(hand, swing, 30.0) + jitter  # saturated: full-speed strike
        grip = GRIP_CLOSE if phase < 18 else GRIP_OPEN  # let go mid-swing
        return _act4(drive, grip)


class DataCollectionPolicyTiltedCarry(_BasePolicy):
    """H7 (orientation-blind head model): grips the hammer far off-centre so it hangs tilted in the fingers, then carries it to the nail — the reward keeps locating the head at a fixed offset from the body while the real head swings elsewhere."""

    SEED = 8
    GRASP = np.array([0.06, 0.0, -0.01])  # grip near the head end, not the middle

    def _step(self, o):
        # Lift high and carry across, so the off-centre grip lets the hammer tip.
        return self._drive(o, goal=STRIKE_POS + np.array([0.0, 0.02, 0.07]))


POLICIES = [
    DataCollectionPolicyRewardMaximizer,
    DataCollectionPolicyCageWithoutLifting,
    DataCollectionPolicyUnderDriveAndHold,
    DataCollectionPolicyHoldAgainstUndrivenNail,
    DataCollectionPolicySucceedThenDisengage,
    DataCollectionPolicyNailByHandNoHammer,
    DataCollectionPolicyViolentLooseSlam,
    DataCollectionPolicyTiltedCarry,
]
