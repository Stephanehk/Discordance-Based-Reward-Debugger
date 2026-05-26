"""Round 2 data-collection policies for the pandemic-control env.

The designed reward changed between rounds: the old change-based critical
term has been replaced by a quadratic level-based term
(``-W_CRIT_LEVEL * cur_critical**2`` with W_CRIT_LEVEL = 1_000_000), which
dominates every other term. Even cur_critical = 0.005 yields ~-25/step,
i.e. ~-3000 over a 120-day episode. So under the new reward, a high-reward
policy must keep critical proportion extremely low across the whole
episode — not merely suppress its rate of change.

Round 1 already covered: always-open, always-stage-4, a reactive ramp
(stage = int(critical*200), clipped), a 20/20 blind cycle on stages 0/3,
and a maximizer rooted in the OLD reward (escalates only when critical is
rising). Round 2 targets the new reward's quirks:

  * V2 maximizer: preemptive stage-2 prophylaxis from day 1 (suppresses
    outbreak before it grows; never uses stage 4) — the new optimal.
  * Smooth stable lockdown: fixed stage 2 forever. Same outbreak-control
    benefit as V2 but without the bet that early action is needed; the
    new reward should *like* this if it stays low-critical and *dislike*
    it via the low-critical-stage penalty when critical never rises.
  * Late-only escalator: stay at stage 0 until day 90, then jump to
    stage 3 — probes whether the level-based term rewards procrastinating
    on the stage-cost side at the expense of a late critical-load spike.

obs is a numpy array of shape (history, 1, 13); use np.asarray(obs)[-1, 0]
to get the latest length-13 state vector. Combined critical proportion
is state[3] + state[8].
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across alpha + delta strains."""
    return float(state[3] + state[8])


class DataCollectionPolicyCurrentRewardMaximizerV2:
    """Approximate argmax of the NEW reward (level-based critical penalty dominant): preemptive stage-2 prophylaxis from day 1, escalate to 3 if critical rises, drop only if critical truly stays at 0 for long; never stage 4."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        # Days observed with cur_critical exactly 0 in a row.
        self._zero_streak = 0

    def act(self, obs):
        state = _latest(obs)
        cur_critical = _critical(state)

        # The quadratic level penalty dominates: at cur_critical = 0.005 the
        # per-step penalty is ~-25, vs the W_LOW_CRITICAL_STAGE penalty of
        # -5/step for being at stage > 0 with critical below 0.01, and the
        # one-time -1 churn penalty per stage change. So suppressing
        # critical pays for the stage cost many times over.
        if cur_critical >= 0.005:
            # Already meaningfully critical — escalate hard, but never to
            # stage 4 (which adds -10/step W_MAX_STAGE on top of the rest).
            target = 3
        elif cur_critical > 0.0:
            # Some critical present — preemptive mid-strength lockdown.
            target = 2
        else:
            # cur_critical == 0. Most of the time we still want a low
            # prophylactic stage to keep it from rising — the quadratic
            # term punishes any rise far more than W_LOW_CRITICAL_STAGE
            # punishes a low nonzero stage. Only drop to 0 after a long
            # confirmed zero streak (where the W_LOW_CRITICAL_STAGE
            # penalty would compound and there's nothing to suppress).
            self._zero_streak += 1
            if self._zero_streak >= 30:
                target = 0
            else:
                target = 2

        if cur_critical > 0.0:
            self._zero_streak = 0

        return int(target)


class DataCollectionPolicySmoothStableLockdown:
    """Fixed stage 2 for the entire episode: prophylactic without churn or stage 4 — civic-trust and hospital-capacity should like it; the new reward penalises it via the low-critical-stage term whenever critical stays below 0.01."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 2


class DataCollectionPolicyLateOnlyEscalator:
    """Stage 0 until day 90, then jump to stage 3 for the remainder: probes whether the level-based critical penalty over-credits early stage-0 openness at the cost of a late critical-load spike."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.step = 0

    def act(self, obs):
        # Days 0..89: stay fully open. Days 90+: hold stage 3.
        action = 0 if self.step < 90 else 3
        self.step += 1
        return int(action)


POLICIES = [
    DataCollectionPolicyCurrentRewardMaximizerV2,
    DataCollectionPolicySmoothStableLockdown,
    DataCollectionPolicyLateOnlyEscalator,
]
