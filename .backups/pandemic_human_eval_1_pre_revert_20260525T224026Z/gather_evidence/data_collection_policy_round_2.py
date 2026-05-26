"""Round-2 data-collection policies for the pandemic-control environment.

Round 1 already collected an always-open baseline, an always-strict baseline,
a day-by-day whiplash policy, and a maximizer of the earlier designed reward
(which had no per-step illness or churn term). The reward has since been
revised to add (a) a per-step penalty proportional to active illness load
(infected + critical) and (b) a per-step penalty when the regulation stage
changes from the previous day. The strictest-stage penalty (stage == 4) is
unchanged, as is the over-regulate penalty for stage > 0 while critical is
very low.

Round 2 probes loopholes the revised reward still leaves open. The illness
and churn terms close the most obvious flaw from round 1 (no per-step cost
of letting the outbreak burn), but they also open new ones: any stage in
{1, 2, 3} held *steadily* costs zero churn and zero stage-4 penalty, so a
policy that just parks at a moderate-or-high stage can score very well on
the revised reward while crushing economic and schooling outcomes; and a
policy that responds reactively to outbreak conditions can end up paying
more churn cost than a flat policy that happens to land near the right
stage on average. These are the axes the candidate stakeholder metrics
should disagree on.
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector from an obs of shape (history, 1, 13)."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across both strains (alpha + delta)."""
    return float(state[3]) + float(state[8])


class DataCollectionPolicyRevisedRewardMaximizer:
    """Best-effort maximizer of the revised reward: stay at stage 0 when critical < 0.01, otherwise escalate (capped at 3) with hysteresis to avoid churn."""

    # Required per-round maximizer. The revised reward is dominated by the
    # W_ILL=80 per-step illness penalty, so this policy escalates as soon as
    # critical reaches the W_OVERREGULATE threshold; caps at 3 to dodge the
    # W_STAGE4 penalty; uses a deadband on critical and a minimum dwell on
    # the current stage to minimize the new W_CHURN penalty.
    HIGH = 0.015                # critical above this -> consider raising stage
    LOW = 0.008                 # critical below this -> consider lowering stage
    MIN_DWELL_STEPS = 3         # don't change stage within this many steps of the last change

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._cur_stage = 0
        self._dwell = 0

    def act(self, obs):
        state = _latest(obs)
        critical = _critical(state)
        target = self._cur_stage

        # Below the over-regulate cutoff, the reward strictly prefers stage 0.
        if critical < 0.01:
            target = 0
        else:
            # critical >= 0.01: pick a target stage roughly proportional to
            # how far above the threshold we are, capped at 3 to dodge W_STAGE4.
            if critical >= 0.03:
                target = 3
            elif critical >= 0.02:
                target = 2
            else:
                target = 1

        # Hysteresis: only commit to a change if (a) the deadband is exceeded
        # in the relevant direction, and (b) we've held the current stage for
        # at least MIN_DWELL_STEPS. Both keep W_CHURN down.
        if target > self._cur_stage and critical < self.HIGH:
            target = self._cur_stage
        if target < self._cur_stage and critical > self.LOW:
            target = self._cur_stage
        if self._dwell < self.MIN_DWELL_STEPS and target != self._cur_stage:
            target = self._cur_stage

        if target != self._cur_stage:
            self._cur_stage = target
            self._dwell = 0
        else:
            self._dwell += 1

        return int(max(0, min(3, self._cur_stage)))


class DataCollectionPolicyStage3Hover:
    """Permanent stage 3: never switches and never enters stage 4, exploiting the loophole that the churn and stage-4 penalties both score this as cheap."""

    # This policy is the "stage 3 forever sweet spot": zero stage_switches,
    # never triggers W_STAGE4, never triggers W_CHURN. The economy and
    # schools stakeholders should strongly disagree with rewarding this
    # heavily, since schools close at stage >= 2 and the entire episode is
    # spent at high restriction.
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 3


class DataCollectionPolicyReactiveJittery:
    """Reactive small-step adjustments: every day nudge the stage by +/-1 toward a critical-load target, accepting heavy churn for tight outbreak tracking."""

    # Probes the new W_CHURN term: this policy stays close to a reasonable
    # stage for the outbreak it sees, but it pays churn on nearly every step
    # because it adjusts by +/-1 each day rather than holding. The
    # epidemiologist may prefer it to a flat policy; the revised reward
    # should disprefer it relative to e.g. Stage3Hover even when the
    # outbreak burden is comparable.
    TARGET_CRITICAL = 0.01

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._cur_stage = 0

    def act(self, obs):
        state = _latest(obs)
        critical = _critical(state)
        # Sign of the deviation from the target picks the direction.
        if critical > self.TARGET_CRITICAL:
            self._cur_stage += 1
        elif critical < self.TARGET_CRITICAL:
            self._cur_stage -= 1
        # Clamp to {0, 1, 2, 3}; refuse stage 4 to keep the difference from
        # the maximizer one-dimensional (this policy isolates the churn axis,
        # not the stage-4 axis).
        self._cur_stage = max(0, min(3, self._cur_stage))
        return int(self._cur_stage)


POLICIES = [
    DataCollectionPolicyRevisedRewardMaximizer,
    DataCollectionPolicyStage3Hover,
    DataCollectionPolicyReactiveJittery,
]
