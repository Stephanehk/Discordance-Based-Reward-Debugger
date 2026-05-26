"""Round 1 data-collection policies for the pandemic-control env.

Each policy is a hand-coded heuristic over the 5 regulation stages
(0 = no restrictions ... 4 = strictest lockdown). The set is designed to
produce trajectories that vary along the recorded trajectory features
(mean_stage, max_stage, frac_days_open, frac_days_strict, stage_switches,
peak_critical, mean_infected, final_recovered) so that disagreements
between the designed reward and the stakeholder metrics become visible.

obs is a numpy array of shape (history, 1, 13); use np.asarray(obs)[-1, 0]
to get the latest length-13 state vector. Combined critical proportion is
state[3] + state[8].
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across alpha + delta strains."""
    return float(state[3] + state[8])


def _infected(state):
    """Combined currently-infected (non-critical) proportion."""
    return float(state[5] + state[10])


class DataCollectionPolicyCurrentRewardMaximizer:
    """Approximate argmax of the designed reward: stay open, escalate only when critical proportion is rising, never use stage 4, minimize stage changes."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.prev_critical = 0.0
        self.current_stage = 0

    def act(self, obs):
        state = _latest(obs)
        cur_critical = _critical(state)

        # W_CRIT_INCREASE = 10000 dominates: if critical is rising, escalate
        # to suppress further increase. Otherwise, the reward strongly prefers
        # stage 0 (avoids both the low-critical-stage penalty and the
        # stage-4 penalty, and minimizes stage-change churn).
        rising = cur_critical > self.prev_critical + 1e-6

        if rising:
            # Pick stage 2 if critical is small but rising, 3 if larger.
            # Avoid stage 4 — the reward explicitly penalizes it (-10/step).
            if cur_critical >= 0.005:
                target = 3
            else:
                target = 2
        else:
            # Not rising. The reward prefers stage 0 whenever possible. The
            # low-critical-stage penalty (-5) also pushes toward stage 0 when
            # critical < 0.01. Reverting to 0 incurs a one-time -1 churn cost
            # but saves -5 every subsequent step it would have been > 0.
            target = 0

        self.prev_critical = cur_critical
        self.current_stage = target
        return int(target)


class DataCollectionPolicyAlwaysOpen:
    """High openness and education-access, untested hospital-capacity: keep stage 0 for the entire episode regardless of outbreak."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 0


class DataCollectionPolicyAlwaysStrict:
    """High hospital-capacity, low openness/education: hold stage 4 throughout — the designed reward should dislike both the stage-4 penalty and the low-critical-stage penalty here."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        return 4


class DataCollectionPolicyReactiveLockdown:
    """Ramps stage with current critical proportion; produces many stage switches and frequent stage 4 — exercises the churn and max-stage penalties while scoring well on hospital-capacity."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        state = _latest(obs)
        cur_critical = _critical(state)
        # Scale: 0 critical -> stage 0; 0.02+ critical -> stage 4.
        # 0.005 -> 1; 0.01 -> 2; 0.015 -> 3; 0.02 -> 4.
        stage = int(np.clip(int(cur_critical * 200), 0, 4))
        return stage


class DataCollectionPolicyPeriodicCycle:
    """Cycles stages on a fixed schedule (20 days open, 20 days at stage 3); probes the stage-change churn penalty and the predictability stakeholder."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.step = 0

    def act(self, obs):
        # 40-day cycle: 20 days at stage 0, then 20 days at stage 3.
        phase = self.step % 40
        action = 0 if phase < 20 else 3
        self.step += 1
        return int(action)


POLICIES = [
    DataCollectionPolicyCurrentRewardMaximizer,
    DataCollectionPolicyAlwaysOpen,
    DataCollectionPolicyAlwaysStrict,
    DataCollectionPolicyReactiveLockdown,
    DataCollectionPolicyPeriodicCycle,
]
