"""Trajectory features for the Pandemic Simulator.

Summarizes a rollout of daily regulation decisions so a reward designer can
see, at a glance, how locked-down the policy was, how stable it was, and how
the outbreak played out (infections, critical load, deaths). Features cover
each outcome named in the reward design (keep critical load low, keep lockdown
low, avoid deaths, avoid fluctuating levels) and also expose how the designed
reward could be optimized (e.g. a policy that keeps critical load low while
letting mild infections run wide still scores well, since the reward only
penalizes critical cases and deaths).

Each transition's `next_obs[-1, 0]` is the length-13 state vector documented
in env_config.OBS_DESCRIPTION:
  [1] stage (0..4),
  alpha = [3]critical [4]dead [5]infected [6]none [7]recovered,
  delta = [8]critical [9]dead [10]infected [11]none [12]recovered.
Category fields are proportions of the population (each strain sums to 1).
"""

import numpy as np


def _row(o):
    """Latest state row obs[-1, 0] as a length-13 numpy vector."""
    return np.asarray(o)[-1, 0]


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("episode_days", "Number of simulated days in the rollout"),
        ("mean_stage", "Average regulation stage in force (0=open .. 4=lockdown)"),
        ("peak_stage", "Strictest regulation stage reached"),
        ("frac_full_lockdown", "Fraction of days at full lockdown (stage 4)"),
        ("frac_open", "Fraction of days with no restrictions (stage 0)"),
        ("mean_stage_change", "Mean day-to-day change in stage (fluctuation)"),
        ("mean_critical", "Mean critical-case fraction (alpha+delta)"),
        ("peak_critical", "Peak critical-case fraction (alpha+delta)"),
        ("peak_infected", "Peak active-infection fraction (alpha+delta)"),
        ("mean_infected", "Mean active-infection fraction (alpha+delta)"),
        ("final_death_fraction", "Cumulative death fraction at episode end (alpha+delta)"),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        stages, infected, critical = [], [], []
        stage_changes = []
        last_dead = 0.0
        for obs, act, next_obs in trajectory:
            prev = _row(obs)
            cur = _row(next_obs)
            stages.append(float(cur[1]))
            stage_changes.append(abs(float(cur[1]) - float(prev[1])))
            critical.append(float(cur[3]) + float(cur[8]))
            infected.append(float(cur[5]) + float(cur[10]))
            last_dead = float(cur[4]) + float(cur[9])

        stages = np.array(stages)
        return [
            float(len(trajectory)),                 # episode_days
            float(np.mean(stages)),                 # mean_stage
            float(np.max(stages)),                  # peak_stage
            float(np.mean(stages == 4)),            # frac_full_lockdown
            float(np.mean(stages == 0)),            # frac_open
            float(np.mean(stage_changes)),          # mean_stage_change
            float(np.mean(critical)),               # mean_critical
            float(np.max(critical)),                # peak_critical
            float(np.max(infected)),                # peak_infected
            float(np.mean(infected)),               # mean_infected
            float(last_dead),                       # final_death_fraction
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(name) for name, _ in self.feature_names)
        for (name, desc), val in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {val:8.2f}   {desc}")
