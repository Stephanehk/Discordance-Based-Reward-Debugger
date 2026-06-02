"""Trajectory-summary features for the pandemic-regulator environment.

Each feature is a scalar summary of a single trajectory. Features cover
both the desired and avoided outcomes the user named in the reward-design
template (low lockdown vs. low critical/death proportions), and they
expose how the per-step designed reward could be over- or under-weighted
(stage-4 frequency, peak vs. mean critical share, final deaths).

A trajectory is a list of (obs, action, next_obs) tuples. Each obs has
shape (history, 1, 13); we read the latest row, obs[-1, 0], where index
1 is the regulation stage and indices [3,4,5] / [8,9,10] are
[critical, dead, infected] population proportions for the alpha / delta
strains.
"""

import numpy as np


def _latest(o):
    """Return the latest length-13 observation row from an obs array."""
    return np.asarray(o)[-1, 0]


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("frac_stage_0",        "Fraction of days at stage 0 (no restrictions; economy fully open)."),
        ("frac_stage_4",        "Fraction of days at stage 4 (strictest lockdown)."),
        ("mean_stage",          "Average regulation stage over the episode (0..4)."),
        ("peak_critical",       "Peak share of the population in critical condition (alpha+delta)."),
        ("mean_critical",       "Mean per-day share of the population in critical condition (alpha+delta)."),
        ("final_dead",          "Cumulative share of the population deceased at the end of the episode (alpha+delta)."),
        ("peak_infected",       "Peak share of the population currently infected, non-critical (alpha+delta)."),
        ("final_recovered",     "Cumulative share of the population recovered at end (alpha+delta)."),
        ("outbreak_day_frac",   "Fraction of days with any critical cases present (>0)."),
        ("stage_changes",       "Number of regulation-stage changes during the episode."),
        ("episode_days",        "Number of simulated days in the trajectory."),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        stages = []
        crit = []
        dead = []
        infected = []
        recovered = []

        for (_obs, _act, next_obs) in trajectory:
            row = _latest(next_obs)
            stages.append(int(row[1]))
            crit.append(float(row[3]) + float(row[8]))
            dead.append(float(row[4]) + float(row[9]))
            infected.append(float(row[5]) + float(row[10]))
            recovered.append(float(row[7]) + float(row[12]))

        stages_arr = np.asarray(stages)
        crit_arr = np.asarray(crit)
        infected_arr = np.asarray(infected)

        n = len(trajectory)
        # Count stage changes from one step to the next.
        n_changes = int(np.sum(np.diff(stages_arr) != 0)) if n >= 2 else 0

        return [
            float(np.mean(stages_arr == 0)),         # frac_stage_0
            float(np.mean(stages_arr == 4)),         # frac_stage_4
            float(np.mean(stages_arr)),              # mean_stage
            float(np.max(crit_arr)),                 # peak_critical
            float(np.mean(crit_arr)),                # mean_critical
            float(dead[-1]),                         # final_dead (monotone, so last == max)
            float(np.max(infected_arr)),             # peak_infected
            float(recovered[-1]),                    # final_recovered
            float(np.mean(crit_arr > 0.0)),          # outbreak_day_frac
            float(n_changes),                        # stage_changes
            float(n),                                # episode_days
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(name) for name, _desc in self.feature_names)
        for (name, desc), value in zip(self.feature_names, features):
            print(f"  {name:<{width}}  {value:>8.2f}   {desc}")
