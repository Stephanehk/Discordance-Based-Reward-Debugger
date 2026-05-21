"""Trajectory summary features for the pandemic-control environment.

Each trajectory is a list of (obs, action, next_obs) tuples. obs/next_obs are
numpy arrays of shape (history, 1, 13); the latest state is np.asarray(obs)[-1, 0].
Features summarize how the public-health authority used the regulation stages
over the episode and how the outbreak (infections and critical cases) evolved.
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector from an obs of shape (history,1,13)."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across both strains: alpha + delta critical."""
    return state[3] + state[8]


def _infected(state):
    """Combined currently-infected (non-critical) proportion across both strains."""
    return state[5] + state[10]


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("n_days", "Number of simulated days (transitions) in the trajectory."),
        ("mean_stage", "Mean regulation stage imposed over the episode (0=none .. 4=strictest)."),
        ("max_stage", "Strictest regulation stage imposed at any point in the episode."),
        ("frac_days_open", "Fraction of days with no/light restrictions (stage <= 1)."),
        ("frac_days_strict", "Fraction of days under strict lockdown (stage >= 3)."),
        ("stage_switches", "Number of days the imposed stage changed from the previous day."),
        ("mean_infected", "Mean combined currently-infected proportion over the episode."),
        ("peak_infected", "Peak combined currently-infected proportion reached during the episode."),
        ("mean_critical", "Mean combined critical-case proportion over the episode."),
        ("peak_critical", "Peak combined critical-case proportion reached during the episode."),
        ("final_critical", "Combined critical-case proportion at the end of the episode."),
        ("final_recovered", "Combined recovered proportion at the end of the episode (alpha+delta)."),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        if len(trajectory) == 0:
            return [0.0] * len(self.feature_names)

        # Stage chosen each day = the integer action.
        stages = np.array([float(act) for (_obs, act, _next) in trajectory], dtype=float)

        # Outbreak series taken from next_obs of each transition (the state the
        # day's regulation produced), in episode order.
        next_states = [_latest(next_obs) for (_obs, _act, next_obs) in trajectory]
        infected = np.array([_infected(s) for s in next_states], dtype=float)
        critical = np.array([_critical(s) for s in next_states], dtype=float)
        final = next_states[-1]
        final_recovered = float(final[7] + final[12])

        n_days = float(len(trajectory))
        mean_stage = float(np.mean(stages))
        max_stage = float(np.max(stages))
        frac_days_open = float(np.mean(stages <= 1))
        frac_days_strict = float(np.mean(stages >= 3))
        stage_switches = float(np.sum(np.diff(stages) != 0)) if len(stages) > 1 else 0.0

        mean_infected = float(np.mean(infected))
        peak_infected = float(np.max(infected))
        mean_critical = float(np.mean(critical))
        peak_critical = float(np.max(critical))
        final_critical = float(_critical(final))

        return [
            n_days,
            mean_stage,
            max_stage,
            frac_days_open,
            frac_days_strict,
            stage_switches,
            mean_infected,
            peak_infected,
            mean_critical,
            peak_critical,
            final_critical,
            final_recovered,
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(short) for short, _desc in self.feature_names)
        for (short, desc), value in zip(self.feature_names, features):
            print(f"  {short:<{width}}  {float(value):>8.2f}   {desc}")
