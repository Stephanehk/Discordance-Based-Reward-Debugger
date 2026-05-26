"""Candidate performance metrics for the pandemic-control environment.

Each metric encodes a single stakeholder's view of what makes a trajectory
good or bad. Higher = better, by that stakeholder's lights. Each function
returns a single float aggregating a trajectory of (obs, action, next_obs)
tuples; trajectories are not assumed to be non-empty.

Observation layout (latest row = np.asarray(obs)[-1, 0], length 13):
  [0] time_day, [1] stage, [2] unused (do not read),
  [3..7]   alpha strain split [critical, dead, infected, none, recovered],
  [8..12]  delta strain split [critical, dead, infected, none, recovered].
"""

import numpy as np


def _latest(obs):
    """Return the length-13 latest-state vector from an obs of shape (history,1,13)."""
    return np.asarray(obs, dtype=float)[-1, 0]


def _critical(state):
    """Combined critical-case proportion across alpha + delta strains."""
    return float(state[3] + state[8])


def _infected(state):
    """Combined currently-infected (non-critical) proportion across both strains."""
    return float(state[5] + state[10])


def _recovered(state):
    """Combined recovered proportion across both strains."""
    return float(state[7] + state[12])


def _none_susceptible(state):
    """Combined never-infected / still-susceptible proportion across both strains."""
    return float(state[6] + state[11])


def performance_metric_economic_openness(traj) -> float:
    """Economic stakeholder: prefers more days at low regulation stages so venues stay open."""
    if len(traj) == 0:
        return 0.0
    stages = np.array([float(act) for (_o, act, _n) in traj], dtype=float)
    # Average "openness" = 4 - stage, normalized to [0, 1]; higher = more open economy.
    openness = (4.0 - stages) / 4.0
    return float(np.mean(openness))


def performance_metric_hospital_capacity(traj) -> float:
    """Hospital-capacity stakeholder: prefers the peak critical-case load to stay below ICU capacity."""
    if len(traj) == 0:
        return 0.0
    # ICU capacity in a 500-person town is around ~0.01 of population in critical
    # condition; once peak critical exceeds that, the hospital is overwhelmed.
    # Score is the negative of how far above capacity the peak landed, capped at 0
    # when peak stays under capacity (i.e. a fully-served outbreak scores 0, an
    # overwhelmed one scores negative).
    ICU_CAPACITY = 0.01
    peak_critical = max(_critical(_latest(next_obs)) for (_o, _a, next_obs) in traj)
    return float(-max(0.0, peak_critical - ICU_CAPACITY))


def performance_metric_epidemic_control(traj) -> float:
    """Epidemic-control stakeholder: prefers a smaller final attack rate (fewer people ever infected)."""
    if len(traj) == 0:
        return 0.0
    # Final attack rate = 1 - still-susceptible at episode end. We negate so
    # higher = better (fewer people ever got sick).
    final = _latest(traj[-1][2])
    susceptible = _none_susceptible(final)
    # Two strains each contribute up to 1.0 of "never infected"; the per-strain
    # attack rate is (1 - none). Average across the two strains.
    alpha_attack = 1.0 - float(final[6])
    delta_attack = 1.0 - float(final[11])
    mean_attack = 0.5 * (alpha_attack + delta_attack)
    return float(-mean_attack)


def performance_metric_policy_predictability(traj) -> float:
    """Civic-trust stakeholder: prefers regulations that change in small, gradual steps rather than abrupt swings."""
    if len(traj) == 0:
        return 0.0
    stages = np.array([float(act) for (_o, act, _n) in traj], dtype=float)
    if len(stages) < 2:
        return 0.0
    # Average absolute day-over-day stage change; higher swings (e.g. 0 -> 4)
    # are worse than small adjustments (e.g. 2 -> 3). Negate so higher = better.
    mean_abs_change = float(np.mean(np.abs(np.diff(stages))))
    return float(-mean_abs_change)


def performance_metric_education_access(traj) -> float:
    """Education stakeholder: prefers more days with schools open (stages 0 or 1)."""
    if len(traj) == 0:
        return 0.0
    stages = np.array([float(act) for (_o, act, _n) in traj], dtype=float)
    # Schools close starting at stage 2; in-school days correspond to stages <= 1.
    return float(np.mean(stages <= 1))


def performance_metric_long_term_health(traj) -> float:
    """Long-term-health stakeholder: prefers more of the eventually-infected population to recover rather than remain ill."""
    if len(traj) == 0:
        return 0.0
    # At episode end, of everyone who was ever infected, what fraction has
    # transitioned to recovered (vs still infected or critical)? Higher = better.
    final = _latest(traj[-1][2])
    recovered = _recovered(final)
    infected_now = _infected(final)
    critical_now = _critical(final)
    still_ill = infected_now + critical_now
    ever_ill = recovered + still_ill
    if ever_ill <= 1e-9:
        # No one was infected: trivially "healthy" outcome.
        return 1.0
    return float(recovered / ever_ill)
