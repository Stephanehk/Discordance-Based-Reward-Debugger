"""Environment config for the Sony / Pan-et-al. Pandemic Simulator.

Wraps `pandemic_simulator.environment.PandemicGymEnv`, a gymnasium env in
which the agent is a public-health authority choosing, once per simulated
day, how strict the population-wide COVID regulations should be. The
underlying agent-based simulator advances a town of people through an
infection model and reports back the true global infection summary of the
outbreak (the actual share of the population in each infection category,
including deaths).
"""

import sys
import os
import itertools

import numpy as np

# The pandemic_simulator package lives in the local vendored checkout, not on
# the default path. Add its `python/` source root so `import pandemic_simulator`
# resolves regardless of which directory the runtime chdirs into.
_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "local_environments", "pandemic", "python")
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pandemic_simulator as ps
from pandemic_simulator.environment import (
    PandemicGymEnv,
    PandemicSimOpts,
    NoPandemicDone,
)
from pandemic_simulator.script_helpers import austin_regulations, tiny_town_config

# Each make_env() call gets its own seed so repeated rollouts explore
# different stochastic outbreaks rather than replaying one fixed town.
_seed_counter = itertools.count()


def make_env():
    """Build a fresh PandemicGymEnv.

    The agent picks one of five regulation stages (0 = no restrictions ...
    4 = full lockdown) each simulated day; the simulator runs a day of
    person-to-person contact under that regulation and returns the updated
    true infection summary as the observation.
    """
    ps.init_globals(seed=next(_seed_counter))
    env = PandemicGymEnv.from_config(
        sim_config=tiny_town_config,           # 500-person town
        sim_opts=PandemicSimOpts(),
        pandemic_regulations=austin_regulations,  # 5 regulation stages, 0..4
        done_fn=NoPandemicDone(30),            # end after 30 pandemic-free days
    )
    # PandemicGymEnv.step() unconditionally evaluates a "true" and "proxy"
    # reward and unpacks each as a (value, breakdown) tuple; left as None they
    # raise. We never read these values (the repairer recomputes the designed
    # reward from saved transitions), so point them at the env's own default
    # SumReward, which returns the expected tuple.
    env._true_reward_fn = env._reward_fn
    env._proxy_reward_fn = env._reward_fn
    return env


# The observation is a numpy array of shape (history, 1, 13). With the default
# single-step history the meaningful row is obs[-1, 0], a length-13 vector.
# Indices 3-7 and 8-12 are each a 5-way split over infection categories in the
# fixed order [critical, dead, infected, none, recovered], reported as the
# proportion of the town's population in each category (the categories sum to 1).
# These are the TRUE infection summary (the simulator's ground-truth population
# state), so the dead fields reflect actual mortality, not testing detections.
OBS_DESCRIPTION = """\
Observation: numpy array of shape (history, 1, 13). Use the latest row,
obs[-1, 0], a length-13 vector. The infection categories report the TRUE
population state (ground-truth shares, not testing-detected counts), so the
dead fields reflect actual mortality:

  [0]  time_day           - fraction of a year elapsed ((day*24 + hour)/(365*24)),
                            so it grows ~0.0027 per simulated day.
  [1]  stage              - current regulation stage in force, an integer 0..4
                            (0 = no restrictions, 4 = strictest lockdown).
  [2]  (unused)           - a legacy flag still present in the raw observation
                            array but ignored by this system; do not read it.

  Outbreak summary for the ALPHA strain (proportions of the population, sum to 1):
  [3]  alpha_critical     - proportion currently in critical condition.
  [4]  alpha_dead         - proportion deceased.
  [5]  alpha_infected     - proportion currently infected (non-critical).
  [6]  alpha_none         - proportion never infected / susceptible.
  [7]  alpha_recovered    - proportion recovered.

  Outbreak summary for the DELTA strain (proportions of the population, sum to 1):
  [8]  delta_critical     - proportion currently in critical condition.
  [9]  delta_dead         - proportion deceased.
  [10] delta_infected     - proportion currently infected (non-critical).
  [11] delta_none         - proportion never infected / susceptible.
  [12] delta_recovered    - proportion recovered.

Every category value is a proportion of the town's population (each strain's
five fields sum to 1): infections, critical cases, and deaths are all shares
of the population, not raw counts.
"""

ACTION_DESCRIPTION = """\
Action: a single discrete integer in {0, 1, 2, 3, 4} naming the regulation
stage to impose for the coming simulated day.

  0 - No restrictions. No stay-home-if-sick rule, hygiene, or masking;
      baseline (default) social distancing; no gathering-size limits. All
      venues open (offices, schools, hair salons, retail, restaurants, bars).
  1 - Stay home if sick + practice good hygiene; no masking; baseline social
      distancing; gathering caps of 25 (high-risk) / 50 (low-risk). All venues
      remain open.
  2 - Adds mandatory masking; social distancing 0.3; gathering caps of 10
      (high-risk) / 25 (low-risk). Schools and hair salons closed; offices,
      retail, restaurants, and bars open.
  3 - Masking; social distancing 0.5; no gatherings allowed (caps of 0).
      Schools and hair salons closed; offices, retail, restaurants, and bars
      open.
  4 - Strictest. Masking; social distancing 0.7; no gatherings allowed.
      Offices, schools, hair salons, and retail closed; restaurants and bars
      remain open.

Higher stages suppress transmission but impose greater economic and social
cost (more closed venues, fewer people allowed to gather).
"""

# Episodes run 80 simulated days each to keep rollout time down (the contact
# simulation is ~0.38 s/day); NoPandemicDone may end an episode sooner once the
# outbreak has been clear for 30 days.
MAX_EPISODE_STEPS = 80
