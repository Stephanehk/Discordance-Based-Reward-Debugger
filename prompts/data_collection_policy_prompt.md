# Prompt: implement a data collection policy (one round)

> **Task:** Implement a data collection policy that will collect
> informative trajectories to identify possible flaws of this reward
> function, given these previously implemented data collection policies.

## Context to use

- `env_config.py` (`make_env()`, `OBS_DESCRIPTION`, `ACTION_DESCRIPTION`).
- `designed_reward_fn.py` — what the reward credits/penalizes. This is the
  thing whose flaws we want to surface.
- `.repair_history/_reward_analysis.md` — the reward decomposition: the
  per-term breakdown and the exploit hypotheses `H1, H2, …` your policies
  must make occur.
- `candidate_performance_metrics/performance_metrics.py` — the
  alternative-perspective metrics that `/repair-preferences` will later
  use to detect disagreements with the designed reward. Useful as one
  source of policy ideas (see Guidance below), but not the only one.
- `gather_evidence/trajectory_features.py` — the features through which
  the user reads trajectories. Behaviors that vary along these features
  are more legible.
- Every existing `gather_evidence/data_collection_policy_round_{1..k-1}.py`
  — what's already been tried. Round k must produce behavior different
  from what's been collected.

## Output requirements

Write `gather_evidence/data_collection_policy_round_{k}.py` (with `k` set
to whatever the next round number is). It must define:

```python
class DataCollectionPolicy<Hypothesis>:
    """One-line statement of the hypothesis this policy probes."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        # any other state

    def act(self, obs):
        # return an action valid for env.action_space
        ...
```

and a module-level list:

```python
POLICIES = [DataCollectionPolicy<Name1>, DataCollectionPolicy<Name2>, ...]
```

You pick how many classes.

## Guidance

Each policy should be designed to produce trajectories that **illuminate
whether the designed reward is well-specified**. There are several
informative strategies, and a good round usually mixes a few of them.
Use whichever fits the env, the reward, and what prior rounds have
already covered:

**Required in every round** (these are not optional among the strategies
below):
1. A policy that **maximizes the current `designed_reward_fn`**.
2. At least one **Strategy B exploit probe** per still-untested exploit
   hypothesis `H1, H2, …` in `_reward_analysis.md` — a policy that
   decouples a rewarded proxy from the true outcome it stands in for.
   Name and document each by the term and hypothesis id it targets.

Strategies A, C, and D below are optional additions on top of those two.

### Strategy A — Target a performance metric

Pick a specific metric in `candidate_performance_metrics/performance_metrics.py`
and construct a policy that will score *very differently* on it than
under `designed_reward_fn`. Two flavors:

- **High designed reward, low metric.** A policy that scores well under
  `designed_reward_fn` while doing badly on one of the other metrics —
  the classic Goodhart / reward-exploit case.
- **Low designed reward, high metric.** A policy that the designed
  reward dislikes, but that another perspective's metric considers good
  — surfaces cases where the reward under-credits genuinely desirable
  behavior.

Name and document such policies by the
metric they target and the direction of disagreement, e.g.
`DataCollectionPolicyExploitPrimaryObjective` with docstring
*"High designed reward, low smoothness metric — maximises the primary
objective while ignoring control-smoothness."*

### Strategy B — Directly target flaws in the designed reward (required)

Work from the reward decomposition (`_reward_analysis.md`) and the code
body of `designed_reward_fn.py`. For each exploit hypothesis `H1, H2, …`
not yet exercised by prior rounds, construct a policy that makes it
*occur*. The most informative kind, and the one to reach for first:

- **Decouple a proxy from its true outcome.** Find a term that rewards a
  proxy for what the user actually wants — any quantity the reward credits
  that is correlated with the goal but not identical to it — and write a
  policy that **drives that proxy term high while never achieving the true
  outcome it stands in for**. This is the case that most reliably exposes
  misspecification.
- **Reward exploits / Goodharting.** Maximize one rewarded term over all
  others, exploit dense terms while ignoring sparse ones, accumulate
  small per-step rewards via behavior the user didn't envision.
- **Specification-gap behaviors.** Achieve outcomes the user listed
  while violating *un*-listed common-sense norms (excessive jitter,
  destructive side effects, near-misses with avoided outcomes).

Name and document each policy by the reward term and hypothesis id it
targets, e.g. a class whose docstring reads
*"H[n]: drives [proxy term] high while never achieving [true outcome] —
decouples the proxy from task success."*

### Strategy C — Cover unexplored parts of the state space

If prior rounds stayed in one mode, push toward regions of the state /
action space where the reward's behavior is *untested*. The goal here is
not to chase a known flaw but to expose unknown ones — the reward might
be silently wrong in regions no policy has visited.

- Biased random walks toward unusual obs values.
- Policies that explicitly maximize variance of some observation dim.
- Policies that force the env into rarely-reached terminal conditions.

### Strategy D — Adversarial to prior rounds

Read every prior `data_collection_policy_round_{1..k-1}.py`. Identify an
axis those policies ignore or all share, and build a policy that varies
along it. Useful when prior rounds inadvertently collapsed onto similar
behavior.

---

You don't need to use every strategy in a single round. Mix to taste based on what would most expose the current reward's flaws.

Policies do **not** need to be optimal. Hand-coded heuristics, biased
random walks, hard-coded scripts, oscillators, deliberately suboptimal
controllers — all welcome, as long as they generate trajectories that
*illuminate* whether the reward is well-specified.

Avoid implementing learned policies (no training loop) — these need to
run by `act(obs)` deterministically (or with a fixed RNG) immediately.
