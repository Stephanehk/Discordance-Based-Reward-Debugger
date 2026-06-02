# Prompt: implement candidate performance metrics

> **Task:** Implement a small set of diverse performance metrics that
> represent things other perspectives may find desirable for the
> environment. The goal is for the set of performance metrics to
> encapsulate what other different users may consider as good or bad
> outcomes.

## Context to use

- `env_config.py` — what the env is, what obs/action mean.
- The filled reward-design template (top of `designed_reward_fn.py`) —
  what the *current* user said they care about. Your metrics should
  cover **other** perspectives, not be variants of this.
- `gather_evidence/trajectory_features.py` — features you can lean on, but
  metrics should be self-contained (compute from `traj` directly).

## Output requirements

Write `candidate_performance_metrics/performance_metrics.py`. Each metric
is a module-level function:

```python
def performance_metric_<short_name>(traj) -> float:
    """One short line describing whose perspective this represents."""
    # traj: list of (obs, action, next_obs) tuples
    ...
    return float(value)
```

The function name must start with `performance_metric_`. The docstring
must be one short line (it's the label printed to the user when a
disagreement is found, e.g., "Energy efficiency: lower control effort is
better").

## Guidance

- **Different perspectives, not different formulations.** Examples:
  - Task-success rate (engineer / product manager)
  - Energy / control effort minimization (operations)
  - Safety / smoothness / low-jerk motion (regulator / safety reviewer)
  - Time-to-completion (end user)
  - Robustness / consistency across trajectory (reliability engineer)
  - Side-effect avoidance (environment / cohabitants)
- Pick a count that fits the env — typically 3-6. If two perspectives are
  near-duplicates, keep one.
- **Higher should always mean "better" by this perspective's lights.**
  If naturally lower-is-better (energy use, time), return the *negation*
  so the disagreement detector can use a single direction.
- Metrics should be **non-trivial** — at least one should correlate
  positively with the designed reward (so disagreements are surprising)
  and at least one should be capable of disagreeing.
- No external state, no env access — compute from `traj` only.
