# Prompt: implement trajectory features

> **Task:** Implement a set of features to summarize trajectories so that
> the human reward designer can easily understand what happened in the
> trajectory. In particular, the set of features should shed light on the
> extent to which the specified outcomes were achieved or avoided, and
> should additionally help someone understand how the specified reward
> function will be optimized for.

## Context to use

- `env_config.OBS_DESCRIPTION` and `ACTION_DESCRIPTION` — the semantics of
  observation and action dimensions.
- The filled reward-design template (preserved at the top of
  `designed_reward_fn.py`) — the specified outcomes to achieve / avoid.
- The body of `designed_reward_fn.py` — what the reward actually rewards,
  so features can expose how it might be over- or under-rewarded. Read it
  and enumerate the terms it sums or multiplies before choosing features.

## Output requirements

Write `gather_evidence/trajectory_features.py`. It must define:

```python
class Trajectory2Features:
    # list of (short_name, one-line description); same length and order
    # as the list returned by get_traj_features.
    feature_names: list[tuple[str, str]] = [
        ("name_1", "what it measures"),
        ...
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        # trajectory: list of (obs, action, next_obs) tuples (numpy arrays / scalars).
        # Return a list of floats, same length as feature_names.
        ...

    def pretty_print_features(self, features) -> None:
        # Print each feature value next to its short_name and description.
        # Two-decimal floats, aligned. Use feature_names for labels.
        ...
```

## Guidance

- **Mix scopes.** Some features should aggregate transitions (mean, max,
  fraction of timesteps satisfying a predicate); some should be
  trajectory-level (final-state distance, total path length, time to
  first success).
- **Cover both achieve and avoid.** For every outcome the user listed,
  there should be at least one feature that measures whether it happened.
- **Cover every reward term**
  From the terms you enumerated in `designed_reward_fn.py`, ensure each
  term has a feature that measures whether / how strongly it fired.
  Wherever a term rewards a *proxy* for what the user actually wants (a
  quantity correlated with the goal but not identical to it), include
  features that expose the **proxy and the true outcome separately** — so
  a trajectory that drives the proxy high without achieving the outcome is
  visibly distinct from one that genuinely succeeds. Record the
  feature→term mapping in a comment block at the top of the file.
- **Expose reward-shape sensitivity.** If the reward function uses, e.g.,
  velocity squared, include features that would distinguish "high
  velocity briefly" from "moderate velocity throughout".
- **Keep it small.** Aim for 6-12 features. More than that becomes hard
  to read.
- **No hidden dependencies on reward magnitude** — features should be
  interpretable on their own terms.
