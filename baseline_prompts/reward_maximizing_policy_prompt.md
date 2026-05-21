# Prompt: implement a reward-maximizing policy (baseline)

> **Task:** Implement a single policy that maximizes the expected sum of
> `designed_reward_fn` over an episode, as a Python class. Do not train it —
> implement the controller directly.

## Context to use

- `env_config.py` (`make_env()`, `OBS_DESCRIPTION`, `ACTION_DESCRIPTION`).
- `designed_reward_fn.py` — the reward to maximize.
- `gather_evidence/trajectory_features.py` — the features through which the
  human reads the resulting trajectories.

## Output requirements

Write `gather_evidence/data_collection_policy_round_{k}.py` (with `k` set to
the next round number). It must define exactly one policy class and a
module-level `POLICIES` list:

```python
class DataCollectionPolicyRewardMaximizer:
    """Maximizes the expected sum of the designed reward."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        # any other state

    def act(self, obs):
        # return an action valid for env.action_space
        ...


POLICIES = [DataCollectionPolicyRewardMaximizer]
```

The class name must start with `DataCollectionPolicy` (the rollout boilerplate
discovers policies by that prefix). `POLICIES` must contain this one class.

## What the policy must do

The policy must maximize the expected sum of rewards with respect to
`designed_reward_fn`. Implement it directly with no learned parameters and no
training loop: `act(obs)` must run immediately and deterministically (or with a
fixed RNG). Acceptable approaches include a hand-derived closed-form controller,
greedy one-step maximization of the designed reward over candidate actions, or
short-horizon lookahead/rollout using a copy of the env. Choose the approach
that most closely tracks the reward's incentives for this environment and
action space.

The policy targets the reward as written, including any flaw in it: if the
reward is exploitable, the maximizer is expected to exploit it. Do not add
behavior the reward does not credit.
