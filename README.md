# Automated Reward Repairer

A Claude-Code-driven workflow for surfacing **evidence of reward
misspecification** in gym/gymnasium environments. You bring an env and a
hand-written reward function; the system rolls out diverse policies,
scores trajectories with both your reward and a set of
stakeholder-perspective performance metrics, and shows you trajectory
pairs where the two disagree. You read the pairs and decide how to edit
the reward.

The pairs are the end product. There is no automated reward learning —
this is a debugger for reward functions.

## Requirements

- Python 3.10+
- `numpy`, `gymnasium` (see `requirements.txt`)
- Claude Code (the skills below are invoked as slash commands)
- Optional per-workspace dependencies (declared in each workspace's
  own `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Quickstart

All commands are slash commands run inside Claude Code, from the repo
root. `<name>` is the workspace name (e.g. `hopper`).

```
/repair-setup <name>     # author env_config.py + designed_reward_fn.py
/repair-init             # author feature extractor + performance metrics
/repair-round            # roll out policies, score them, surface disagreements
/repair-preferences      # re-show the most informative disagreement pairs
/repair-verify           # after editing the reward, recheck snapshotted pairs
```

Each skill ends by telling you what to run next, so you can usually just
follow the prompts.

## How a session flows

1. **`/repair-setup <name>`** — elicits a reward-design template from
   you (or stubs one), then writes `env_config.py` and
   `designed_reward_fn.py` under `workspaces/<name>/`.
2. **`/repair-init`** — writes a `Trajectory2Features` class that turns
   trajectories into interpretable feature vectors, plus a set of
   `performance_metric_*` functions encoding different stakeholder views
   ("the safety reviewer", "the highway-operations view", ...).
3. **`/repair-round`** — authors data-collection policies, rolls them
   out in the env, computes features, and checks whether the evidence
   is sufficient. Every round includes a policy that maximizes the
   *current* designed reward, plus additional policies that probe
   reward terms the prior evidence didn't exercise. May run up to 3
   rounds per invocation, then hands off to `/repair-preferences`.
4. **`/repair-preferences`** — prints up to 15 trajectory pairs as
   preference labels: feature key, anonymized Trajectory A vs
   Trajectory B, reward + metric scores with explicit A≻B / B≻A
   labels, and a feature delta table. Each pair is followed by two
   summary lines — *why they disagree* and *what you might change in
   the reward function to flip this preference* — and the run ends
   with an overall pattern interpretation. Pairs are snapshotted to
   disk so `/repair-verify` can recheck them later.
5. **You edit the reward.** Then either:
   - **`/repair-verify`** — recomputes the designed reward on the
     snapshotted pairs and reports which disagreements are now resolved.
   - **`/repair-round`** — skips verification and collects fresh
     evidence under the updated reward; later rounds target the
     current reward's remaining flaws.

`/repair-preferences` can be re-run any time — it reloads from disk and
re-scores under whatever reward is currently in the workspace.

### How pairs are selected

The pool of disagreeing pairs is all unordered trajectory pairs where
the designed reward and at least one performance metric rank the two
trajectories in opposite orders (rank-product check with an optional
`--margin` threshold). From that pool, pairs are picked greedily to
maximize `xᵀ V⁻¹ x` over the feature delta `x = φ(B) − φ(A)`, where
`V` starts as `κλ · I` and accumulates `x xᵀ` from each prior pick.
Each new pick prefers a direction in feature-delta space that the
prior picks haven't already covered, so the surfaced pairs span
different reward terms, different stakeholders, and different
trade-off axes rather than re-showing variants of the loudest single
disagreement.

## Workspaces

Everything env-specific lives under `workspaces/<name>/`. The shared
`runtime/`, `find_informative_pairs.py`, `prompts/`, and skills are
workspace-agnostic and select the active workspace at runtime via
`--workspace workspaces/<name>`. The active workspace is persisted to
`.current_workspace` at the repo root.

You can have many workspaces side by side; they share zero state.

```
workspaces/<name>/
├── env_config.py
├── designed_reward_fn.py
├── gather_evidence/
│   ├── trajectory_features.py
│   ├── data_collection_policy_round_{k}.py
│   ├── round_{k}_trajectories/
│   └── round_{k}_trajectory_features/
└── candidate_performance_metrics/
    └── performance_metrics.py
```

## Running the boilerplate directly

The skills do this for you, but you can also drive the Python directly:

```bash
python -m runtime.run_round --workspace workspaces/<name> --round K
python find_informative_pairs.py --workspace workspaces/<name> \
       --top-overall 15 --format preference-labels
```

If `.current_workspace` is set, `--workspace` may be omitted.

## Project layout

See `CLAUDE.md` for the full file layout, the contracts every authored
file must satisfy (`env_config.py`, `designed_reward_fn.py`,
`Trajectory2Features`, data-collection policies, performance metrics,
on-disk trajectory/feature formats), and the conventions Claude follows
when authoring Python in this repo.
