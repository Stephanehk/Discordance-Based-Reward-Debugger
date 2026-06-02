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
4. **`/repair-preferences`** — prints up to 2 trajectory pairs plus up
   to 1 lottery comparison (a deliberately tight budget — every shown
   comparison must be maximally informative). Pairs render as preference
   labels (feature key, anonymized Trajectory A vs Trajectory B, reward
   + metric scores with explicit A≻B / B≻A labels, feature delta table).
   Lottery comparisons render as a question — "Do you prefer this
   certain trajectory C, or a p% chance of A and (1−p)% chance of B?"
   — with the designed reward's and stakeholders' preferences over C vs
   the lottery. Each comparison is followed by a *why they disagree*
   summary; if the workspace ships a `viz.py` hook, each comparison also
   saves a per-trajectory visualization under
   `workspaces/<name>/viz/round{i}/{pair,lottery}{j}/` and prints the
   paths. The run ends with an overall pattern interpretation. Both
   kinds are snapshotted to disk so `/repair-verify` can recheck them
   later.
5. **You decide what to do.** After the pairs, Claude asks whether to
   collect another round, edit the reward yourself, have Claude apply a
   concrete edit you specify, or **finish and submit** (final, not
   reversible). Once the reward is edited (by you or by Claude), then
   either:
   - **`/repair-verify`** — recomputes the designed reward on the
     snapshotted pairs and reports which disagreements are now resolved.
   - **`/repair-round`** — skips verification and collects fresh
     evidence under the updated reward; later rounds target the
     current reward's remaining flaws.

`/repair-preferences` can be re-run any time — it reloads from disk and
re-scores under whatever reward is currently in the workspace.

### How comparisons are selected

Selection is **Claude-driven**, not algorithmic. `select_pairs.py
--mode dump` writes a table of every collected trajectory — its features
(the same encoding you read), its total under the designed reward, and
every stakeholder metric. Claude reads that table and picks the
comparisons most useful to show: up to 2 head-to-head **pairs** where
the designed reward and a stakeholder you plausibly care about rank the
two trajectories oppositely, and up to 1 **lottery** comparison of the
form "certain trajectory C versus a p-vs-(1−p) lottery over A and B"
where the reward's risk attitude or curvature plausibly disagrees with
a stakeholder. Claude must justify each lottery's `p` in the selection
file (a free real in [0, 1]). The same selection criterion applies to
both kinds: maximize informativeness about where the designed reward
might be misaligned, favoring diversity across stakeholders, behaviors,
and reward terms over re-showing variants of the loudest single
disagreement. Claude writes its picks to a selection file, and
`select_pairs.py --mode render` orients each pair (A = the
reward-preferred side), lists the disagreeing stakeholders for both
pairs and lotteries, prints each comparison, and snapshots all of them
so `/repair-verify` can recheck them later.

A legacy algorithmic selector (`find_informative_pairs.py`, a
rank-product disagreement check plus a Mahalanobis-greedy diversity
pick) is kept for reference and is still used by `/repair-verify` to
re-score snapshotted pairs and lotteries under the current reward.

## Workspaces

Everything env-specific lives under `workspaces/<name>/`. The shared
`runtime/`, `select_pairs.py`, `find_informative_pairs.py`, `prompts/`,
and skills are workspace-agnostic and select the active workspace at
runtime via `--workspace workspaces/<name>`. The active workspace is
persisted to `.current_workspace` at the repo root.

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
# roll out + featurize one round
python -m runtime.run_round --workspace workspaces/<name> --round K

# Claude-driven pair selection: dump the table, then render chosen pairs
python select_pairs.py --workspace workspaces/<name> --mode dump
python select_pairs.py --workspace workspaces/<name> --mode render \
       --selection workspaces/<name>/.repair_history/_selection.json

# re-score snapshotted pairs under the current reward (used by /repair-verify)
python find_informative_pairs.py --workspace workspaces/<name> --check-snapshots
```

If `.current_workspace` is set, `--workspace` may be omitted.

## Project layout

See `CLAUDE.md` for the full file layout, the contracts every authored
file must satisfy (`env_config.py`, `designed_reward_fn.py`,
`Trajectory2Features`, data-collection policies, performance metrics,
on-disk trajectory/feature formats), and the conventions Claude follows
when authoring Python in this repo.
