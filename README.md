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

## Where to start

To see how the system works, ask Claude to run **`merge_human_eval_1`** —
a bundled evaluation environment. Just say something like *"run
merge_human_eval_1"* (or start it yourself with `/repair-setup
merge_human_eval_1`). It walks the full loop on a highway on-ramp merge:
you're given a deliberately flawed starting reward and the criteria it
should satisfy, and Claude surfaces trajectory pairs that expose the
flaws while you revise the reward and verify the fixes. See
[Evaluation workspaces](#evaluation-workspaces) for what these
environments are.

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

### How pairs are selected

Pair selection is **Claude-driven**, not algorithmic. `select_pairs.py
--mode dump` writes a table of every collected trajectory — its features
(the same encoding you read), its total under the designed reward, and
every stakeholder metric. Claude reads that table and picks the pairs
most useful to show: ones where the designed reward and a stakeholder
you plausibly care about rank the trajectories oppositely, favoring a
diverse set across different stakeholders, behaviors, and reward terms
rather than re-showing variants of the loudest single disagreement.
Claude writes its picks to a selection file, and `select_pairs.py
--mode render` orients each pair (A = the reward-preferred side), lists
the disagreeing stakeholders, prints the preference-label block, and
snapshots the pair so `/repair-verify` can recheck it later.

A legacy algorithmic selector (`find_informative_pairs.py`, a
rank-product disagreement check plus a Mahalanobis-greedy diversity
pick) is kept for reference and is still used by `/repair-verify` to
re-score snapshotted pairs under the current reward.

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

## Evaluation workspaces

The repairer can also be used to **evaluate itself**: how good is the
system at letting a person fix a reward function when they are told only
the *properties* the reward must satisfy, and left to translate those
properties into concrete edits? It's a test of whether the surfaced
disagreement pairs actually help a human find and articulate a reward's
flaws.

An evaluation workspace is marked by an `eval_spec.md` file holding the
reward-design criteria the person must hit, in plain language. It ships
pre-authored: a fixed `env_config.py`, an intentionally misaligned
`designed_reward_fn.py` as the starting point, and the `eval_spec.md`
criteria. `/repair-setup` skips elicitation, shows the person the spec
and the starting reward, and the normal flow proceeds from there.

A guardrail keeps the test honest: when the person asks Claude to edit
the reward, Claude refuses if the request is just a paraphrase of the
spec criteria with no concrete edit (a term, weight, threshold, or
functional shape). The evaluation measures whether the *person* can
identify and articulate the flaws — not whether Claude can infer fixes
from a restated goal.

`evaluate_against_spec.py` scores the result with a **ranking test**: the
reward is aligned with the spec when every spec-satisfying trajectory
out-scores every violating one. Because real rollouts rarely satisfy the
strict criteria, it synthesizes a guaranteed-compliant set by ablating
each collected trajectory (forcing speed and gaps to compliant values and
clearing crashes), then reports whether the reward ranks all compliant
trajectories above all violators, with an inversion count and a YES/NO.
It lives inside the workspace (`workspaces/<name>/evaluate_against_spec.py`)
as a persistent eval artifact — kept across workspace resets — and is run
from the repo root: `python workspaces/<name>/evaluate_against_spec.py`.

Two evaluation environments ship with the repo, both on the same highway
on-ramp merge env, both starting from the same deliberately misaligned
speed-only reward:

- **`workspaces/merge_human_eval_1`** — revise the reward so the ego keeps
  at least 10 m from other vehicles, never crashes, and averages about
  25 m/s.
- **`workspaces/merge_human_eval_2`** — the same criteria plus a left-lane
  preference ("all else equal, the left lane is better than the right").
  Its evaluator adds an all-else-equal lane test: for each compliant
  trajectory it builds left- and right-lane variants and checks that the
  reward scores the left one strictly higher.

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

# evaluation workspaces only: score the reward against the spec
python workspaces/<name>/evaluate_against_spec.py
```

If `.current_workspace` is set, `--workspace` may be omitted.

## Project layout

See `CLAUDE.md` for the full file layout, the contracts every authored
file must satisfy (`env_config.py`, `designed_reward_fn.py`,
`Trajectory2Features`, data-collection policies, performance metrics,
on-disk trajectory/feature formats), and the conventions Claude follows
when authoring Python in this repo.
