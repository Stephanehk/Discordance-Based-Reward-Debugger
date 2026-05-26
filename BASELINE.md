# Reward-Design Baseline

This is a **separate process** from the Automated Reward Repairer described in
`CLAUDE.md`. It exists to be compared against that method. It reuses the same
runtime boilerplate and the same reward/feature-authoring style, but it
collects evidence in a fundamentally different way.

**Do not edit `CLAUDE.md` or any `/repair-*` skill/prompt to accommodate the
baseline.** The baseline lives in its own files (listed below); where it needs
behavior identical to the method, it copies the relevant prompt/skill rather
than sharing or editing it. The baseline skills still inherit the **global
rules in `CLAUDE.md`** (user-facing tone, "Finish and submit", the reward-edit
gate, the eval-mode information barrier), which apply to *every* skill in
`.claude/skills/`.

## How the baseline differs from the method

The method collects diverse trajectories from many policies, scores them with
the designed reward **and** a set of independent stakeholder performance
metrics, and surfaces **trajectory pairs where the two disagree**. The user
reads the disagreements.

The baseline does none of that:

1. Specify a reward function the same way the method does — a menu with either
   "describe it and I'll implement it" (filling out
   `prompts/reward_design_template.md`) or "I'll write it myself".
2. Author trajectory features the same way the method does, so a trajectory can
   be presented back to the user in feature space. **No stakeholder performance
   metrics are authored.**
3. Implement **one** policy, as a Python class, that maximizes the expected sum
   of rewards under the specified reward (no training — the policy is
   implemented directly), and roll it out.
4. Present **2–5 sampled trajectories from only that policy**, each encoded as
   its feature vector plus its total under the designed reward. These are
   **individual** trajectories, not pairs.
5. Ask the user whether to edit the reward, collect more evidence, or finish —
   exactly as the method does.

The single difference that defines the baseline: it gathers evidence **only by
rolling out the reward-maximizing policy and showing its trajectories**, never
by contrasting them against stakeholder metrics or each other.

## End-to-end workflow

```
/baseline-setup <name>   →  baseline_workspaces/<name>/env_config.py
                            baseline_workspaces/<name>/designed_reward_fn.py
                            (after authoring the reward, show it back to the user)
/baseline-init <name>    →  baseline_workspaces/<name>/gather_evidence/trajectory_features.py
/baseline-evidence <name> → authors the reward-maximizing policy, rolls it out,
                            shows 2-5 of its trajectories as feature vectors +
                            designed-reward totals, then asks the user what to
                            do next (edit reward / collect more / finish).
```

`/baseline-evidence` is re-runnable: "collect more evidence" appends another
round (`round_{k+1}`) — a fresh maximizer under the *current* reward — and
shows new trajectories. There is no verify step (the baseline surfaces no pairs
to re-check).

## Workspace layout

Baseline workspaces are fully independent of the method's `workspaces/`. They
live under `baseline_workspaces/<name>/`, and the active baseline workspace is
persisted to `.current_baseline_workspace` at the repo root (never
`.current_workspace`).

```
baseline_workspaces/<name>/
├── env_config.py                 # authored in /baseline-setup
├── designed_reward_fn.py         # authored or stubbed in /baseline-setup
├── eval_spec.md                  # eval mode only (pre-authored / copied in)
├── gather_evidence/
│   ├── __init__.py
│   ├── trajectory_features.py             # authored in /baseline-init
│   ├── data_collection_policy_round_{k}.py # the reward-maximizing policy
│   ├── round_{k}_trajectories/            # pickles, one per rollout
│   └── round_{k}_trajectory_features/     # json, one per rollout
└── .baseline_history/
    ├── _trajectory_table.txt              # dump for Claude's selection
    ├── _trajectory_pool.json
    ├── shown_round_{NNN}.json             # provenance of shown trajectories
    └── SUBMISSION.md                      # on "Finish and submit"
```

There is **no** `candidate_performance_metrics/` directory and **no**
`.repair_history/`.

## Reuse — what the baseline shares with the method (unchanged)

- All of `runtime/` (`workspace.activate(path)` is path-driven; `run_round`
  rolls out any class named `DataCollectionPolicy*` and computes features).
  The baseline policy conforms to that invariant, so the same rollout CLI runs
  it. **Always pass the full path** `--workspace baseline_workspaces/<name>`
  (a bare name resolves to `workspaces/<name>`).
- `prompts/reward_design_template.md` and `prompts/trajectory_features_prompt.md`
  (read-only).
- `find_informative_pairs._print_feature_key` / `_load_designed_reward_fn` /
  `_read_features_for`, imported by `show_trajectories.py`. The baseline never
  calls `load_records` / `_load_performance_metrics` (those require metrics).

## Baseline-specific files

```
automated_reward_repairer/
├── BASELINE.md                                  # this file
├── show_trajectories.py                         # dump + render single trajectories
├── baseline_prompts/
│   └── reward_maximizing_policy_prompt.md        # frozen prompt for the policy
├── .claude/skills/{baseline-setup,baseline-init,baseline-evidence}/SKILL.md
└── baseline_workspaces/<name>/                   # one dir per env
```

## Invariants (same as the method, minus metrics)

Every authored file conforms to the invariants in `CLAUDE.md` ("Invariants
(do not break)"), with these baseline specifics:

- `gather_evidence/data_collection_policy_round_{k}.py` defines **exactly one**
  policy class (the reward maximizer) plus `POLICIES = [ThatClass]`.
- There is no `performance_metrics.py`; nothing in the baseline depends on one.

## How to run the baseline boilerplate

All Python runs in the `llm` conda env:

```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm
```

```
# roll out the reward-maximizing policy for round K:
python -m runtime.run_round --workspace baseline_workspaces/<name> --round K

# dump the trajectory table, then render the selected subset:
python show_trajectories.py --workspace baseline_workspaces/<name> --mode dump
python show_trajectories.py --workspace baseline_workspaces/<name> --mode render \
       --selection baseline_workspaces/<name>/.baseline_history/_selection.json

# eval mode only — spec evaluation harness (run from repo root; the same
# eval_scripts/<name>.py scores either the human-eval or the baseline copy):
python eval_scripts/<name>.py --workspace baseline_workspaces/<name>
```

## Evaluation workspaces

A baseline workspace is in evaluation mode iff
`baseline_workspaces/<name>/eval_spec.md` exists. Eval mode mirrors the
method's eval rules (see "Evaluation workspaces" and the information barrier in
`CLAUDE.md`):

- The pre-authored eval artifacts inside the workspace are `env_config.py`,
  `designed_reward_fn.py` (the human's intentionally-misaligned starting
  point), and `eval_spec.md`. The spec evaluator itself lives at the repo root
  as `eval_scripts/<name>.py` (one per env, shared with the matching human-eval
  workspace) — it is **not** part of the workspace. `/baseline-setup` will
  offer to copy the three in-workspace artifacts from a matching
  `workspaces/<name>/` eval workspace if they are not already present. A
  workspace reset must keep those three files; the eval script sits outside the
  workspace and is unaffected by resets.
- **Information barrier.** `eval_spec.md` is the human's private task. It may be
  read **only** to **display it to the human** — in `/baseline-setup` and as the
  criteria reminder shown after the feature key during evidence presentation
  (`/baseline-evidence`, Step 5). Both are human-facing reminders of the human's
  own task; the reminder is assembled by the main thread and must never reach
  the spec-blind subagent. Authoring the trajectory features (`/baseline-init`),
  authoring the reward-maximizing policy (`/baseline-evidence`), and writing the
  per-trajectory descriptions shown to the user must all be **spec-blind** and
  delegated to a subagent whose context contains only the env's obs/action
  semantics, the verified denormalization facts, and the designed reward's
  operational mechanics — never `eval_spec.md`, `eval_scripts/<name>.py`, the
  criteria-naming text, or the criteria-naming docstring lines of
  `designed_reward_fn.py`.
- **Reward-edit gate.** When the user asks Claude to modify the reward, apply
  the same gate as the method (see "Reward-edit gate" in `CLAUDE.md`). The gate
  has **two checks**: (1) the request must specify a concrete edit (a term,
  weight, threshold, or functional shape), and (2) every clause must read as a
  reward-mechanics instruction (term, weight, gating condition, functional
  shape) — **not** a policy-desideratum description of what the policy should
  do. Clauses that prescribe a reward term are fine even when they reference
  values/thresholds also named in `eval_spec.md`; clauses that only describe
  the desired behavior trip Check 2. If either check fails, refuse the
  **entire** request, quote the offending clause alongside the matching
  `eval_spec.md` line, and ask the user to rephrase.
- **Criteria reminder above any reward-edit menu.** Whenever a baseline skill
  presents an `AskUserQuestion` whose options include "Edit the reward
  function" (or any option that may lead to a reward edit), reprint the
  criteria from `eval_spec.md` verbatim in the main thread's text immediately
  above the question, prefaced by "As a reminder, here are the criteria your
  reward function must encode:" — in addition to any earlier display of the
  criteria in the same response. The reprint is human-facing only and must
  never reach a spec-blind subagent.
- The per-trajectory descriptions must stay purely descriptive of the observed
  behavior/features — never citing, quoting, or paraphrasing the criteria.
