---
name: baseline-evidence
description: Steps 3-5 of the reward-design Baseline (a separate process from /repair-*; see BASELINE.md). Authors a single reward-maximizing policy, rolls it out, shows 2-5 of its trajectories as feature vectors plus designed-reward totals, and asks the user whether to edit the reward, collect more evidence, or finish. Invoke whenever the user wants to see what maximizing the current reward produces.
---

# /baseline-evidence — roll out the reward-maximizer and show its trajectories

This is the baseline's evidence step. Unlike the method (`/repair-round` +
`/repair-preferences`), the baseline collects evidence **only** by rolling out
one reward-maximizing policy and showing **individual** trajectories from it —
never pairs, never stakeholder-metric disagreements. See `BASELINE.md`.

A single invocation runs **one** round: author the maximizer, roll it out,
show 2-5 of its trajectories, then ask the user what to do next. "Collect more
evidence" re-enters this skill as the next round.

## Tone for messages to the user

The user will never read the codebase. Don't list file names, internal
scripts, rollout counts, or policy class names. Talk about what the system is
**doing** in plain language ("rolling out the behavior that maximizes your
reward", "showing you what that behavior looks like"). **Never call any
behavior, reward, or trajectory "correct" or "incorrect"** — at most *possibly*
correct/incorrect or "appears aligned/misaligned with what you described."
(See `CLAUDE.md` for the full tone rules; they apply here too.)

## Eval mode — information barrier

If `<WORKSPACE_DIR>/eval_spec.md` exists, this is an evaluation workspace.
Authoring the reward-maximizing policy and writing the per-trajectory
descriptions shown to the user must be **spec-blind**: do not read or condition
on `eval_spec.md`, `evaluate_against_spec.py`, or any criteria-naming text.
Delegate those judgment steps to a subagent whose context holds only the env
semantics, the verified denormalization facts, and the designed reward's
operational mechanics. Describe trajectories purely by their observed
behavior/features, never citing the criteria. See "Information barrier" in
`BASELINE.md` and `CLAUDE.md`.

## Resolve the workspace

Takes the workspace name as `args` (e.g. `/baseline-evidence hopper`). If
`args` is empty, read `.current_baseline_workspace` at the repo root; if that
doesn't exist, tell the user to run `/baseline-setup` first.

`WORKSPACE_DIR = baseline_workspaces/<WORKSPACE>`. Always pass the full path
`--workspace baseline_workspaces/<WORKSPACE>` to the CLIs (a bare name resolves
to `workspaces/<WORKSPACE>`).

## Preconditions

- `<WORKSPACE_DIR>/env_config.py`, `<WORKSPACE_DIR>/designed_reward_fn.py`, and
  `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py` all exist. If any is
  missing, tell the user (in user-friendly terms) to run `/baseline-setup` or
  `/baseline-init` first.

## Procedure

### Step 1 — figure out the next round number

```
import glob, re
existing = glob.glob(f"{WORKSPACE_DIR}/gather_evidence/data_collection_policy_round_*.py")
nums = [int(re.search(r"round_(\d+)\.py$", p).group(1)) for p in existing]
k = (max(nums) + 1) if nums else 1
```

### Step 2 — author the reward-maximizing policy

Write `<WORKSPACE_DIR>/gather_evidence/data_collection_policy_round_{k}.py`.
Read `baseline_prompts/reward_maximizing_policy_prompt.md` for the verbatim
task text. The file must define **exactly one** policy class (whose name starts
with `DataCollectionPolicy`) plus `POLICIES = [ThatClass]`, conforming to the
invariants in `CLAUDE.md`. The class approximates the argmax of the expected
return under the **current** `designed_reward_fn`, implemented directly with no
training. In eval mode, delegate this authoring to a spec-blind subagent.

### Step 3 — roll out + featurize

```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python -m runtime.run_round --workspace baseline_workspaces/<WORKSPACE> --round K
```

This writes trajectories and features under
`<WORKSPACE_DIR>/gather_evidence/round_{K}_trajectories/` and
`.../round_{K}_trajectory_features/`. Just confirm the round completed.

### Step 4 — select 2-5 trajectories to show

Trajectory selection is **Claude-driven** — you pick a small, feature-diverse
set, not an algorithm.

**4a. Dump the trajectory table.**
```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python show_trajectories.py --workspace baseline_workspaces/<WORKSPACE> --mode dump
```
This writes `<WORKSPACE_DIR>/.baseline_history/_trajectory_table.txt`: one row
per trajectory, with a stable id (`T001`, …), its total under the designed
reward, and every feature.

**4b. Read the table and choose 2-5 trajectories** that best show what
maximizing this reward actually produces. Favor a feature-diverse set (use the
internal `src` column only for diversity reasoning; never reveal round or
rollout indices). If all rollouts collapsed to near-identical behavior, it is
fine to show fewer.

**4c. Write the selection and render it.** Write
`<WORKSPACE_DIR>/.baseline_history/_selection.json`:
```json
{"trajectories": ["T003", "T012", "T021"]}
```
Then render:
```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python show_trajectories.py --workspace baseline_workspaces/<WORKSPACE> --mode render \
       --selection baseline_workspaces/<WORKSPACE>/.baseline_history/_selection.json
```
The render step prints each selected trajectory anonymized — its designed-reward
total and a feature-value table under a shared feature key — and saves a
snapshot for provenance.

### Step 5 — reprint the trajectories verbatim, with a per-trajectory note

This is the load-bearing deliverable. The bash output is invisible to the
user — only what you type into your response is.

**Open the response with these four leading elements, in order** (before any
trajectory):
1. A short (≤ 6 lines) **overall read** of what the reward *appears* to
   incentivize, referencing the features the maximizer drove (what the reward
   credits/penalizes — not file paths or variable names) + 1–2 possible
   tweaks. Stay descriptive: "the reward *appears* to prioritize X over Y — if
   that wasn't what you intended, one option is …", never "the reward is wrong."
2. An **orientation line**: "View the following trajectories to note where the
   designed reward function may be misaligned. You'll be asked to edit the
   reward function afterwards; you can also request additional trajectory
   features for more detail on these trajectories, or ask to view more
   trajectories." (Keep *may be misaligned* — not "wrong.")
3. The **feature key** inside a fenced code block.
4. A **criteria reminder**: "As a reminder, here are the criteria your reward
   function must encode:" followed by the criteria the reward must encode (the
   filled reward-design template; in an evaluation workspace, the criteria in
   `eval_spec.md`, displayed to the human verbatim — human-facing only, never
   fed to the spec-blind policy/description subagent).

Then, for **every** trajectory the render step returned, in order:
   a. Print its `TRAJECTORY k/N` block **verbatim** inside its own fenced code
      block (header through the feature table; do not change any text or
      numbers).
   b. Immediately after that code block, write a **`What this trajectory
      does:`** line followed by 1-2 sentences describing the behavior purely
      from its features and reward total (e.g. "spends most steps at high
      speed and accumulates a large reward total; the closeness feature stays
      near zero throughout"). Stay descriptive — never "correct"/"incorrect",
      and in eval mode never cite the criteria.

Do not paraphrase or truncate any trajectory block. Do not skip the feature
key or the criteria reminder. Do not re-introduce round/rollout/policy
provenance — the render step intentionally anonymizes it.

### Step 6 — always close by asking the user what to do next

Use `AskUserQuestion` with exactly these four options (wording may vary
slightly), phrased in user-friendly language ("How would you like to
proceed?"). Do **not** mention command names in the question text.

- **Edit the reward function** — change `designed_reward_fn.py` to address
  what the trajectories revealed.
- **Add another trajectory feature** — add a new feature to the set that
  encodes each trajectory, then re-present the trajectories with it visible.
- **View more trajectories** — show additional trajectories from the
  reward-maximizing behavior.
- **Finish and submit (not reversible)** — finalize the current reward.

Act on the answer:
- **Edit the reward function** → ask whether the user wants to make the edit
  themselves or have Claude make it.
  - *Claude makes the edit:* first print the reward-edit tip box (see
    "Reward-edit tips" in `CLAUDE.md`). **In eval mode, apply the reward-edit
    gate first** (see "Reward-edit gate" in `CLAUDE.md` / `BASELINE.md`): if the
    request merely paraphrases the criteria with no concrete edit, refuse —
    quote the relevant `eval_spec.md` lines and ask for a concrete edit (a term,
    weight, threshold, or functional shape). Only with a concrete edit may you
    modify `designed_reward_fn.py`. Apply edits with `Edit`, show the diff back,
    then suggest running `/baseline-evidence` to roll out a fresh maximizer
    under the new reward.
  - *User edits it themselves:* tell the user the system will wait; when
    they're done they can run `/baseline-evidence` to roll out a fresh maximizer
    under the new reward and see how the behavior changed.
- **Add another trajectory feature** → ask the user what feature they want (or
  implement the one they named). Add it to
  `gather_evidence/trajectory_features.py` (in eval mode keep the
  implementation spec-blind — derive it from env semantics, not the criteria),
  re-run `runtime.compute_features` over every existing round so all
  trajectories carry the new feature, then re-present the trajectories
  (Steps 4–5) with the feature visible.
- **View more trajectories** → select additional trajectories distinct from
  those already shown (Step 4) and present them (Step 5). If the collected
  rollouts are exhausted, re-enter this skill as round k+1 (a fresh maximizer
  under the *current* reward) to produce more.
- **Finish and submit (not reversible)** → write
  `<WORKSPACE_DIR>/.baseline_history/SUBMISSION.md` containing a UTC timestamp
  and the full current reward source, tell the user the design is submitted and
  the session is complete, and stop. Do not offer further steps or edit the
  reward afterward. Only do this when the user explicitly selects it.

Skipping the four leading elements (overall read, orientation line, feature
key, criteria reminder), the verbatim trajectory blocks, the per-trajectory
note, or the closing `AskUserQuestion` is not optional — they are the core
deliverable.
