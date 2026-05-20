# Automated Reward Repairer

## What this project is

The user has a gym/gymnasium environment and a manually written (or
LLM-elicited) reward function `designed_reward_fn(obs, act, next_obs) -> float`.
This system surfaces *evidence of reward misspecification* by collecting
diverse trajectories, scoring them with both the designed reward and a set
of stakeholder-style performance metrics, and showing the user **trajectory
pairs where the two disagree on which trajectory is better**. The pairs are
the end product: the user reads them and decides how to edit the reward.

Claude is part of the system. The user-invocable skills below correspond to
the numbered steps in the spec. Each skill that authors a Python file is
driven by a frozen prompt in `prompts/`.

## Workspaces

Every environment-specific file lives in a **workspace** under
`workspaces/<name>/`. The boilerplate (`runtime/`,
`find_informative_pairs.py`, `prompts/`, `.claude/skills/`) is
env-agnostic — it is selected at runtime via `--workspace workspaces/<name>`.

Each `/repair-*` skill takes the workspace name as `args`
(e.g. `/repair-setup hopper`). The active workspace is persisted to
`.current_workspace` at the repo root, so subsequent skill calls default
to it. To switch workspaces, pass `args` explicitly again.

You can have many workspaces side by side; they share zero state.

## End-to-end workflow

```
/repair-setup <name>  →  workspaces/<name>/env_config.py
                          workspaces/<name>/designed_reward_fn.py
                          (after authoring the reward, show it back to the user)
/repair-init          →  workspaces/<name>/gather_evidence/trajectory_features.py
                          workspaces/<name>/candidate_performance_metrics/performance_metrics.py
/repair-round         →  one or more rounds of policies + rollouts + features,
                          then automatically continues into /repair-preferences.
                          Claude decides how many rounds to run (cap: 3 per
                          invocation) based on the sufficiency rubric in
                          .claude/skills/repair-round/SKILL.md.
/repair-preferences   →  Claude reads every collected trajectory (as
                          features, the same encoding the human sees) plus
                          the designed reward and stakeholder metrics, and
                          *itself selects* the most-telling pairs (up to
                          15) — this is the Claude-driven selection step,
                          not an algorithm. It renders each as a preference
                          label: feature key, anonymized A vs B, reward +
                          metric scores side-by-side with explicit A≻B /
                          B≻A labels, and a feature delta table. Plus
                          per-pair "why they disagree" summaries and an
                          overall pattern interpretation. Also snapshots the
                          pairs so /repair-verify can later re-check them.
                          (The legacy algorithmic selector still lives in
                          find_informative_pairs.py for reference and for
                          /repair-round's sufficiency probe.)
```

`/repair-init` runs once after `/repair-setup`. `/repair-round` is
append-only across invocations — each call adds the next k rounds and
ends in `/repair-preferences`. `/repair-preferences` can also be re-run
freely on its own; it just reloads everything from disk and re-scores
under the current reward.

## User-facing tone

Skills follow these tone rules in every message they send the user:

- **Never list internal filenames, directories, or boilerplate scripts.**
  Talk in terms of outcomes ("setting up the environment", "looking for
  evidence of misalignment", "surfacing the disagreements"), not
  implementation details.
- **Never refer to any behavior, reward, or trajectory as "correct" or
  "incorrect."** At most, say something is *possibly* correct/incorrect
  or that it *appears* aligned/misaligned with a particular stakeholder.
  Whether a behavior is actually correct is always the user's call —
  the system surfaces evidence, not verdicts.
- **Never name a candidate performance metric by its internal id**
  (e.g. `performance_metric_safety`). Either say "the candidate
  performance metric" or describe the stakeholder perspective from the
  docstring ("the safety reviewer's view", "the operations stakeholder").

## Skill chaining — always tell the user what to run next

Every skill must end by telling the user what they can run next. Never
finish a skill response without either (a) naming the next skill
explicitly, or (b) using `AskUserQuestion` to let the user pick from
the relevant options. The expected chain is:

- `/repair-setup` → tell the user to run `/repair-init`.
- `/repair-init` → tell the user to run `/repair-round`.
- `/repair-round` → after surfacing disagreements, **ask** the user
  whether they want to (1) run `/repair-round` again because the
  surfaced pairs were not informative, (2) modify the reward themselves,
  (3) ask Claude to modify the reward to satisfy the shown preferences,
  or (4) **Finish and submit (not reversible)**. Use `AskUserQuestion`.
- `/repair-preferences` → same closing question as `/repair-round`
  (another round / modify yourself / Claude modifies / Finish and submit).
  Use `AskUserQuestion`.
- `/repair-verify` → ask whether to look for new disagreements, collect
  more evidence first, or **Finish and submit (not reversible)**. Use
  `AskUserQuestion`.

### Finish and submit (not reversible)

When the user picks **Finish and submit**, treat the current
`designed_reward_fn.py` as the final answer: write
`workspaces/<name>/.repair_history/SUBMISSION.md` containing a UTC
timestamp and the full current reward source, tell the user the design
is submitted and the session is complete, and stop — do not offer
further steps, run more rounds, or edit the reward afterward. Because it
is irreversible, only do this when the user explicitly selects it.

### Reward-edit tips (print when asked to edit the reward)

Whenever the user asks Claude to modify the reward (the "Modify the
reward for me" path), print this tip box before requesting or applying
the edit:

> 💡 **Tips for guiding a reward edit**
> - Be concrete: name the term to change, and — if you can — the
>   preference label you want for each shown pair (e.g. "for pair 2,
>   A ≻ B should become B ≻ A").
> - You can pin an edit to an outcome: e.g. "increase the above-target
>   speed penalty until the preference for pair 2 flips," and Claude will
>   solve for the smallest change that produces it.

These rules apply to every skill in `.claude/skills/`. If a skill's own
`SKILL.md` doesn't repeat them, they still apply.

## Evaluation workspaces

Some workspaces are **evaluation workspaces**, used to evaluate the
repairer system against a human stakeholder. They follow the same
runtime invariants as any other workspace, but they hand the
reward-design task to the human as a *fixed assignment* instead of
eliciting it, and they constrain how Claude is allowed to edit the
reward function.

### Marker

A workspace is in evaluation mode iff `workspaces/<name>/eval_spec.md`
exists. That file contains the reward-design criteria the human is
required to design toward, written in plain language. Any future
evaluation workspace follows the same shape: a pre-authored
`env_config.py`, a pre-authored (intentionally misaligned)
`designed_reward_fn.py` as the human's starting point, a populated
`eval_spec.md`, and a pre-authored `evaluate_against_spec.py` (the spec
evaluator described below).

### Spec evaluation harness

`evaluate_against_spec.py` (repo root) scores the current
`designed_reward_fn.py` against `eval_spec.md` for an evaluation
workspace with a **ranking test**: the reward is aligned with the spec
when every spec-satisfying trajectory out-scores every violating one
(checkable criteria: no crash, nearest gap ≥ 10 m, average speed near
25 m/s). Because real rollouts rarely satisfy the strict criteria, it
synthesizes a guaranteed-compliant set by **ablating** each collected
trajectory (forcing ego speed → 25 m/s, every observed gap → ≥ 10 m, no
crash), then checks whether the reward ranks all compliant trajectories
above all real violating ones and reports the inversion count plus a
YES/NO. It lives **inside the eval workspace**
(`workspaces/<name>/evaluate_against_spec.py`) as a persistent
pre-authored eval artifact: a workspace reset must **keep it** (alongside
`env_config.py`, `eval_spec.md`, and the starting `designed_reward_fn.py`),
not treat it as authored/collected output. Run it from the repo root with
`python workspaces/<name>/evaluate_against_spec.py`.

### `/repair-setup` behavior in eval mode

When `/repair-setup <name>` is invoked on an evaluation workspace:

1. Skip env elicitation — `env_config.py` is pre-authored.
2. Skip reward-design elicitation — `designed_reward_fn.py` is
   pre-authored as the human's *starting point*, intentionally
   misaligned with the criteria in `eval_spec.md`.
3. Display the contents of `eval_spec.md` to the human as their
   reward-design task.
4. Display the contents of `designed_reward_fn.py` inline as
   "the reward function you've been given to revise."
5. Run the standard smoke test from the regular `/repair-setup` flow.
6. Exit by telling the user to run `/repair-init` next, same as the
   normal flow.

### Reward-edit gate

In an evaluation workspace, whenever the user asks Claude to modify
`designed_reward_fn.py` (either via the post-`/repair-round` or
post-`/repair-preferences` question, or organically mid-conversation),
Claude must first check whether the request specifies a **concrete
edit** — at least one of:

- a particular term to add or remove,
- a numerical weight or coefficient,
- a threshold, condition, or feature to gate on,
- a specific function shape (e.g. quadratic in distance, linear in
  speed, clipped above some value).

If the request is a **paraphrase of the criteria in `eval_spec.md`**
(i.e. a restatement of the reward-design goal with no implementation
detail) and does *not* contain any concrete edit, Claude must refuse to
make the edit. The refusal must:

1. Quote the relevant lines from `eval_spec.md` back to the user, so
   the user can see Claude has read their request as a restatement of
   the criteria they were given.
2. Ask for more specific guidance, naming the kind of guidance Claude
   needs — which term to change, what weight, under what condition,
   what functional shape.

Only after the user supplies a concrete edit may Claude modify
`designed_reward_fn.py`. This gate exists because the evaluation is
about whether the human can identify and articulate reward flaws —
not whether Claude can infer fixes from a restated objective.

## File layout

```
automated_reward_repairer/
├── CLAUDE.md
├── select_pairs.py                     # Claude-driven pair selection (dump + render)
├── find_informative_pairs.py           # legacy algorithmic selector; kept for reference + /repair-verify snapshot check
├── .current_workspace                  # name of the active workspace
│
├── runtime/                             # env/reward-agnostic boilerplate
│   ├── workspace.py                    # --workspace flag + sys.path/cwd setup
│   ├── env_loader.py                   # load env + gym/gymnasium compat shims
│   ├── trajectory_io.py                # pickle save/load
│   ├── rollout.py                      # run a policy in the env
│   ├── compute_features.py             # batch-apply Trajectory2Features
│   └── run_round.py                    # CLI: roll out + featurize one round
│
├── prompts/                             # frozen prompt text consumed by skills
│   ├── reward_design_template.md
│   ├── trajectory_features_prompt.md
│   ├── data_collection_policy_prompt.md
│   └── performance_metrics_prompt.md
│
├── .claude/skills/{repair-setup,repair-init,repair-round,repair-preferences,repair-verify}/SKILL.md
│
└── workspaces/
    └── <name>/                          # one directory per env (e.g. hopper)
        ├── env_config.py               # authored in /repair-setup
        ├── designed_reward_fn.py       # authored or stubbed in /repair-setup
        ├── requirements.txt            # optional, env-specific deps
        │
        ├── gather_evidence/
        │   ├── trajectory_features.py             # authored in /repair-init
        │   ├── data_collection_policy_round_1.py  # authored in /repair-round
        │   ├── data_collection_policy_round_2.py
        │   ├── ...
        │   ├── round_{k}_trajectories/            # pickle files, one per rollout
        │   └── round_{k}_trajectory_features/     # json files, one per rollout
        │
        └── candidate_performance_metrics/
            └── performance_metrics.py             # authored in /repair-init
```

## Invariants (do not break)

These contracts are what makes `runtime/` and `find_informative_pairs.py`
workspace-agnostic. Every authored file inside a workspace must conform.
Paths below are relative to the workspace root unless noted.

**`env_config.py`** exposes:
- `make_env()` → a gym/gymnasium-compatible env (reset/step). Wraps either
  `gym.make(...)` or a custom class.
- `OBS_DESCRIPTION: str` — human-readable per-dimension obs description.
- `ACTION_DESCRIPTION: str` — human-readable action-space description.
- `MAX_EPISODE_STEPS: int` — episode horizon used by rollout (default 1000
  if you can't infer one).

**`designed_reward_fn.py`** exposes:
- `designed_reward_fn(obs, act, next_obs) -> float`.

**`gather_evidence/trajectory_features.py`** exposes:
- `class Trajectory2Features` with:
  - `feature_names: list[tuple[str, str]]` — `(short_name, description)` per feature, declared on the class.
  - `get_traj_features(self, trajectory) -> list[float]` — trajectory is `[(obs, act, next_obs), ...]`.
  - `pretty_print_features(self, features) -> None` — uses `feature_names` for labels.

**`gather_evidence/data_collection_policy_round_{k}.py`** exposes:
- One or more classes whose names start with `DataCollectionPolicy`. Each:
  - `__init__(self, env)` — receives the env so it can read action/obs space.
  - `act(self, obs) -> action` — returns an action valid for `env.action_space`.
- Optionally a module-level `POLICIES = [Cls1, Cls2, ...]` list (preferred);
  if absent, `run_round.py` discovers classes by name prefix.

**`candidate_performance_metrics/performance_metrics.py`** exposes:
- One or more functions whose names start with `performance_metric_`. Each:
  - Signature `(traj) -> float`, where `traj = [(obs, act, next_obs), ...]`.
  - Docstring is the **stakeholder-perspective description** — shown to
    the user when this metric flags a disagreement. Keep it one line.

**Trajectory on disk (workspace-relative):**
- File: `gather_evidence/round_{k}_trajectories/{policy_class}__traj_{i}.pkl`
- Payload: `{"trajectory": [(obs, act, next_obs), ...], "metadata": {...}}`.

**Features on disk (workspace-relative):**
- File: `gather_evidence/round_{k}_trajectory_features/{same_basename}.json`
- Payload: `{"features": [...], "names": [[short, desc], ...], "source": "..."}`.

## How the workspace flag works

`runtime/workspace.py:activate(path)`:
1. resolves `path` (or reads `.current_workspace` if `path` is `None`),
2. prepends the workspace directory to `sys.path` so imports like
   `import env_config` and `from gather_evidence.trajectory_features import
   Trajectory2Features` resolve against the workspace,
3. chdirs into the workspace so relative paths in `trajectory_io.py` /
   `compute_features.py` land under the workspace.

Both CLI entrypoints (`runtime.run_round`, `find_informative_pairs.py`)
call `activate()` right after argparse. Run them from the repo root.

## When the user iterates

- **Edits `workspaces/<name>/designed_reward_fn.py`:** just rerun
  `/repair-preferences` (or `/repair-verify` to recheck previously
  snapshotted pairs). No new rollouts needed; reward totals are
  recomputed from saved trajectories.
- **Wants new evidence after editing the reward:** run `/repair-round` to
  add another round; policies in round k+1 see the *current* reward and
  prior rounds' policies, so they target the latest reward's flaws.
- **Wants to update features or metrics:** ask, then rewrite the relevant
  file under the workspace in place; re-run `/repair-preferences` (and
  re-run `runtime/compute_features.py` over existing rounds if features
  changed).

## Conventions for authored Python

- Imports at top, no `if __name__` blocks unless the file is a CLI script.
- Use `numpy` freely.
- Prefer plain functions/methods over deep class hierarchies.
- Per-feature/per-metric one-line docstrings are required (they are what
  the user sees during pair pretty-printing).
- Do not write trajectories or features yourself — call into `runtime/`.

## How to run the Python boilerplate

All Python in this repo runs in the `llm` conda env — `gymnasium`,
`highway_env`, and the other env dependencies are not installed in the
base interpreter. Activate it before any boilerplate script or smoke
test (the `/repair-*` skills already prepend this to their commands):

```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm
```

```
# from the repo root, with `llm` active:
python -m runtime.run_round --workspace workspaces/<name> --round K

# Claude-driven pair selection (the /repair-preferences step):
python select_pairs.py --workspace workspaces/<name> --mode dump
python select_pairs.py --workspace workspaces/<name> --mode render \
       --selection workspaces/<name>/.repair_history/_selection.json

# legacy algorithmic selector (reference + /repair-verify snapshot check):
python find_informative_pairs.py --workspace workspaces/<name> --top-overall 5 --margin 0.05

# spec evaluation harness (lives in the eval workspace; run from repo root):
python workspaces/<name>/evaluate_against_spec.py
```

If `.current_workspace` is set, `--workspace` may be omitted.
