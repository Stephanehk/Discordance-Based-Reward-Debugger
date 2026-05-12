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
/repair-preferences   →  prints the top 5 most-telling trajectory pairs as
                          preference labels: feature key, A vs B identifying
                          info (policy + round + steps), reward + metric
                          scores side-by-side with explicit A≻B / B≻A labels,
                          and a side-by-side feature delta table. Plus
                          per-pair "why they disagree" summaries and an
                          overall pattern interpretation. Also snapshots the
                          pairs so /repair-verify can later re-check them.
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
  or (3) ask Claude to modify the reward to satisfy the shown
  preferences. Use `AskUserQuestion`.
- `/repair-preferences` → same closing question as `/repair-round`
  (another round / modify yourself / Claude modifies). Use
  `AskUserQuestion`.
- `/repair-verify` → ask whether to look for new disagreements or
  collect more evidence first. Use `AskUserQuestion`.

These rules apply to every skill in `.claude/skills/`. If a skill's own
`SKILL.md` doesn't repeat them, they still apply.

## File layout

```
automated_reward_repairer/
├── CLAUDE.md
├── find_informative_pairs.py           # boilerplate, workspace-agnostic
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

```
# from the repo root:
python -m runtime.run_round --workspace workspaces/<name> --round K
python find_informative_pairs.py --workspace workspaces/<name>
python find_informative_pairs.py --workspace workspaces/<name> --top-k 5
```

If `.current_workspace` is set, `--workspace` may be omitted.
