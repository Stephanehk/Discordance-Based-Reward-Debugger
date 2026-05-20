---
name: repair-init
description: Steps 2 + 6 of the Automated Reward Repairer. One-time after /repair-setup. Authors gather_evidence/trajectory_features.py and candidate_performance_metrics/performance_metrics.py. Invoke once after the env and reward function are in place.
---

# /repair-init — set up the misalignment-detection scaffolding

This skill is run **once** after `/repair-setup`. It builds the
trajectory-feature summarizer and the set of stakeholder performance
metrics that the system uses to detect reward misspecification.

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Don't mention
internal file names, directory structure, or boilerplate scripts when
talking to them. Talk in terms of outcomes — "set up the trajectory
summary", "identify stakeholder perspectives" — not file paths.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with what you described." Whether something
is actually right is the user's call — the system surfaces evidence,
not verdicts. (See `CLAUDE.md` for the full tone rules.)

## Eval mode — information barrier

If `<WORKSPACE_DIR>/eval_spec.md` exists, this is an evaluation
workspace. The features and performance metrics this skill authors must
be **spec-blind**: do not read or condition on `eval_spec.md`,
`evaluate_against_spec.py`, or any criteria-naming text. Delegate the
authoring to a subagent whose context holds only the env's obs/action
semantics, the verified denormalization facts, and the designed reward's
operational mechanics — never the criteria. Metrics must be independent
stakeholder perspectives justifiable from the environment alone. See
"Information barrier (eval mode)" in `CLAUDE.md`.

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-init hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, tell the user to run `/repair-setup` first.

`WORKSPACE_DIR = workspaces/<WORKSPACE>`. All files this skill writes go
into `<WORKSPACE_DIR>/...`.

## Preconditions

- `<WORKSPACE_DIR>/env_config.py` exists and `make_env()` works.
- `<WORKSPACE_DIR>/designed_reward_fn.py` exists and
  `designed_reward_fn(obs, act, next_obs)` returns a float (not
  `NotImplementedError`).

If either is missing or broken, tell the user to run `/repair-setup` first.

## Step 2: write `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py`

Read `prompts/trajectory_features_prompt.md` for the verbatim task text.
Use the following context to write the file:
- `env_config.OBS_DESCRIPTION` and `ACTION_DESCRIPTION`
- The filled reward-design template / source comment in
  `designed_reward_fn.py`

Must conform to the invariants in `CLAUDE.md`:
- `class Trajectory2Features` with `feature_names: list[tuple[str, str]]`,
  `get_traj_features(self, trajectory) -> list[float]`,
  `pretty_print_features(self, features) -> None`.
- `feature_names` and the list returned by `get_traj_features` must be the
  same length and in the same order.
- Features can be trajectory-level (e.g. "fraction of steps spent stationary")
  or transition-aggregates (e.g. "mean action magnitude").

Smoke-test after writing (run from the repo root):
```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python -c "import sys; sys.path.insert(0, 'workspaces/<WORKSPACE>'); \
from gather_evidence.trajectory_features import Trajectory2Features; \
from env_config import make_env; e = make_env(); \
obs = e.reset()[0] if isinstance(e.reset(), tuple) else e.reset(); \
a = e.action_space.sample(); next_obs = e.step(a)[0]; \
f = Trajectory2Features(); v = f.get_traj_features([(obs,a,next_obs)]*5); \
print(len(v), '==', len(f.feature_names)); f.pretty_print_features(v)"
```

## Step 6: write `<WORKSPACE_DIR>/candidate_performance_metrics/performance_metrics.py`

Read `prompts/performance_metrics_prompt.md` for the verbatim task text.
Use the env description and reward design as context.

Must conform to the invariants:
- One or more module-level functions whose names start with
  `performance_metric_`. Signature `(traj) -> float`.
- Each function's **docstring is one short line** describing the
  stakeholder perspective ("Energy efficiency: lower action magnitude is
  preferred"). This docstring is what the user sees during pair printing.
- Pick a count that fits the env — typically 3–6 diverse perspectives.
  Aim for genuinely different stakeholders, not minor variants of the
  designed reward.

Smoke-test: import the module and call each metric on a random trajectory.

## Exit — always tell the user what to run next

Tell the user, in plain language, that the system is now ready to look
for possible misalignment. **Always** close with this line (or a very
close variant):

> "To collect evidence of possible misalignment in your reward function,
> run `/repair-round` next."

Do not finish this skill without explicitly naming `/repair-round` as
the next step. Do not list which files were authored or describe their
contents.
