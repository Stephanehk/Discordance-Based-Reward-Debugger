---
name: baseline-init
description: Step 2 of the reward-design Baseline (a separate process from /repair-*; see BASELINE.md). One-time after /baseline-setup. Authors gather_evidence/trajectory_features.py for a baseline workspace. No performance metrics are authored.
---

# /baseline-init — set up the trajectory summary (baseline)

This skill is the baseline analogue of `/repair-init`, but it authors **only**
the trajectory-feature summarizer. The baseline has **no stakeholder
performance metrics** — it shows the user individual trajectories from the
reward-maximizing policy, not disagreements. See `BASELINE.md`.

Run **once** after `/baseline-setup`.

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Talk in terms of
outcomes — "set up the trajectory summary" — not file paths. **Never call any
behavior, reward, or trajectory "correct" or "incorrect."** (See `CLAUDE.md`
for the full tone rules; they apply here too.)

## Eval mode — information barrier

If `<WORKSPACE_DIR>/eval_spec.md` exists, this is an evaluation workspace. The
features this skill authors must be **spec-blind**: do not read or condition on
`eval_spec.md`, `evaluate_against_spec.py`, or any criteria-naming text.
Delegate the authoring to a subagent whose context holds only the env's
obs/action semantics, the verified denormalization facts, and the designed
reward's operational mechanics — never the criteria. See "Information barrier"
in `BASELINE.md` and `CLAUDE.md`.

(If this baseline workspace's `evaluate_against_spec.py` reads features by
specific short-names — e.g. `mean_speed` / `min_gap` / `crashed` — those
short-names are part of the env's measurement vocabulary, not the criteria;
the subagent may expose features under those names. Do not pass it the spec.)

## Resolve the workspace

Takes the workspace name as `args` (e.g. `/baseline-init hopper`). If `args` is
empty, read `.current_baseline_workspace` at the repo root; if that doesn't
exist, tell the user to run `/baseline-setup` first.

`WORKSPACE_DIR = baseline_workspaces/<WORKSPACE>`.

## Preconditions

- `<WORKSPACE_DIR>/env_config.py` exists and `make_env()` works.
- `<WORKSPACE_DIR>/designed_reward_fn.py` exists and returns a float (not
  `NotImplementedError`).

If either is missing or broken, tell the user to run `/baseline-setup` first.

## Before authoring — don't overwrite existing work

`/baseline-init` is normally a one-time step, but it may be invoked on a
workspace where `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py` already
exists. Before authoring it, check whether it is already present. If it exists,
do not silently overwrite it — use `AskUserQuestion` to ask the user whether to
**keep** the existing trajectory summary or **overwrite** it with a freshly
authored version. Frame this in plain language ("you already have a trajectory
summary set up"), not by filename. Then:

- **Keep** → skip authoring; leave it untouched (and skip the smoke test, since
  it is unchanged), then go to the exit step.
- **Overwrite** → author it fresh as described below.

If the file does not exist, proceed straight to authoring with no prompt.

## Author `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py`

Read `prompts/trajectory_features_prompt.md` for the verbatim task text. Use
`env_config.OBS_DESCRIPTION` / `ACTION_DESCRIPTION` and the designed reward's
operational mechanics as context.

Must conform to the invariants in `CLAUDE.md`:
- `class Trajectory2Features` with `feature_names: list[tuple[str, str]]`,
  `get_traj_features(self, trajectory) -> list[float]`,
  `pretty_print_features(self, features) -> None`.
- `feature_names` and the list returned by `get_traj_features` are the same
  length and order.

Smoke-test after writing (from the repo root):
```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python -c "import sys; sys.path.insert(0, 'baseline_workspaces/<WORKSPACE>'); \
from gather_evidence.trajectory_features import Trajectory2Features; \
from env_config import make_env; e = make_env(); \
obs = e.reset()[0] if isinstance(e.reset(), tuple) else e.reset(); \
a = e.action_space.sample(); next_obs = e.step(a)[0]; \
f = Trajectory2Features(); v = f.get_traj_features([(obs,a,next_obs)]*5); \
print(len(v), '==', len(f.feature_names)); f.pretty_print_features(v)"
```

**Do not author any `performance_metrics.py`.** The baseline has no
`candidate_performance_metrics/` directory.

## Exit — always tell the user what to run next

Tell the user, in plain language, that the system is ready to roll out the
reward-maximizing behavior and show it back. **Always** close with this line
(or a very close variant):

> "To roll out the reward-maximizing behavior and see what it produces, run
> `/baseline-evidence` next."

Do not finish without explicitly naming `/baseline-evidence`.
