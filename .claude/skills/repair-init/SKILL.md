---
name: repair-init
description: Steps 2 + 6 of the Automated Reward Repairer. One-time after /repair-setup. Authors gather_evidence/trajectory_features.py and candidate_performance_metrics/performance_metrics.py. Invoke once after the env and reward function are in place.
---

# /repair-init — set up the misalignment-detection scaffolding

This skill is run **once** after `/repair-setup`. It builds the
trajectory-feature summarizer and the set of alternative-perspective
performance metrics that the system uses to detect reward
misspecification.

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Don't
mention internal file names, directory structure, or boilerplate scripts
in anything posted to the website. Talk in terms of outcomes —
"set up the trajectory summary", "identify other perspectives" — not
file paths. Do not use the word "stakeholder" in messages posted to
the user; say "other perspectives" or describe the specific
perspective.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with what you described." Whether something
is actually right is the user's call — the system surfaces evidence,
not verdicts. (See `CLAUDE.md` for the full tone rules.)

## Web UI is the only user-facing channel

The web UI was launched by `/repair-setup`. Every message the user
reads here goes through `python -m web_ui.cli post`; every question
goes through `python -m web_ui.cli ask`. **Do not** call the
`AskUserQuestion` tool, and do not narrate to the terminal. See
`CLAUDE.md` → "Web UI — the user-facing channel" for the full
reference.

If the server isn't running for some reason (the user invoked
`/repair-init` without first running `/repair-setup`, etc.), bring it
up first: `python -m web_ui.cli launch`.

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-init hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, post a message telling the user to run `/repair-setup`
first and stop.

`WORKSPACE_DIR = workspaces/<WORKSPACE>`. All files this skill writes go
into `<WORKSPACE_DIR>/...`. Update the UI header:

```bash
python -m web_ui.cli set workspace=<WORKSPACE> status=running
```

## Preconditions

- `<WORKSPACE_DIR>/env_config.py` exists and `make_env()` works.
- `<WORKSPACE_DIR>/designed_reward_fn.py` exists and
  `designed_reward_fn(obs, act, next_obs)` returns a float (not
  `NotImplementedError`).

If either is missing or broken, post a message asking the user to run
`/repair-setup` first and stop.

## Before authoring — don't overwrite existing work

`/repair-init` is normally a one-time step, but it may be invoked on a
workspace where these files already exist. Before authoring either file,
check whether it is present:

- `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py`
- `<WORKSPACE_DIR>/candidate_performance_metrics/performance_metrics.py`

If **either** exists, do not silently overwrite. Ask the user via the
web UI, in plain language — "you already have a trajectory summary set
up" / "you already have other perspectives set up" — not by
filename. Example:

```bash
python -m web_ui.cli ask --prompt "A trajectory summary is already set up for this workspace. Keep it, or replace it with a freshly authored one?" \
    --options '[{"label":"Keep the existing one"},{"label":"Replace it"}]'
```

- **Keep** → skip authoring that file; leave it untouched (and skip its
  smoke test, since it is unchanged).
- **Replace it** → author it fresh as described below.

If neither file exists, proceed straight to authoring with no prompt.

## Step 1.5: read the reward function before authoring

Feature authoring is **reward function aware**: before writing the feature set,
read `<WORKSPACE_DIR>/designed_reward_fn.py` (the design-template comment
and the code body) and enumerate the terms it uses to produce a reward. The
features you write must let the user see how each term changes (the verbatim
requirement is in `prompts/trajectory_features_prompt.md`).

This stage does **not** write the persisted reward decomposition
(`CLAUDE.md` → "Reward introspection"); that artifact is produced later,
by `/repair-round` and `/repair-preferences`, where the policy-authoring
and selection logic need it. Keeping feature authoring driven only by the
reward source (not the decomposition) is deliberate.

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
- Features can be trajectory-level (e.g. "fraction of steps spent
  stationary") or transition-aggregates (e.g. "mean action magnitude").

The mandatory reward-term coverage requirement (every term measured;
proxy and true outcome exposed separately; feature→term mapping comment)
lives in `prompts/trajectory_features_prompt.md` — follow it there.

Smoke-test after writing (run from the repo root):

```bash
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate $(cat workspaces/<WORKSPACE>/.conda_env 2>/dev/null||echo llm) && \
python -c "import sys; sys.path.insert(0, 'workspaces/<WORKSPACE>'); \
from gather_evidence.trajectory_features import Trajectory2Features; \
from env_config import make_env; e = make_env(); \
obs = e.reset()[0] if isinstance(e.reset(), tuple) else e.reset(); \
a = e.action_space.sample(); next_obs = e.step(a)[0]; \
f = Trajectory2Features(); v = f.get_traj_features([(obs,a,next_obs)]*5); \
print(len(v), '==', len(f.feature_names)); f.pretty_print_features(v)"
```

The result is for the agent's verification only — do **not** post the
raw smoke-test output to the website.

## Step 6: write `<WORKSPACE_DIR>/candidate_performance_metrics/performance_metrics.py`

Read `prompts/performance_metrics_prompt.md` for the verbatim task text.
Use the env description and reward design as context.

Must conform to the invariants:
- One or more module-level functions whose names start with
  `performance_metric_`. Signature `(traj) -> float`.
- Each function's **docstring is one short line** describing the
  alternative perspective ("Energy efficiency: lower action magnitude is
  preferred"). This docstring is what the user sees during pair printing.
- Pick a count that fits the env — typically 3–6 diverse perspectives.
  Aim for genuinely different perspectives, not minor variants of the
  designed reward.

Smoke-test by importing the module and calling each metric on a random
trajectory.

## Step 7: confirm to the user via the web UI

Post a short status update naming what was set up — talk in terms of
outcomes (trajectory summary, other perspectives), never file
names:

```bash
python -m web_ui.cli post --type status --content "The trajectory summary and other perspectives are in place."
```

## Exit — Continue button that chains to /repair-round

Post a CTA button on the website and invoke the next skill when the
user clicks it.

```bash
ANSWER=$(python -m web_ui.cli cta \
    --prompt "The system is ready to look for possible misalignment." \
    --label  "Collect evidence of possible misalignment" \
    --value  "repair-round")
NEXT=$(printf %s "$ANSWER" | python -c "import json,sys;print(json.load(sys.stdin)['value'])")
```

Then call the `Skill` tool with `skill="$NEXT"` (i.e. `repair-round`).
Do not finish the turn without either posting the CTA or handing off
via the Skill tool. Do not describe which files were authored.
