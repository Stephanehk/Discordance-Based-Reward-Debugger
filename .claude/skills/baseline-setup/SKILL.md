---
name: baseline-setup
description: Step 1 of the reward-design Baseline (a separate process from /repair-*; see BASELINE.md). Sets up env_config.py and designed_reward_fn.py for a baseline workspace — either by eliciting a reward-design template from the user or by generating a stub for the user to fill in. Invoke when starting a new baseline session.
---

# /baseline-setup — environment + reward function (baseline)

This skill is the baseline analogue of `/repair-setup`. The baseline is a
**separate process** from the Automated Reward Repairer — see `BASELINE.md`
for the full workflow and layout. It writes to `baseline_workspaces/<name>/`,
not `workspaces/<name>/`, and never touches `.current_workspace` or any
`/repair-*` file.

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Don't mention
internal file names, directory structure, or boilerplate scripts when talking
to them. Talk in terms of **outcomes** — "set up the environment", "specify
your reward function" — not implementation. Save file paths for tool calls.

**Never call any behavior, reward, or trajectory "correct" or "incorrect."**
At most, frame things as *possibly* correct/incorrect or "appears
aligned/misaligned with what you described." Whether something is actually
right is the user's call. (See `CLAUDE.md` for the full tone rules; they apply
to baseline skills too.)

## Workspaces

Each environment lives in its own baseline workspace under
`baseline_workspaces/<name>/`. The boilerplate in `runtime/` and
`show_trajectories.py` is env-agnostic and is selected at runtime via
`--workspace baseline_workspaces/<name>` (always pass the full path — a bare
name resolves to `workspaces/<name>`).

This skill (and every other `/baseline-*` skill) takes the workspace name as
`args` (e.g. `/baseline-setup hopper`). After it runs it writes the name to
`.current_baseline_workspace` at the repo root so subsequent baseline skills
can default.

If the user didn't pass `args`, ask them for a workspace name before doing
anything else. Use kebab-case.

## What to do

### 0. Resolve the workspace
- `WORKSPACE = args` if provided, else read `.current_baseline_workspace`, else ask.
- `WORKSPACE_DIR = baseline_workspaces/<WORKSPACE>`
- `mkdir -p <WORKSPACE_DIR>/gather_evidence`
- Create an empty `__init__.py` in `<WORKSPACE_DIR>/gather_evidence`.
- Write `<WORKSPACE>` to `.current_baseline_workspace` at the repo root.

### 1. Identify the environment
Ask the user **which gym environment** they want to use. Accept either a
gymnasium / gym id (`"Hopper-v4"`) or a path to a custom env class
(`module:ClassName`).

If `<WORKSPACE_DIR>/env_config.py` already exists, ask before overwriting.

Otherwise write `<WORKSPACE_DIR>/env_config.py` exporting `make_env()`,
`OBS_DESCRIPTION`, `ACTION_DESCRIPTION`, and `MAX_EPISODE_STEPS` exactly as the
invariants in `CLAUDE.md` require. If the env needs a non-default pip install,
also write `<WORKSPACE_DIR>/requirements.txt`.

Verify it imports cleanly:
```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python -c "import sys; sys.path.insert(0, 'baseline_workspaces/<WORKSPACE>'); \
from env_config import make_env; e = make_env(); \
print(e.observation_space, e.action_space)"
```

To the user, say only that the environment is set up.

### 2. Ask the user how they want to specify the reward

> "Do you want to (a) describe what you want and let me implement the reward
> function, or (b) write it yourself?"

#### Branch 2a — "describe it"
Display `prompts/reward_design_template.md` inline. **Above the template**,
give the user a plain-language summary of what the observation and action
spaces measure — one short paragraph each, in everyday terms, derived from
`OBS_DESCRIPTION` / `ACTION_DESCRIPTION` but with no array indices,
normalisation details, or wrapper conventions. Ask the user to fill it in.

Once they reply, write `<WORKSPACE_DIR>/designed_reward_fn.py` with signature
`designed_reward_fn(obs, act, next_obs) -> float`. It must be a faithful
translation of the filled template, carry the filled template verbatim as a
top-of-file comment, and be commented inline (units + meaning on every weight
and intermediate, the outcome each term credits/penalizes).

**Show the reward function back to the user**: print its contents inline
(fenced, language = python), note the key knobs you chose in one line, and ask
the user to skim and confirm. If they push back, edit and re-show.

#### Branch 2b — "I'll write it myself"
Write a stub `<WORKSPACE_DIR>/designed_reward_fn.py` with the env summary in
the docstring and a body that raises `NotImplementedError`. Print the
obs/action summary to the conversation and tell the user the system is ready
once they've filled it in.

### 3. Sanity check
Smoke-test the reward:
```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python -c "import sys; sys.path.insert(0, 'baseline_workspaces/<WORKSPACE>'); \
from env_config import make_env; from designed_reward_fn import designed_reward_fn; \
import numpy as np; e = make_env(); \
obs = e.reset()[0] if isinstance(e.reset(), tuple) else e.reset(); \
a = e.action_space.sample(); \
step = e.step(a); next_obs = step[0]; \
print('reward sample:', designed_reward_fn(obs, a, next_obs))"
```
If branch 2b, expect `NotImplementedError` and tell the user that's expected.

### 4. Exit — always tell the user what to run next
Tell the user, in plain language, that the environment and reward function are
set up. Then **always** close with this line (or a very close variant):

> "To set up how trajectories will be summarized for you, run `/baseline-init`
> next."

Do not finish without explicitly naming `/baseline-init`. Do not list which
files exist.
