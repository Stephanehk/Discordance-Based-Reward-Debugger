---
name: repair-setup
description: Step 0+1 of the Automated Reward Repairer. Sets up env_config.py and designed_reward_fn.py — either by eliciting a reward-design template from the user or by generating a stub for the user to fill in. Invoke when the user is starting a new repairer session.
---

# /repair-setup — environment + reward function

This skill handles steps (0) and (1) of the spec. See `CLAUDE.md` for the
overall workflow and the workspace layout.

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Don't mention
internal file names, directory structure, or boilerplate scripts when
talking to them. Talk in terms of **outcomes** — "set up the environment",
"identify flaws in your reward function" — not implementation. Save file
paths for tool calls.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with what you described." Whether something
is actually right is the user's call — the system surfaces evidence,
not verdicts. (See `CLAUDE.md` for the full tone rules.)

## Workspaces

Each environment lives in its own workspace under `workspaces/<name>/`. The
boilerplate in `runtime/` and `find_informative_pairs.py` is env-agnostic
and is selected at runtime via `--workspace workspaces/<name>`.

This skill (and every other `/repair-*` skill) takes the workspace name as
`args` (e.g. `/repair-setup hopper`). After it runs it writes the name to
`.current_workspace` at the repo root so subsequent skills can default.

If the user didn't pass `args`, ask them for a workspace name before doing
anything else. Use kebab-case (e.g. `hopper`, `lunar-lander`, `my-task`).

## What to do

### 0. Resolve the workspace
- `WORKSPACE = args` if provided, else read `.current_workspace`, else ask.
- `WORKSPACE_DIR = workspaces/<WORKSPACE>`
- `mkdir -p <WORKSPACE_DIR>/gather_evidence <WORKSPACE_DIR>/candidate_performance_metrics`
- Create empty `__init__.py` in those two subdirectories.
- Write `<WORKSPACE>` to `.current_workspace` at the repo root.

### 1. Identify the environment
Ask the user **which gym environment** they want to use. Accept either:
- a gymnasium / gym id like `"Hopper-v4"` or `"CartPole-v1"`, or
- a path to a custom env class (`module:ClassName`).

If `<WORKSPACE_DIR>/env_config.py` already exists, ask before overwriting —
the user may have hand-crafted it.

Otherwise, write `<WORKSPACE_DIR>/env_config.py`. It must export:
- `make_env()` — returns the env. Use `gymnasium.make(...)` for ids, or
  import-and-instantiate for custom classes.
- `OBS_DESCRIPTION` — multi-line string describing each observation
  dimension. Use the env's documentation; if unclear, instantiate the env
  once and inspect `env.observation_space`.
- `ACTION_DESCRIPTION` — same for the action space (discrete values or
  per-dim meaning for Box).
- `MAX_EPISODE_STEPS` — int; use `env.spec.max_episode_steps` if present,
  else a reasonable cap (default 1000 if you really can't infer one).

If the env requires a non-default pip install, also write
`<WORKSPACE_DIR>/requirements.txt` listing those packages.

Verify it imports cleanly:
```
python -c "import sys; sys.path.insert(0, 'workspaces/<WORKSPACE>'); \
from env_config import make_env; e = make_env(); \
print(e.observation_space, e.action_space)"
```

To the user, say only that the environment is set up — don't recite file
names.

### 2. Ask the user how they want to specify the reward

> "Do you want to (a) describe what you want and let me implement the
> reward function, or (b) write it yourself?"

#### Branch 1a — "describe it"
Display `prompts/reward_design_template.md` inline. **Above the
template**, give the user a plain-language summary of what the
observation and action spaces measure — one short paragraph for each.
No technical jargon, no array indices, no normalisation details, no
internal flags or wrapper conventions. Just *what the agent perceives*
and *what it can do*, in everyday terms. Derive the summary from
`OBS_DESCRIPTION` and `ACTION_DESCRIPTION` in `env_config.py`, but
rewrite into plain English the user can read without prior context.
Ask the user to fill the template in.

Once they reply, write `<WORKSPACE_DIR>/designed_reward_fn.py` with:

```python
def designed_reward_fn(obs, act, next_obs) -> float:
    ...
```

The function must:
- Take a single transition (numpy obs, action, numpy next_obs) and return a
  single float.
- Be a faithful translation of the user's filled template — every listed
  desired outcome should contribute positively, every avoided outcome
  negatively, with the trade-offs the user described.
- Have a top-of-file comment containing the filled template verbatim, so
  the reward design is auditable from the source file.
- **Be commented inline.** Every weight constant and every intermediate
  variable that has a physical or semantic meaning gets a one-line
  comment giving its **units** (m/s, fraction of steps, normalised
  position, normalised velocity, dimensionless, etc.) and what it
  represents. Each term that contributes to the returned sum gets a
  brief comment naming the outcome it credits or penalizes — so the
  user can audit the reward by reading the source without re-deriving
  the semantics.

**Show the reward function back to the user.** After writing the file,
print its contents inline (wrap in a fenced code block, language = python)
and add a one-line note explaining the key knobs you chose (e.g.,
"I capped the primary-objective bonus and used a weight of −20 on the
failure-state penalty"). Ask the user to skim it and confirm before
proceeding. If they push back, edit and re-show.

#### Branch 1b — "I'll write it myself"
Write a stub `<WORKSPACE_DIR>/designed_reward_fn.py`:

```python
"""Designed reward for <env id>.

<paste env summary: obs dims with meanings, action space>

Fill in the body. Signature must be unchanged.
"""

def designed_reward_fn(obs, act, next_obs) -> float:
    raise NotImplementedError("Fill in designed_reward_fn before running /repair-init.")
```

Print the obs/action summary to the conversation so the user has it in
context while writing. Tell the user where the stub is and that the
system is ready when they're done filling it in.

### 3. Sanity check
After the file exists and (if 1a) is implemented, smoke-test it:
```
python -c "import sys; sys.path.insert(0, 'workspaces/<WORKSPACE>'); \
from env_config import make_env; from designed_reward_fn import designed_reward_fn; \
import numpy as np; e = make_env(); \
obs = e.reset()[0] if isinstance(e.reset(), tuple) else e.reset(); \
a = e.action_space.sample(); \
step = e.step(a); next_obs = step[0]; \
print('reward sample:', designed_reward_fn(obs, a, next_obs))"
```
If branch 1b, expect `NotImplementedError` and tell the user that's
expected — they need to implement the body, then continue.

### 4. Exit — always tell the user what to run next
Tell the user, in plain language, that the environment and reward
function are set up. Then **always** close with this line (or a very
close variant):

> "To begin identifying possible flaws in the specified reward function,
> run `/repair-init` next."

Do not finish this skill without explicitly naming `/repair-init` as the
next step. Do not list which files exist or what `/repair-init` does
internally.
