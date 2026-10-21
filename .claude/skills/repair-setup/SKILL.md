---
name: repair-setup
description: Step 0+1 of the Automated Reward Repairer. Launches the web UI, sets up env_config.py and designed_reward_fn.py — either by eliciting a reward-design template from the user on the website or by generating a stub for the user to fill in. Invoke when the user is starting a new repairer session.
---

# /repair-setup — environment + reward function

This skill handles steps (0) and (1) of the spec. See `CLAUDE.md` for the
overall workflow, workspace layout, and the **Web UI** contract (the
user interacts through http://127.0.0.1:8000, not the terminal).

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Don't
mention internal file names, directory structure, or boilerplate scripts
in anything you post to the website. Talk in terms of **outcomes** —
"set up the environment", "identify flaws in your reward function" —
not implementation. Save file paths for tool calls.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with what you described." Whether something
is actually right is the user's call — the system surfaces evidence,
not verdicts. (See `CLAUDE.md` for the full tone rules.)

## Web UI is the only user-facing channel

Every message the user reads goes to the web UI via `python -m
web_ui.cli post`. Every question goes via `python -m web_ui.cli ask`,
and the agent reads the answer from the CLI's stdout. **Do not** call
the `AskUserQuestion` tool here, and do not write narrative to the
terminal beyond the one URL hint in Step 0a. See `CLAUDE.md` →
"Web UI — the user-facing channel" for the full reference.

## What to do

### Step 0a — launch the web UI (must come first)

```bash
python -m web_ui.cli launch
python -m web_ui.cli clear
```

Then print, **in the terminal only**, a single one-line hint so the
user knows where to open their browser:

> Open http://127.0.0.1:8000 — the rest of the session happens there.

### Step 0b — resolve the workspace

- `WORKSPACE = args` if provided, else read `.current_workspace`, else
  ask the user *on the website*:
  ```bash
  python -m web_ui.cli ask --prompt "Which workspace name do you want to use? (kebab-case, e.g. hopper, lunar-lander, my-task)" --input text
  ```
  Read `value` from the returned JSON.
- `WORKSPACE_DIR = workspaces/<WORKSPACE>`
- `mkdir -p <WORKSPACE_DIR>/gather_evidence <WORKSPACE_DIR>/candidate_performance_metrics`
- Create empty `__init__.py` in those two subdirectories.
- Write `<WORKSPACE>` to `.current_workspace` at the repo root.
- Tell the UI which workspace is active:
  ```bash
  python -m web_ui.cli set workspace=<WORKSPACE> status=running
  ```

### Step 1 — identify the environment

Ask the user **which gym environment** they want to use. Accept either:
- a gymnasium / gym id like `"Hopper-v4"` or `"CartPole-v1"`, or
- a path to a custom env class (`module:ClassName`).

If `<WORKSPACE_DIR>/env_config.py` already exists, confirm via the web UI
before overwriting — the user may have hand-crafted it:

```bash
python -m web_ui.cli ask --prompt "An environment is already set up for this workspace. Keep it, or replace it?" \
    --options '[{"label":"Keep the existing one"},{"label":"Replace it"}]'
```

If env_config.py is missing or being replaced, ask for the env id /
class via:

```bash
python -m web_ui.cli ask --prompt "Which environment do you want to use? (e.g. a gym id like 'Hopper-v4', or a 'module:ClassName' path.)" --input text
```

…then write `<WORKSPACE_DIR>/env_config.py` so it exports:
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

Verify it imports cleanly (the leading `conda activate` is required —
see "How to run the Python boilerplate" in `CLAUDE.md`):

```bash
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate $(cat workspaces/<WORKSPACE>/.conda_env 2>/dev/null||echo llm) && \
python -c "import sys; sys.path.insert(0, 'workspaces/<WORKSPACE>'); \
from env_config import make_env; e = make_env(); \
print(e.observation_space, e.action_space)"
```

Post a one-line status update saying the environment is set up — do not
list file names:

```bash
python -m web_ui.cli post --type status --content "The environment is set up."
```

### Step 2 — ask the user how they want to specify the reward

This is the **first reward decision in every session** — always elicit
the reward from the user. Do **not** assume a pre-existing
`designed_reward_fn.py` as a default to show back; if one already
exists in the workspace (e.g. a leftover from a prior session), it is
overwritten by whatever the user specifies here. Ask the user how they
want to author the reward:

```bash
python -m web_ui.cli ask --prompt "How would you like to specify the reward function?" \
    --options '[
      {"label":"Describe it — I will implement","description":"Fill in a short template; I translate it into Python and show it back for review."},
      {"label":"I will write it myself","description":"I show a stub with the observation/action summary as a docstring; you fill in the body here or in the file."}
    ]'
```

Parse `value` to decide the branch.

#### Branch 1a — "Describe it"

Compose a *plain-English* one-paragraph summary of what the agent
perceives and one paragraph of what it can do — rewritten from
`OBS_DESCRIPTION` / `ACTION_DESCRIPTION`, with no array indices, no
normalisation details, no internal flags. Post that summary, then ask
the user to fill in the reward-design template *as a structured form*
on the website — one labelled textarea per section:

```bash
# 1. Plain-language summary of obs + action (post as markdown).
python -m web_ui.cli post --type markdown --content "<your two paragraphs>"

# 2. Form-style question — three labelled boxes (achieve / avoid / tradeoff).
python -m web_ui.cli ask --prompt "Fill in the reward design template — one box per section." \
    --fields '[
      {"name":"achieve",
       "label":"What outcomes should the agent try to achieve?",
       "description":"List specific desired outcomes — what successful behavior looks like.",
       "placeholder":"e.g. reach the goal location; reach the goal quickly"},
      {"name":"avoid",
       "label":"What outcomes should the agent avoid?",
       "description":"List specific bad outcomes the agent should not produce.",
       "placeholder":"e.g. collisions with obstacles; excessive control effort"},
      {"name":"tradeoff",
       "label":"How should the agent trade off between different objectives?",
       "description":"Describe when it is acceptable to prioritise one outcome over another.",
       "placeholder":"e.g. never trade safety for speed; energy is fine if it means succeeding"}
    ]'
```

The returned `value` is a JSON object `{"achieve": "...", "avoid": "...",
"tradeoff": "..."}`. Use the three fields to author
`<WORKSPACE_DIR>/designed_reward_fn.py`. The file must:
- Take a single transition `(obs, action, next_obs)` and return a single
  float.
- Be a faithful translation of the user's filled template — every listed
  desired outcome contributes positively, every avoided outcome
  negatively, with the trade-offs the user described.
- Have a top-of-file comment containing the filled template verbatim, so
  the reward design is auditable from the source file.
- **Be commented inline.** Every weight constant and every intermediate
  variable with a physical/semantic meaning gets a one-line comment
  giving its **units** (m/s, fraction of steps, normalised position,
  dimensionless, etc.) and what it represents. Each contributing term
  gets a brief comment naming the outcome it credits or penalizes.

**Show the reward function back to the user on the website.** After
writing the file:

```bash
python -m web_ui.cli post --type code --language python \
    --caption "designed_reward_fn.py" \
    --content-file workspaces/<WORKSPACE>/designed_reward_fn.py

python -m web_ui.cli post --type markdown --content "Key knobs I chose: <one-line summary of the main weights, e.g. 'failure-state weight X vs +Y per step of progress (so a single failure dominates an entire episode of progress reward)'>"

python -m web_ui.cli ask --prompt "Does the reward function above look right, or should I adjust it?" \
    --options '[
      {"label":"Looks good"},
      {"label":"Adjust it","description":"Describe what to change next."}
    ]'
```

If the user picks "Adjust it", follow up with a long-text question
asking what to change, edit the file, re-post the new source, and
re-ask for confirmation. Loop until confirmed.

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

Post the obs/action summary, then show the stub in an editable Python
text box, naming the file so the user can edit it there instead:

```bash
python -m web_ui.cli post --type markdown --content "<plain-English obs + action summary>"
python -m web_ui.cli ask --input longtext \
    --prompt "Write your reward function below (Python), or paste an existing one. You can also edit the file directly at <absolute path to WORKSPACE_DIR>/designed_reward_fn.py and submit the box as is." \
    --prefill-file workspaces/<WORKSPACE>/designed_reward_fn.py
```

If the returned `value` differs from the file on disk, write it to
`<WORKSPACE_DIR>/designed_reward_fn.py`. Then post the file back as a
code message (`post --type code --language python --caption
"designed_reward_fn.py" --content-file ...`).

### Step 3 — sanity check

Smoke-test the reward (the result is for the agent only — do **not**
post the raw output to the website):

```bash
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate $(cat workspaces/<WORKSPACE>/.conda_env 2>/dev/null||echo llm) && \
python -c "import sys; sys.path.insert(0, 'workspaces/<WORKSPACE>'); \
from env_config import make_env; from designed_reward_fn import designed_reward_fn; \
import numpy as np; e = make_env(); \
obs = e.reset()[0] if isinstance(e.reset(), tuple) else e.reset(); \
a = e.action_space.sample(); \
step = e.step(a); next_obs = step[0]; \
print('reward sample:', designed_reward_fn(obs, a, next_obs))"
```

If branch 1b, `NotImplementedError` is expected — that's fine.

### Step 4 — close out with a Continue button, then chain to /repair-init

Don't tell the user to type a slash-command. Post a CTA button on the
website; when the user clicks it, the agent invokes the next skill
directly via the `Skill` tool.

```bash
ANSWER=$(python -m web_ui.cli cta \
    --prompt "The environment and reward function are set up." \
    --label  "Identify possible flaws" \
    --value  "repair-init")
NEXT=$(printf %s "$ANSWER" | python -c "import json,sys;print(json.load(sys.stdin)['value'])")
```

`cta` blocks until the user clicks. Once it returns, invoke the next
skill in the same agent turn:

```
Skill(skill="$NEXT")     # i.e. Skill tool with skill="repair-init"
```

Do not finish the skill turn without either (a) posting a CTA that
advances to the next step or (b) handing off via the Skill tool.
