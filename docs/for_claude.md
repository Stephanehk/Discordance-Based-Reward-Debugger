# Installing the Discordance-Based Reward Debugger (instructions for Claude Code)

You are Claude Code, opened in a user's project directory that contains the
source code of a reinforcement-learning environment. The user asked you to
install the reward debugging tool for a named task (e.g. "for task
drawer-open"). Follow the steps below in order. Ask the user to confirm at
the two marked points; otherwise proceed on your own.

Everything you write goes inside the cloned repo, never into the user's own
source tree. Call the directory you were opened in `PROJECT_DIR` (absolute
path) and the clone `REPO_DIR` = `PROJECT_DIR/Discordance-Based-Reward-Debugger`.

## 1. Clone and install

```bash
git clone https://github.com/Stephanehk/Discordance-Based-Reward-Debugger.git
pip install -r Discordance-Based-Reward-Debugger/requirements.txt
```

Skip the clone if `REPO_DIR` already exists. Use the same Python that runs
the user's environment. If this file is newer in the clone than the copy you
are reading, follow the clone's copy at `REPO_DIR/docs/for_claude.md`.

## 2. Find the environment

Read the user's source code and identify the Gym/Gymnasium environment for
the task the user named: the class (or registered id), how to construct it,
the observation and action spaces, and the episode horizon. If the name
matches nothing, or matches several environments, ask the user which one
they mean.

Derive the workspace name from the task name (lowercase, underscores), for
example `drawer_open`. Call it `NAME`.

## 3. Find the reward function

Find where the environment computes its per-step reward (usually inside
`step`). Rewrite it as a pure function of a single transition:

```python
def designed_reward_fn(obs, act, next_obs) -> float:
```

It may use only `obs`, `act`, `next_obs`, and constants. If the original
reward reads simulator state that is not in the observation, make
`make_env()` (step 5) wrap the environment and append those quantities to
the end of the observation, then read them from `next_obs` here. Document
any appended dimensions in `OBS_DESCRIPTION`.

**Always ask the user to confirm (1 of 2), and wait for the answer.**
Never assume the reward you found is the one they mean. Show them the
original code you found, where in their codebase it lives, and the
`designed_reward_fn` you wrote from it, then ask: "Is this the reward
function you want debugged?" If the codebase has several candidates, list
them all and ask which one. Do not write any workspace files until the
user has said yes; if they say no, ask them to point you to the right
function or paste it, and show the result back for confirmation again.

## 4. Decide how to visualize trajectories

Pick one, in this order of preference, and **ask the user to confirm
(2 of 2)**:

1. **Video of the real environment**, if the environment can render RGB
   frames (`render_mode="rgb_array"`). Set `CAPTURE_FRAMES = True` in
   `env_config.py` and make `make_env` accept a `render_mode` keyword. The
   `viz.py` then just copies the recorded video:

   ```python
   import os, shutil
   from runtime.trajectory_io import video_path_for

   VIZ_EXT = ".mp4"

   def visualize_trajectory(trajectory, metadata, out_path, source_path=None, **_):
       sidecar = video_path_for(source_path)
       os.makedirs(os.path.dirname(out_path), exist_ok=True)
       shutil.copyfile(sidecar, out_path)
   ```

2. **Plots**, if there is no renderer: `viz.py` draws the trajectory
   itself (for example, time series of the most meaningful observation
   dimensions) and saves a PNG. `VIZ_EXT = ".png"`.

3. **Text only**: omit `viz.py`. Each comparison then shows only the list
   of how the two trajectories differ.

## 5. Write the workspace

Create `REPO_DIR/workspaces/NAME/` with:

- `env_config.py` exposing
  - `make_env(render_mode=None)` returning the environment (add
    `sys.path.insert(0, "PROJECT_DIR")` at the top so the user's code
    imports; use the absolute path),
  - `OBS_DESCRIPTION: str`, a plain-English meaning for every observation
    dimension,
  - `ACTION_DESCRIPTION: str`, a plain-English description of the action
    space,
  - `MAX_EPISODE_STEPS: int`, the horizon,
  - `CAPTURE_FRAMES = True` if you chose video.
- `designed_reward_fn.py` from step 3.
- `viz.py` from step 4 (unless text only).

Then write the workspace name to `REPO_DIR/.current_workspace` (just
`NAME`, no newline needed).

Smoke-test from `REPO_DIR`:

```bash
python -c "import sys; sys.path.insert(0, 'workspaces/NAME'); \
from env_config import make_env, MAX_EPISODE_STEPS; from designed_reward_fn import designed_reward_fn; \
e = make_env(); r = e.reset(); obs = r[0] if isinstance(r, tuple) else r; \
a = e.action_space.sample(); s = e.step(a); \
print('reward sample:', designed_reward_fn(obs, a, s[0]), 'horizon:', MAX_EPISODE_STEPS)"
```

If you chose video, also check that `make_env(render_mode="rgb_array").render()`
returns an image array after `reset()`. Fix anything that fails.

## 6. Start the tool and keep going in this session

From `REPO_DIR`:

```bash
python -m web_ui.cli launch
python -m web_ui.cli clear
python -m web_ui.cli set workspace=NAME status=awaiting
python -m web_ui.cli post --type markdown --content "The environment and reward function for **NAME** are set up."
```

(`launch` is fine to run when a server is already up; `clear` empties
the page of anything from an earlier session.) Tell the user, in the
terminal, that the tool's page is at **http://127.0.0.1:8000** and that
everything from here on happens there.

Then run the tool yourself, in this same session. Its steps are written
as instruction files in the clone, not as commands you can call from the
user's directory, so read them and carry them out:

1. `cd REPO_DIR` and keep that as your working directory from now on;
   every path in the instruction files is relative to it.
2. Read `CLAUDE.md` and follow its rules for the rest of the session.
3. Read `.claude/skills/repair-init/SKILL.md` and carry it out.
4. Whenever an instruction file says to invoke another skill (for example
   `Skill(skill="repair-round")`, or `/repair-round`), read that skill's
   `SKILL.md` under `.claude/skills/` and carry it out instead.

Do not run or follow `/repair-setup`; this installation replaced it. Do
not tell the user to open a new Claude Code session.
