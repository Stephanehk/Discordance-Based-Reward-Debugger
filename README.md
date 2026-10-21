# Discordance-Based Reward Debugger

> **Looking for the abstract-trajectory version?** The companion repo
> [Discordance-Based-Reward-Debugger-Abstract](https://github.com/Stephanehk/Discordance-Based-Reward-Debugger-Abstract)
> contains the codebase needed to debug reward functions with **abstract
> trajectories** (short text descriptions of runs instead of simulator
> rollouts) and **text-based reward functions** (natural-language rubrics
> instead of Python), as we did for the ExploitGym experiments.

A tool for debugging reward functions. The tool inputs a candidate reward function implemented in Python and the environment source code, and then construct comparisons where it hypothesizes your preferences will most likely differ from the ranking induced by candidate reward function. In lite of those hypotheses, you can update your reward function as you see fit. 

The tool implements policies, rolls them out in the environment to build a pool of trajectories, and then construct from that pool the comparisons. We found that in popular RL environments, such as Meta-World, this discordance-based reward debugger uncovers previously documented and novel reward misspecifications. Further, in ExploitGym, both the reward debugger uncovers a reward misspecification that was later exploited in practice, when a collective of AI agents attacked OpenAI and Hugging Face infrastructure while being evaluated on similar tasks.

For more details and results, check out our paper!

> **This tool is built on [Claude Code](https://claude.com/claude-code).**
> The whole workflow is a set of slash commands you run inside Claude
> Code; We developed and tested it with a **Claude Max
> subscription** running **Claude Opus 4.8**, and that is the setup we
> recommend — the quality of the comparisons depends on the model doing
> the reasoning. In Claude Code, select Opus with `/model`.

---

## What you need

1. **Claude Code** Every command below is run from this repo's root
   directory.
2. **Python 3.10+** with `numpy` and `gymnasium`. 
3. **A Gym/Gymnasium environment** that contains your environment source code. This environment can be
   installable (`gym.make("Hopper-v4")`) or your own Python implementation. This
   is the key artifact you must supply; see
   [Getting started](#getting-started-describe-your-environment) below.
4. **A reward function, or just an idea of one.** You can either input an implementation of a reward function for your environment, or describe a reward function in natural language when the tool asks you to. 

---

## Getting started: describe your environment

The only thing the tool needs from you to begin is a description of your
environment, written as a single file: **`workspaces/<name>/env_config.py`**.
Everything else is derived from it. This file must expose four things:

| Name | What it is |
|---|---|
| `make_env()` | returns a Gym/Gymnasium-compatible env (`reset`/`step`); wraps `gym.make(...)` or your own class |
| `OBS_DESCRIPTION: str` | plain-English description of **what every observation dimension means** |
| `ACTION_DESCRIPTION: str` | plain-English description of the action space |
| `MAX_EPISODE_STEPS: int` | the episode horizon |

The observation/action descriptions matter as much as the code: they are
how Claude understands your environment well enough to write sensible
probe policies and characteristics. Write them for a reader who has never
seen your env.

### You can ask Claude to generate it

You don't have to write `env_config.py` by hand. Point Claude Code at
your environment's source and ask it to generate the file. A prompt that
works well:

> I'm setting up the Discordance-Based Reward Debugger on a new environment. Read
> my environment's source code (below / at `<path>`) and write
> `workspaces/<name>/env_config.py` conforming to the tool's contract:
>
> - `make_env()` returning a Gym/Gymnasium-compatible env. If the env can
>   render RGB frames, make `make_env` accept a `render_mode` kwarg and
>   set `CAPTURE_FRAMES = True` at module level so rollouts record a video
>   per episode.
> - `OBS_DESCRIPTION: str` describing what **each** observation dimension
>   means, in plain English.
> - `ACTION_DESCRIPTION: str` describing the action space.
> - `MAX_EPISODE_STEPS: int` (the horizon).
>
> Also write the companion `workspaces/<name>/viz.py` (see below for how I
> want trajectories visualized). Follow the contracts in `CLAUDE.md`
> exactly, and model both files on the committed examples in
> `workspaces/merge/` (video) and `workspaces/pandemic/` (plots).
>
> My environment: `<paste the source, or give the import path / gym id>`

Several worked examples are committed in this repo, which you can use as templates:

- `workspaces/pandemic/env_config.py` — a custom simulator class with a
  rich, per-field `OBS_DESCRIPTION`.
- `workspaces/merge/env_config.py`, `workspaces/highway_two_way/env_config.py`
  — `highway-env` tasks.
- `workspaces/metaworld_drawer_open/`, `workspaces/metaworld_hammer/`,
  `workspaces/wipe/`, `workspaces/pointmaze_umaze/`,
  `workspaces/rocket_landing/`, `workspaces/robopianist/` — MuJoCo /
  Meta-World / robosuite tasks, several with `CAPTURE_FRAMES = True`.

### Decide how your trajectories should be visualized

You have three choices, and they are set up alongside `env_config.py`:

1. **A video of the real environment** — best when your env has a
   renderer. Set `CAPTURE_FRAMES = True` in `env_config.py` and make
   `make_env` accept `render_mode="rgb_array"`; rollouts then record one
   frame per step into an mp4, and a tiny `viz.py` just surfaces it.
   Examples: `workspaces/merge/viz.py`, `workspaces/metaworld_drawer_open/viz.py`.
2. **Rendered plots or images** — best for observation-rich envs with no
   natural picture. `viz.py` draws the trajectory itself (e.g. per-feature
   time series). Example: `workspaces/pandemic/viz.py` produces a stacked
   PNG of population curves and the regulation stage over time.
3. **Text / feature descriptions only** — if you omit `viz.py` entirely,
   each comparison still shows the plain-English list of how the two runs
   differ (higher/lower characteristic values). No picture, just the
   numbers.

The visualization lives in `workspaces/<name>/viz.py`, which exposes
`VIZ_EXT` (e.g. `".mp4"` or `".png"`) and
`visualize_trajectory(trajectory, metadata, out_path, source_path=None, **_)`.
The exact contract is in `CLAUDE.md`; the generation prompt above asks
Claude to write it for you.

---

## Usage

**Start here:**

1. Make sure Opus 4.8 is selected in Claude Code (`/model` → Opus).
2. Open Claude Code in this repo's root directory.
3. Run:
   ```
   /repair-setup <env_name>
   ```
   `<env_name>` is any short name you pick for this piece of work — e.g.
   `hopper`, `pandemic`, `merge`. It becomes a folder under `workspaces/`
   holding everything about this environment and this reward. You can have
   as many as you like side by side; they share nothing. (If you already
   pre-wrote `workspaces/<env_name>/env_config.py`, setup picks it up.)
4. That command prints **one** thing to your terminal: a link to
   **http://127.0.0.1:8000**.
5. **Open that link.** From then on, *everything* happens on that page —
   Claude posts its explanations, code, videos, and questions there, and
   you answer there. The terminal only shows status lines like
   "collecting trajectories…".

Note that some steps can take quite a while, i.e., up to an hour or longer! The speed of this tool depends on the speed of your environment, because it rolls out policies in that environment. We often recommend checking in on the tools progress while you do something else rather than standing by and waiting for it to complete. 

### What the page asks you

First, **how you want to specify the reward**. Either describe in plain
language what the agent should achieve, what it should avoid, and how to
trade those off — Claude writes the code and shows it to you for
approval — or say you'll write it yourself and fill in the stub it
leaves you.

After that, each step ends with a **Continue** button. Click it and the
next step starts. You never have to remember a command: the buttons walk
you from setup, through deciding what to measure about each run, through
collecting behavior, to the disagreements.

### What you get back

Once enough behavior has been collected, the page fills with
**discordances**: comparisons on which the tool hypothesizes your
preference differs from the ranking your reward function induces.

- **Pairs** (up to five). Two videos on top: on the left, the run *your
  reward function preferred*; on the right, the alternative, each with
  its return under your reward function. Underneath, a one-line "why
  they disagree", then two dropdowns: how the two runs differ in more
  detail (the most telling difference comes first), and how the return
  is computed. The first pair is expanded, the rest are collapsed.
- **Lotteries** (up to two). A gamble: "would you rather have this
  certain run C, or a probability distribution between run A and run B?"

### Deciding what to do after the tool shows you comparisons

Below the comparisons, the page asks you to pick one:

| Option | What happens |
|---|---|
| **Edit the reward function** | You change the code yourself, or you tell Claude in plain language what to change — including pinning it to an outcome, e.g. "increase this penalty until the alternative run wins discordance 2" — and Claude applies it. |
| **Add a trajectory characteristic** | The comparisons aren't showing you something you need in order to judge them. Name it, and it gets measured and added to every comparison's list of how the runs differ. It does not change the reward function. |
| **Propose more discordances** | Surface more pairs and lotteries from the behavior already collected. Click it twice in a row and the system goes and collects fresh behavior instead. |
| **End session** | Keeps the current reward function and closes the tool. |

After you edit the reward you have two ways forward, both offered as
buttons. **`/repair-verify`** re-scores the exact comparisons you were
already shown under the new reward and reports which disagreements your
edit fixed and which survived — no new runs, so it's fast.
**`/repair-round`** instead goes and collects *new* behavior, aimed at
whatever weak spots the new reward still has.

### The command list

You rarely need these — the buttons chain them for you — but for
reference:

```
/repair-setup <name>    define the environment and the reward
/repair-init            decide what to measure about each run
/repair-round           collect behavior, then show the disagreements
/repair-preferences     re-pick and re-show the disagreements
/repair-verify          after a reward edit: which disagreements are fixed?
```

`/repair-setup` and `/repair-init` run once per environment.
`/repair-round` can run as often as you like — each call adds to the
pile of collected behavior and never discards any.
`/repair-preferences` and `/repair-verify` can be re-run any time; they
re-read what's on disk and re-score it under whatever the reward
currently says.

---

## Repository layout

```
automated_reward_repairer/
├── CLAUDE.md                  # the contracts every authored file must satisfy
├── README.md
├── select_pairs.py            # dump the run table / render the chosen comparisons
├── find_informative_pairs.py  # legacy selector + snapshot re-scoring for /repair-verify
├── .current_workspace         # which workspace is active
│
├── runtime/                   # environment-agnostic plumbing
│   ├── workspace.py           #   the --workspace flag: import paths + working dir
│   ├── env_loader.py          #   build the env; smooth over gym/gymnasium differences
│   ├── rollout.py             #   run one policy for one episode
│   ├── compute_features.py    #   turn saved runs into characteristics
│   ├── run_round.py           #   CLI: roll out + featurize one round
│   └── run_round_parallel.py  #   the same, in parallel (used by the SLURM scripts)
│
├── prompts/                   # frozen prompt text the slash commands author from
│   ├── reward_design_template.md
│   ├── reward_decomposition_prompt.md
│   ├── trajectory_features_prompt.md
│   ├── data_collection_policy_prompt.md
│   └── performance_metrics_prompt.md
│
├── web_ui/                    # the local page: FastAPI server + the CLI skills post through
├── slurm_scripts/             # cluster rollouts for slow environments
├── docs/method_diagram.mmd    # diagram of the whole method (Mermaid source)
│
└── workspaces/<name>/         # everything about one environment + one reward
    ├── env_config.py                    # make_env(), obs/action descriptions, horizon
    ├── designed_reward_fn.py            # the reward under test
    ├── viz.py                           # optional: how to draw one run
    ├── .conda_env                       # optional: the conda env this workspace needs
    ├── gather_evidence/
    │   ├── trajectory_features.py
    │   ├── data_collection_policy_round_{k}.py
    │   ├── round_{k}_trajectories/          # one pickle per run
    │   └── round_{k}_trajectory_features/   # one json per run
    ├── candidate_performance_metrics/performance_metrics.py
    ├── .repair_history/                 # reward analysis, selections, snapshots
    └── viz/round{i}/{pair,lottery}{j}/  # the videos and plots you were shown
```

If you use it in your research, please cite the paper (citation and link
to be added on release).
