# Discordant-Based Reward Debugger

A tool for checking whether the reward function you wrote actually says
what you meant.

The name says how it works: it hunts for **discordant** runs — pairs
where your reward function disagrees with how you'd rank them — and puts
them in front of you to debug.

If you train an agent in a simulated environment, you have to write down
a **reward function**: a piece of code that scores how good each moment
of behavior was. Getting that code to match what you actually want is
famously hard. A reward can look perfectly reasonable and still prefer
behavior you would reject on sight.

This tool shows you that behavior. It runs many different policies in
your environment, records what happened in each run, and then hunts for
**pairs of runs where your reward function picks the one you probably
wouldn't**. It shows you those pairs side by side — usually as two
videos — with a plain-English account of how the two runs differ. You
look, and you decide whether your reward got it right. If it didn't, you
(or Claude) edit the reward, and the tool re-checks the same comparisons
under the new version.

The comparisons are the product. Nothing here trains a policy or learns
a reward for you — think of it as a debugger for reward functions.

> **This tool is built on [Claude Code](https://claude.com/claude-code).**
> The whole workflow is a set of slash commands you run inside Claude
> Code; Claude is what reads your reward, writes the probe policies, and
> picks the comparisons. We developed and tested it with a **Claude Max
> subscription** running **Claude Opus 4.8**, and that is the setup we
> recommend — the quality of the comparisons depends on the model doing
> the reasoning. In Claude Code, select Opus with `/model`.

---

## What you need

1. **Claude Code**, with a **Max subscription** and **Opus 4.8** selected
   (`/model` → Opus). Every command below is run from this repo's root
   directory.
2. **Python 3.10+** with `numpy` and `gymnasium`. A fresh virtual
   environment is the simplest:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (A conda environment works equally well.) Some RL environments need
   heavier dependencies — MuJoCo, Meta-World, highway-env, … — which you
   install into the same environment, or into a dedicated one. If a
   workspace needs a *different* interpreter than your default, drop a
   one-line `.conda_env` file naming the conda env it needs into the
   workspace folder; the slash commands read it and activate that env for
   you (absent → your current environment is used).
3. **A Gym/Gymnasium environment** to inspect — either something
   installable (`gym.make("Hopper-v4")`) or your own Python class. This
   is the one thing you must supply; see
   [Getting started](#getting-started-describe-your-environment) below.
4. **A reward function, or just an idea of one.** You don't have to write
   the code — setup can interview you and write it for you.
5. **A browser.** All the reading and deciding happens on a local web
   page, not in the terminal.

---

## Getting started: describe your environment

The only thing the tool needs from you to begin is a description of your
environment, written as a single file: **`workspaces/<name>/env_config.py`**.
Everything else (the reward, what to measure, the probe policies, the
comparisons) is derived from it. This file must expose four things:

| Name | What it is |
|---|---|
| `make_env()` | returns a Gym/Gymnasium-compatible env (`reset`/`step`); wraps `gym.make(...)` or your own class |
| `OBS_DESCRIPTION: str` | plain-English description of **what every observation dimension means** |
| `ACTION_DESCRIPTION: str` | plain-English description of the action space |
| `MAX_EPISODE_STEPS: int` | the episode horizon (use `1000` if you can't infer one) |

The observation/action descriptions matter as much as the code: they are
how Claude understands your environment well enough to write sensible
probe policies and characteristics. Write them for a reader who has never
seen your env.

### You can ask Claude to generate it

You don't have to write `env_config.py` by hand. Point Claude Code at
your environment's source and ask it to generate the file. A prompt that
works well:

> I'm setting up the Discordant-Based Reward Debugger on a new environment. Read
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
> - `MAX_EPISODE_STEPS: int` (the horizon; use 1000 if unsure).
>
> Also write the companion `workspaces/<name>/viz.py` (see below for how I
> want trajectories visualized). Follow the contracts in `CLAUDE.md`
> exactly, and model both files on the committed examples in
> `workspaces/merge/` (video) and `workspaces/pandemic/` (plots).
>
> My environment: `<paste the source, or give the import path / gym id>`

Several worked examples are committed in this repo — read them before you
start, and use them as templates:

- `workspaces/pandemic/env_config.py` — a custom simulator class with a
  rich, per-field `OBS_DESCRIPTION`.
- `workspaces/merge/env_config.py`, `workspaces/highway_two_way/env_config.py`
  — `highway-env` tasks.
- `workspaces/metaworld_drawer_open/`, `workspaces/metaworld_hammer/`,
  `workspaces/wipe/`, `workspaces/pointmaze_umaze/`,
  `workspaces/rocket_landing/`, `workspaces/robopianist/` — MuJoCo /
  Meta-World / robosuite tasks, several with `CAPTURE_FRAMES = True`.

### Decide how your trajectories should be shown to you

Part of describing your environment is saying **how you want a single run
rendered** on the review page — this is what you actually look at when you
decide whether the reward got a comparison right. You have three choices,
and they are set up alongside `env_config.py`:

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

> **Prefer to be interviewed?** You don't need to pre-write any of this.
> Running `/repair-setup <name>` (next section) walks you through creating
> `env_config.py` (and the reward) interactively and launches the review
> page for you. Pre-writing `env_config.py` is simply the fast path if you
> already know your environment inside out.

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
**comparisons**:

- **Pairs** (up to five). Two videos on top: on the left, the run *your
  reward function preferred*; on the right, the alternative. Underneath,
  a one-line "why they disagree", then a short list of how the preferred
  run differed — "higher time in lockdown (43 vs. 11 days)", "lower peak
  infections (…)" — ordered so the most telling difference comes first.
  The first pair is expanded, the rest are collapsed.
- **Lotteries** (up to two). A gamble: "would you rather have this
  certain run C, or a coin-flip between run A and run B?" These catch
  cases where your reward is more (or less) willing to take risks than
  you are — something head-to-head pairs can't show.

### Deciding what to do next

Below the comparisons, the page asks you to pick one:

| Option | What happens |
|---|---|
| **Edit the reward function** | You change the code yourself, or you tell Claude in plain language what to change — including pinning it to an outcome, e.g. "increase this penalty until the alternative run wins pair 2" — and Claude applies it. |
| **Add another characteristic** | The comparisons aren't showing you something you need in order to judge them. Name it, and it gets measured and folded into every comparison. |
| **View more comparisons** | Surface more pairs and lotteries from the behavior already collected. Click it twice in a row and the system goes and collects fresh behavior instead. |
| **Finish and submit** | Locks the current reward in as your final answer and ends the session. Not reversible — pick it only when you're done. |

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

## Methods

The premise: a misspecified reward gives itself away by *ranking two
concrete runs the wrong way round*. So the system (1) collects runs
diverse enough to contain such a ranking, (2) scores every run from
several different points of view so it can spot where your reward is the
odd one out, and (3) hands the resulting disagreements to you in a form
you can judge in seconds.

Everything environment-specific lives in `workspaces/<name>/`; all the
machinery around it (`runtime/`, `select_pairs.py`, `prompts/`, the
slash commands) is environment-agnostic and gets pointed at a workspace
at run time. The active workspace name is remembered in
`.current_workspace`.

### Vocabulary

- **Run / trajectory** — one episode in the environment, recorded as the
  list of (observation, action, next observation) at every step.
- **Characteristic / feature** — one interpretable number summarizing a
  whole run ("peak infections", "days of lockdown"). These are what you
  read; you never look at raw observation vectors.
- **Designed reward** — your reward function, the thing under test.
- **Alternative-perspective metrics** — several *other* ways of scoring
  a whole run, each written from a different legitimate point of view
  (task success, energy use, safety, side effects…). They are the
  system's search signal: wherever your reward and one of these rank two
  runs oppositely, there is something worth showing you. They are
  deliberately **never shown to you**, because they are guesses about
  what you might care about, not verdicts.

### The five stages

**`/repair-setup <name>`** — creates the workspace and writes two files.
`env_config.py` says how to build the environment and describes, in
words, what each observation number and each action means.
`designed_reward_fn.py` is the reward itself: a function of `(obs, act,
next_obs)` returning a number. To get it, the step *always asks you
first* — either you fill in a three-box template (what to achieve, what
to avoid, how to trade them off; the wording is
`prompts/reward_design_template.md`) and Claude implements it, showing
you the code and looping until you approve, or you write it yourself
into a stub. It never silently reuses a reward that happened to be lying
around. It also imports the environment once to check it works, and
launches the web page if it isn't already up.

**`/repair-init`** — runs once, and instruments the reward. It reads the
reward's source line by line and writes two files.
`gather_evidence/trajectory_features.py` defines `Trajectory2Features`,
which turns a run into 6–12 characteristics, chosen so every term in
your reward becomes visible *and* — the important part — so each
rewarded **proxy** is measured separately from the **real outcome** it
stands in for. (If your reward pays out for "gripper near the handle",
the characteristics track both that and whether the drawer actually
opened. That separation is what allows a damning pair to exist at all.)
`candidate_performance_metrics/performance_metrics.py` defines 3–6
whole-run scores, each from a different perspective, at least one of
which is capable of disagreeing with your reward. The two frozen prompts
behind these are `prompts/trajectory_features_prompt.md` and
`prompts/performance_metrics_prompt.md`.

**`/repair-round`** — collects the behavior. Before writing anything it
produces (or refreshes) an internal **reward decomposition**: a
term-by-term reading of the reward plus a numbered list of *exploit
hypotheses* — concrete ways this particular reward could be gamed (H1,
H2, …, from `prompts/reward_decomposition_prompt.md`). It's stored at
`.repair_history/_reward_analysis.md`, stamped with a hash of the reward
source so it regenerates itself whenever you change the reward, and it
is never shown to you. Then it authors
`gather_evidence/data_collection_policy_round_k.py`: 3–9 policies in the
first round, 2–5 in later ones, always including a policy that
**maximizes the current reward** (so the reward's own favorite behavior
is on the table) plus at least one probe per untested hypothesis, each
designed to pull a rewarded proxy apart from the real outcome. These are
hand-written control policies, not trained ones. They're rolled out
(`runtime/run_round.py`, or `run_round_parallel.py` under SLURM for slow
environments — see `slurm_scripts/`) at **up to 100 runs per round**,
and every run is converted into characteristics. A rubric then asks
whether the evidence is good enough: is every reward term and every
hypothesis actually exercised? are the characteristics meaningfully
spread out? are there at least five disagreeing pairs to choose from? is
there any flaw-revealing behavior still untried? If something's missing
it authors another round — **up to 3 per invocation** — and otherwise it
rolls straight on into the selection step below. Calling
`/repair-round` again later just appends more rounds.

**`/repair-preferences`** — picks what you see. This is the heart of the
method, and it is **not an algorithm**. `select_pairs.py --mode dump`
lays out every run collected so far as one row of a table — its
characteristics, its total under your reward, and its total under each
alternative metric — and Claude reads that whole table and *chooses*
which comparisons will teach you the most, favoring diversity (different
flaws, different behaviors) over five variations of the loudest single
disagreement. Its picks go to `_selection.json`: up to **5 pairs** and
up to **2 lotteries**. `select_pairs.py --mode render` turns them into
cards. For a pair, it orients A = the side your reward preferred, works
out which perspectives preferred B instead, and ranks the
characteristics most→least informative about *why* the reward's
preference might be wrong. For a lottery, it compares the reward's value
of the certain run C against `p·A + (1−p)·B`, and picks `p` near the
indifference point `p* = (R_C − R_B) / (R_A − R_B)` — right where the
reward barely prefers one side, which is exactly where a human is most
likely to disagree with it. Everything shown is snapshotted to
`.repair_history/pref_round_NNN.json` so it can be re-checked later. If
the workspace has a `viz.py`, each run in each comparison is also
rendered — an mp4 of the real frames when the environment has a
renderer, otherwise a plot — and that's what sits at the top of each
card.

**`/repair-verify`** — after a reward edit, reloads every comparison you
were ever shown and recomputes it under the *new* reward
(`find_informative_pairs.py --check-snapshots`). For each one it reports
whether the reward's preference flipped: whether your edit fixed that
disagreement, left it standing, or broke something that used to be fine.
Any comparison that still disagrees is re-posted in full. It collects no
new behavior.

### Two design choices worth knowing

**Why alternative metrics, and why hide them?** They're a search
heuristic. Disagreement between your reward and *some* plausible way of
valuing a run is a cheap automatic signal that a comparison deserves
your attention — far cheaper than asking you to rank hundreds of runs by
hand. But they're only guesses about what you might value, and showing
them would anchor you on those guesses. So they choose what to show and
then get out of the way: the card shows you the two runs and how they
differ, and nothing else. Whether the reward is wrong is always your
call, never the system's — which is also why nothing in the interface
ever calls a run "correct" or "incorrect".

**Legacy selector.** `find_informative_pairs.py` is the original
algorithmic picker (a rank-product disagreement test plus a
Mahalanobis-greedy diversity pass). Claude-driven selection replaced it,
but it's kept for reference and is still what re-scores snapshots for
`/repair-verify`.

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

Collected runs, features, videos and `viz/` are **not** committed (see
`.gitignore`) — re-running a round regenerates them. The authored Python
(config, reward, characteristics, metrics, policies) is.

## Running the Python directly

The slash commands do all of this for you. To drive it by hand, from the
repo root with the right conda environment active:

```bash
# collect + featurize one round of runs
python -m runtime.run_round --workspace workspaces/<name> --round K

# lay out every run as a table, then render a chosen set of comparisons
python select_pairs.py --workspace workspaces/<name> --mode dump
python select_pairs.py --workspace workspaces/<name> --mode render \
       --selection workspaces/<name>/.repair_history/_selection.json

# re-score the comparisons you were shown, under the current reward
python find_informative_pairs.py --workspace workspaces/<name> --check-snapshots

# the web page, by hand
python -m web_ui.cli launch      # http://127.0.0.1:8000 (safe to call twice)
python -m web_ui.cli clear       # start a fresh feed
python -m web_ui.cli stop
```

`--workspace` can be omitted when `.current_workspace` is set.

## Adding a new environment

Run `/repair-setup <name>` and answer the questions — that's the
supported path, and it writes the files for you. If you'd rather
hand-write them, the exact contracts each file must satisfy (`make_env`,
`OBS_DESCRIPTION`, `MAX_EPISODE_STEPS`, the `Trajectory2Features`
methods, the `performance_metric_*` signature, the on-disk formats for
runs and characteristics) are spelled out in **`CLAUDE.md`** under
*Invariants*. Anything obeying those contracts works with the rest of
the system unchanged. See
[Getting started](#getting-started-describe-your-environment) for the
`env_config.py` contract and a prompt that generates it for you.

---

## Paper

This tool accompanies our paper on manual reward specification for RL.

<!-- TODO: add citation, BibTeX, and arXiv/project link on release. -->

If you use it in your research, please cite the paper (citation and link
to be added on release).
