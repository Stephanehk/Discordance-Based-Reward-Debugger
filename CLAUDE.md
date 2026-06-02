# Automated Reward Repairer

## What this project is

The user has a gym/gymnasium environment and a manually written (or
LLM-elicited) reward function `designed_reward_fn(obs, act, next_obs) -> float`.
This system surfaces *evidence of reward misspecification* by collecting
diverse trajectories, scoring them with both the designed reward and a set
of alternative-perspective performance metrics, and showing the user
**trajectory pairs where the two disagree on which trajectory is better**.
The pairs are the end product: the user reads them and decides how to edit
the reward.

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
/repair-preferences   →  Claude reads every collected trajectory (as
                          features, the same encoding the human sees) plus
                          the designed reward and the alternative-
                          perspective metrics, and *itself selects* the
                          most-telling comparisons — up to **5 trajectory
                          pairs** plus up to **2 lottery comparisons**
                          (certain trajectory C vs a p-probability lottery
                          over A and B). This is the Claude-driven
                          selection step, not an algorithm. Pairs render
                          as preference labels (feature key, anonymized A
                          vs B, reward + metric scores side-by-side with
                          A≻B / B≻A labels, feature delta table). The
                          renderer adds a "Designed reward function
                          prefers this trajectory" badge below the
                          reward-preferred side. Lotteries render with A
                          and B side-by-side at the top (with their
                          probabilities) and the certain trajectory C
                          below; the question is shown with reward and
                          other-perspective preferences. Each comparison
                          also triggers a per-trajectory visualization
                          (`workspaces/<name>/viz/round{i}/{pair,lottery}
                          {j}/...`) via the workspace's `viz.py`; the
                          rendered output prints the saved paths. Each
                          comparison comes with a "why they disagree"
                          summary and an overall pattern interpretation.
                          Both kinds are snapshotted so /repair-verify
                          can later re-check them. (The legacy
                          algorithmic selector still lives in
                          find_informative_pairs.py for reference and
                          for /repair-round's sufficiency probe.)
```

`/repair-init` runs once after `/repair-setup`. `/repair-round` is
append-only across invocations — each call adds the next k rounds and
ends in `/repair-preferences`. `/repair-preferences` can also be re-run
freely on its own; it just reloads everything from disk and re-scores
under the current reward.

## Web UI — the user-facing channel

During every `/repair-*` skill, the user interacts with the system through
a local web UI at **http://127.0.0.1:8000**, not the terminal.
`/repair-setup` launches the server (idempotent) and clears any prior
session feed. Every subsequent skill posts its narrative, code,
comparison cards, and visualizations to the UI and asks every question
there. Until then the user has nothing to look at.

**Do not print user-facing narrative to the terminal during a `/repair-*`
skill.** Use `python -m web_ui.cli post` for every message the user
should see, and `python -m web_ui.cli ask` for every question the user
should answer — replacing the `AskUserQuestion` tool. Keep your own
chat / terminal output to brief operational status only ("collecting
trajectories…", "round 1 complete", "awaiting answer"); the user reads
the website. The single exception is the one-time URL hint that
`/repair-setup` prints in the terminal so the user knows where to open
their browser.

### Message types (see `web_ui/cli/main.py` for full arg list)

| Use for | `post --type` | Notes |
|---|---|---|
| Plain narrative paragraphs | `markdown` | headers, bold, lists, tables, inline code |
| `designed_reward_fn.py` source | `code --language python --caption "designed_reward_fn.py" --content-file <path>` | code is shown in a dark monospace block |
| A trajectory pair block | `comparison --label "PAIR 1/5" --content-file render.txt --viz '[{"src":"/workspace/<env>/viz/round{i}/pair{j}/A.mp4","caption":"Trajectory A","preferred":true}, {"src":"/workspace/<env>/viz/round{i}/pair{j}/B.mp4","caption":"Trajectory B"}]' --why "…"` | the `Why they disagree:` line is `--why`; `"preferred":true` adds a "Designed reward function prefers this trajectory" badge under that video |
| A lottery comparison block | `comparison --label "LOTTERY 1/2" --layout lottery --content-file render.txt --viz '[{…"role":"A","probability":0.5}, {…"role":"B","probability":0.5}, {…"role":"C"}]' --why "…"` | `--layout lottery` puts A and B side-by-side on the top row (with their probability badges) and the role-C item below |
| The feature key (right-hand sidebar) | `feature_key --features-file <json>` | each feature: `{"name":"…","description":"…"}`; sidebar only appears on the pairs/lotteries page |
| Light status update | `status` | e.g. "Collecting trajectories…" |
| Visual separator | `divider` | thin dashed line |

Code blocks with `--language python` are syntax-highlighted in the
browser (keywords, strings, comments, numbers, function/class names).

Mp4/png paths used in `--viz` should start with `/workspace/<env>/…`;
that URL is mapped by the server to `workspaces/<env>/…` on disk, so
the captured-frame videos and rendered plots load directly.

### "Claude is thinking" spinner + short status updates

While a `/repair-*` skill is mid-step (status=`running`) and no
question is pending, the website automatically shows a small "Claude
is thinking…" spinner at the end of the feed. Skills don't have to
toggle it explicitly — it follows the workspace `status` chip, which
is `running` between actions and `awaiting` while a question or CTA
is open.

What skills *should* do: post **one-sentence** `--type status`
messages at key transitions (e.g. "Authoring the next round.",
"Rolling out trajectories.", "Selecting the most informative
comparisons.", "Rendering the visualizations."). Keep them short —
the spinner already conveys "still working"; the status line just
labels *what* the work is.

### Asking the user

```bash
ANSWER=$(python -m web_ui.cli ask \
    --prompt "How would you like to specify the reward function?" \
    --options '[{"label":"Describe it — I'\''ll implement","description":"…"},
                {"label":"I'\''ll write it myself","description":"…"}]')
VALUE=$(printf %s "$ANSWER" | python -c "import json,sys;print(json.load(sys.stdin)['value'])")
```

- Multi-select: add `--multi`; the parsed `value` is a JSON array.
- Single-line: `--input text --placeholder "…"`.
- Long-text (free-form markdown): `--input longtext --prefill-file <md>`.
- **Form** (preferred for structured templates like the reward-design
  template): `--fields '[{"name":…,"label":…,"description":…,"placeholder":…}, …]'`.
  Renders one labelled textarea per field; the returned `value` is a JSON
  object `{name: text, …}`.

### Single "Continue" button (CTA)

When a skill ends and the user should advance to the next step, post a
**call-to-action button** instead of telling them to type a slash-
command in the terminal:

```bash
ANSWER=$(python -m web_ui.cli cta \
    --prompt "Setup is complete." \
    --label  "Identify possible flaws" \
    --value  "repair-init")
NEXT=$(printf %s "$ANSWER" | python -c "import json,sys;print(json.load(sys.stdin)['value'])")
```

Then invoke the next skill via the `Skill` tool with `skill="$NEXT"`.
The cta blocks until the user clicks; this is how skills chain through
the website.

### Hidden answered questions

Once the user submits an answer (or clicks a CTA), the corresponding
card is removed from the feed. The agent still receives the value
through the `ask` / `cta` stdout — only the visual record disappears,
keeping the feed focused on what's currently pending.

`ask` long-polls until the user submits, then prints `{"value": "…",
"notes": null}` on stdout. The agent reads `value` and acts on it. Do
**not** call the `AskUserQuestion` tool inside a `/repair-*` skill.

### Launch / lifecycle

```bash
python -m web_ui.cli launch                              # idempotent
python -m web_ui.cli clear                               # reset the feed (fresh session)
python -m web_ui.cli set workspace=<name> status=running # header chip
python -m web_ui.cli stop                                # at session end (optional)
```

`launch` is safe to call repeatedly — it no-ops if the server is
already running. `clear` is the right thing to call at the top of
`/repair-setup` so each new session starts with an empty feed.

## User-facing tone

Skills follow these tone rules in every message they send the user:

- **Never list internal filenames, directories, or boilerplate scripts.**
  Talk in terms of outcomes ("setting up the environment", "looking for
  evidence of misalignment", "surfacing the disagreements"), not
  implementation details.
- **Never refer to any behavior, reward, or trajectory as "correct" or
  "incorrect."** At most, say something is *possibly* correct/incorrect
  or that it *appears* aligned/misaligned with a particular perspective.
  Whether a behavior is actually correct is always the user's call —
  the system surfaces evidence, not verdicts.
- **Never name a candidate performance metric by its internal id**
  (e.g. `performance_metric_safety`). Either say "the candidate
  performance metric" or describe the perspective from the docstring
  ("the safety reviewer's view", "the operations perspective").
- **Do not use the word "stakeholder" in anything posted to the user.**
  Use "other perspectives" or describe the specific perspective.

## Skill chaining — always tell the user what to run next

Every skill must end by telling the user, **on the web UI**, what they
can run next. Never finish a skill without either (a) posting a
markdown message that names the next skill explicitly, or (b) calling
`python -m web_ui.cli ask` with the relevant options. **Do not** use
the `AskUserQuestion` tool inside a `/repair-*` skill — the website is
the channel. The expected chain is:

- `/repair-setup` → post that the user should run `/repair-init`.
- `/repair-init` → post that the user should run `/repair-round`.
- `/repair-round` → after surfacing disagreements on the website, **ask**
  the user via `web_ui.cli ask` whether they want to (1) **Edit the
  reward function**, (2) **Add another trajectory feature** to the set
  that encodes each trajectory, (3) **View more comparisons (pairs or
  lotteries)** from the evidence already collected, or (4) **Finish and
  submit (not reversible)**. (The "Edit the reward function" option
  then asks whether the user or Claude makes the edit; the Claude path
  posts the reward-edit tip box to the web UI.)
- `/repair-preferences` → same closing question as `/repair-round`,
  asked via `web_ui.cli ask`.
- `/repair-verify` → ask via `web_ui.cli ask` whether to look for new
  disagreements, collect more evidence first, or **Finish and submit
  (not reversible)**.

### Finish and submit (not reversible)

When the user picks **Finish and submit**, treat the current
`designed_reward_fn.py` as the final answer: write
`workspaces/<name>/.repair_history/SUBMISSION.md` containing a UTC
timestamp and the full current reward source, post a confirmation to
the web UI (`python -m web_ui.cli post --type markdown --content "The
reward function is submitted. Session complete."` and
`python -m web_ui.cli set status=done`), and stop — do not offer
further steps, run more rounds, or edit the reward afterward. Because
it is irreversible, only do this when the user explicitly selects it.

### Reward-edit tips (post to the web UI when asked to edit the reward)

Whenever the user asks Claude to modify the reward (the Claude branch
of the "Edit the reward function" option), post this tip box on the
web UI before requesting or applying the edit:

```bash
python -m web_ui.cli post --type markdown --content "💡 **Tips for guiding a reward edit**
- Be concrete: name the term to change, and — if you can — the preference label you want for each shown pair (e.g. \"for pair 2, A ≻ B should become B ≻ A\").
- You can pin an edit to an outcome: e.g. \"increase the above-target speed penalty until the preference for pair 2 flips,\" and Claude will solve for the smallest change that produces it."
```

These rules apply to every skill in `.claude/skills/`. If a skill's own
`SKILL.md` doesn't repeat them, they still apply.

## File layout

```
automated_reward_repairer/
├── CLAUDE.md
├── select_pairs.py                     # Claude-driven pair selection (dump + render)
├── find_informative_pairs.py           # legacy algorithmic selector; kept for reference + /repair-verify snapshot check
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
        ├── viz.py                      # optional: trajectory visualization hook
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
        ├── candidate_performance_metrics/
        │   └── performance_metrics.py             # authored in /repair-init
        │
        └── viz/                                    # written by /repair-preferences
            └── round{i}/{pair,lottery}{j}/        # per-comparison visualizations
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
  - Docstring is the **perspective description** — shown to the user
    when this metric flags a disagreement. Keep it one line.

**`viz.py`** (optional — `/repair-preferences` skips visualization if absent):
- `VIZ_EXT: str` — the file extension produced (e.g. `".png"`, `".mp4"`).
- `visualize_trajectory(trajectory, metadata, out_path, source_path=None,
  **_) -> None` — save a visualization for one trajectory to `out_path`
  (already includes `VIZ_EXT`). `trajectory` is the same `[(obs, act,
  next_obs), ...]` list that lives on disk; `metadata` is the dict stored
  alongside it (may contain `seed`, `policy_class`, etc.); `source_path`
  is the trajectory's `.pkl` path on disk — viz hooks that read sidecar
  files (e.g. captured-frame videos) use this to locate them. Take
  `**_` so future kwargs can be added without breakage. Implementations
  should be workspace-appropriate (e.g. a per-feature time-series plot
  for an observation-rich env, a captured-frame mp4 for an env with a
  renderer).

**Per-step frame capture (optional, opt-in):**
- A workspace can declare `CAPTURE_FRAMES = True` in `env_config.py` to
  ask the rollout runners to record one `env.render()` frame per step
  and write an mp4 sibling to the trajectory pickle:
  `gather_evidence/round_{k}_trajectory_videos/{same_basename}.mp4`.
  When set, `make_env` must accept a `render_mode` kwarg (the runners
  pass `render_mode="rgb_array"`). The video contains the actual rolled-
  out frames — no after-the-fact replay — so a workspace `viz.py` can
  surface it by simply locating the sidecar via
  `runtime.trajectory_io.video_path_for(traj.pkl)` and copying it to
  `out_path`.

**Trajectory on disk (workspace-relative):**
- File: `gather_evidence/round_{k}_trajectories/{policy_class}__traj_{i}.pkl`
- Payload: `{"trajectory": [(obs, act, next_obs), ...], "metadata": {...}}`.

**Features on disk (workspace-relative):**
- File: `gather_evidence/round_{k}_trajectory_features/{same_basename}.json`
- Payload: `{"features": [...], "names": [[short, desc], ...], "source": "..."}`.

**Rollout budget per round:**
- A single round writes **at most 100 trajectories total**. The
  `runtime.run_round` / `runtime.run_round_parallel` CLIs clamp
  `rollouts/policy` so `n_policies * rollouts_per_policy ≤ 100`. When
  authoring a round's policies, account for this cap (e.g. with 5
  policies each gets ≤ 20 rollouts).

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

All Python in this repo runs in the `llm` conda env — `gymnasium` and
any per-workspace env dependencies live there, not in the base
interpreter. Activate it before any boilerplate script or smoke test
(the `/repair-*` skills already prepend this to their commands):

```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm
```

```
# from the repo root, with `llm` active:
python -m runtime.run_round --workspace workspaces/<name> --round K

# Claude-driven pair selection (the /repair-preferences step):
python select_pairs.py --workspace workspaces/<name> --mode dump
python select_pairs.py --workspace workspaces/<name> --mode render \
       --selection workspaces/<name>/.repair_history/_selection.json

# legacy algorithmic selector (reference + /repair-verify snapshot check):
python find_informative_pairs.py --workspace workspaces/<name> --top-overall 5 --margin 0.05
```

If `.current_workspace` is set, `--workspace` may be omitted.
