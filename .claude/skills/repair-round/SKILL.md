---
name: repair-round
description: Steps 3-5 of the Automated Reward Repairer. Authors data-collection policies, rolls them out, evaluates whether the resulting evidence is sufficient, and either adds another round automatically or hands off to /repair-preferences to show the user the disagreements. Invoke whenever the user asks to collect more evidence.
---

# /repair-round — collect evidence of reward misspecification

This skill is the autonomous evidence-collection step. A single invocation
may run **one or more** rounds of data collection — Claude decides whether
the evidence already gathered is sufficient, or whether another round
would help. When the evidence is sufficient, it automatically continues to
`/repair-preferences` so the user sees results without manually chaining
skills.

## Tone for messages to the user

Treat the user as someone who will never read the codebase. Don't post
file names, directories, internal scripts, or rollout counts on the
website. Talk about what the system is **doing** in plain language
("collecting trajectories that probe a particular reward term",
"checking whether the evidence is rich enough"). Save filenames for
tool calls.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect.
Whether something is actually right is the user's call. (See `CLAUDE.md`
for the full tone rules.) The alternative-perspective metrics are
under-the-hood selection signals only — never mention a perspective,
metric, or "stakeholder" in anything posted to the user.

### Short status updates

Post **one-sentence** status messages at key transitions
("Designing the next round of probes.", "Rolling out trajectories.",
"Round complete — analysing what came out.") via `python -m
web_ui.cli post --type status`. The website automatically shows a
"Claude is thinking" spinner when no question is pending, so you
don't need to repeat that idea — just post a single-sentence label
when something interesting changes.

## Web UI is the only user-facing channel

The web UI was launched by `/repair-setup`. Every user-visible message
this skill emits goes through `python -m web_ui.cli post`; every
question goes through `python -m web_ui.cli ask`. **Do not** call the
`AskUserQuestion` tool, and do not narrate to the terminal beyond brief
operational status the agent needs for its own reasoning. See
`CLAUDE.md` → "Web UI — the user-facing channel" for the reference.

If the server isn't running, bring it up first:
`python -m web_ui.cli launch`.

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-round hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, post a message asking the user to run `/repair-setup`
first and stop. Update the UI header:

```bash
python -m web_ui.cli set workspace=<WORKSPACE> status=running
```

`WORKSPACE_DIR = workspaces/<WORKSPACE>`.

## Preconditions

- `<WORKSPACE_DIR>/env_config.py`, `<WORKSPACE_DIR>/designed_reward_fn.py`,
  `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py`, and
  `<WORKSPACE_DIR>/candidate_performance_metrics/performance_metrics.py`
  all exist. If any is missing, post a message asking the user to run
  `/repair-setup` or `/repair-init` first (in user-friendly terms — don't
  list paths) and stop.

## The autonomous loop

Run up to **3 rounds per invocation**. After each round, evaluate the
evidence collected so far and decide:

- **Run another round** if the evidence has clear gaps (see "Sufficiency
  rubric" below). Author a new round that targets the gap.
- **Stop and proceed to `/repair-preferences`** when the evidence is
  rich enough, or when the 3-round cap is reached.

Communicate the decision to the user in one short sentence each time, on
the website:

```bash
python -m web_ui.cli post --type status --content "Collecting one more round — one of the reward terms hasn't been stress-tested yet."
```

Don't dump per-round statistics on the user.

## Per-round procedure

### Step 1 — figure out the next round number

```python
import glob, re, os
existing = glob.glob(f"{WORKSPACE_DIR}/gather_evidence/data_collection_policy_round_*.py")
nums = [int(re.search(r"round_(\d+)\.py$", p).group(1)) for p in existing]
k = (max(nums) + 1) if nums else 1
```

### Step 2 — author the round's policies

Write `<WORKSPACE_DIR>/gather_evidence/data_collection_policy_round_{k}.py`.

**First, build or refresh the reward decomposition.** Following
`prompts/reward_decomposition_prompt.md`, write/update
`<WORKSPACE_DIR>/.repair_history/_reward_analysis.md` (the per-term
breakdown and exploit hypotheses `H1, H2, …`); reuse it if its stamped
hash already matches the current reward, otherwise regenerate it (see
`CLAUDE.md` → "Reward introspection" for the reuse rule). The round you
author next exists to make those hypotheses *occur* in real
trajectories.

Read `prompts/data_collection_policy_prompt.md` for the verbatim task
text. Context to use when writing:
- `env_config.OBS_DESCRIPTION` and `ACTION_DESCRIPTION`
- the source of `designed_reward_fn.py`
- **all prior rounds**:
  `<WORKSPACE_DIR>/gather_evidence/data_collection_policy_round_{1..k-1}.py`,
  so this round is genuinely different from what's already been tried
- `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py` — the features
  illuminate what's worth varying across rounds
- the most recent feature-summary stats from prior rounds (see
  "Sufficiency rubric"); use these to identify which reward terms haven't
  been exercised yet

Each class must conform to the invariants in `CLAUDE.md`:

```python
class DataCollectionPolicy<Name>:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, obs):
        ...  # returns an action valid for env.action_space
```

Add a module-level `POLICIES = [Cls1, Cls2, ...]`. Each class's docstring
states the *hypothesis* it tests.

**Required policy in every round:** one of the classes in `POLICIES`
must be a policy that maximizes the expected sum of rewards with
respect to the *current* `designed_reward_fn`.

**Required exploit probes:** in addition to the maximizer, every round
must include at least one policy that targets a still-untested exploit
hypothesis `H1, H2, …` from the reward decomposition. Each such policy's docstring names
the reward term and the hypothesis id (`H#`) it probes. Across the
rounds of a workspace, every hypothesis must eventually get a probe;
bias each new round toward the hypotheses not yet exercised.

How many policies per round? 3-9 for round 1 (broad coverage, including
the required maximizer); 2-5 for later rounds (focused on a specific
gap — but still re-include the maximizer under the current reward,
since the reward may have changed since the prior round).

### Step 3 — roll out + featurize

```bash
ENV=$(cat workspaces/<WORKSPACE>/.conda_env 2>/dev/null); if [ -n "$ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV"; fi; \
python -m runtime.run_round --workspace workspaces/<WORKSPACE> --round K
```

This writes trajectories and features under
`<WORKSPACE_DIR>/gather_evidence/round_{K}_trajectories/` and
`.../round_{K}_trajectory_features/`. You don't need to surface any of
this to the user — a brief status update is enough:

```bash
python -m web_ui.cli post --type status --content "Round complete — analysing what the policies revealed."
```

### Step 4 — evaluate sufficiency

Read every `round_*_trajectory_features/*.json` written so far (across all
rounds in this workspace) and aggregate per-policy feature means. Use the
rubric below.

## Sufficiency rubric

The evidence is **sufficient** when *all* of the following hold:

1. **Every reward term AND every exploit hypothesis has been exercised
   by at least one trajectory.** Use the reward decomposition
   (`_reward_analysis.md`) to enumerate both. For each *term*, look in
   the feature aggregates for a trajectory where that term clearly fired:
   - a binary failure-state term needs a trajectory whose corresponding
     feature equals 1.0
   - a per-step penalty needs a trajectory whose fraction-of-steps-firing
     feature is meaningfully above zero
   - a sparse terminal-bonus term needs at least one trajectory that
     reaches the bonus condition
   For each *exploit hypothesis* `H#`, the evidence must contain at least
   one trajectory that **instantiates** it — i.e. the proxy feature fired
   while the true-outcome feature did not (the proxy-vs-outcome decoupling
   is actually present, not just hypothesized). If a term is silent, or a
   hypothesis is never instantiated, across **all** trajectories, the
   evidence is incomplete — design a round targeting it.

2. **The trajectories are meaningfully diverse** on the key features.
   "Meaningful spread" ≈ (max − min) / typical-scale > 0.3 on at least
   half the features. If every policy collapses to the same behavior, add
   policies that diverge.

3. **At least 5 distinct disagreeing pairs can be found** in the dumped
   trajectory table (`select_pairs.py --mode dump`) — pairs where the
   designed reward and another perspective's metric rank the two
   trajectories oppositely. Judge this directly from the table; do not
   run the legacy `find_informative_pairs.py --top-overall` probe. If
   fewer than 5 exist, the rollouts didn't generate enough diverging
   behavior.

4. **You covered every behavior you suspected would uncover a flaw in
   `designed_reward_fn.py`.** Beyond the enumerated exploit hypotheses,
   if — after you read the reward function— you can still name a behavior you
   suspect would expose a flaw (a way to Goodhart, a corner of the
   state space the reward may mishandle, a specification gap) and no
   collected trajectory realizes it, the evidence is incomplete. Design a
   round that produces that behavior. The evidence is sufficient on this
   axis only when you can no longer name an untried behavior you'd expect
   to reveal a flaw.

If any of (1)/(2)/(3)/(4) fails AND we haven't hit the 3-round cap, run
another round targeting the failing axis.

## Targeting the gap

When designing round k > 1, look at *why* the rubric failed and bias the
new policies accordingly:

- Rubric (1) failed on a reward term → write a policy whose explicit goal
  is to trigger that term. E.g., a policy that applies maximum-magnitude
  actions stresses a control-effort penalty; a policy that deliberately
  fails the task (falls, stalls, leaves the goal region) stresses a
  failure-state penalty; a policy that enters a forbidden region of the
  state space stresses a boundary penalty.
- Rubric (2) failed (everything collapses) → write policies along an
  ignored axis: extreme-then-zero oscillators, periodic perturbations,
  biased exploration along a specific observation dimension, or a policy
  that holds one action constant.
- Rubric (3) failed (too few distinct disagreements) → write a policy
  that scores highly on one performance metric while being penalized by
  the designed reward (or vice versa). Pick the metric most likely to
  diverge — usually the one most orthogonal to the user's stated goals.

Round 1 has no prior evidence; design a broad-coverage round of 3-5
policies probing diverse hypotheses.

## When the loop ends — produce the /repair-preferences deliverable on the web UI

When the rubric is satisfied (or the 3-round cap is reached), continue
immediately into the `/repair-preferences` deliverable — do **not** stop
and ask the user. Execute `repair-preferences/SKILL.md` inline; the
deliverable is now posted to the web UI rather than printed in chat.
Follow `repair-preferences/SKILL.md` precisely, but in summary:

1. **Select the comparisons.** Dump the trajectory table with
   `select_pairs.py --mode dump`, read the resulting table, choose **up
   to 5 pairs plus up to 2 lottery comparisons** (`{c, a, b, p}`; you
   must justify `p` in the selection JSON), write the selection JSON,
   and render them with `select_pairs.py --mode render`. Favor
   diversity — each comparison shown should illuminate a distinct
   trade-off in the reward. For each pair, add a `"characteristics"`
   list — the characteristic `short_name`s ranked most→least informative
   about why the designed reward's preference may be wrong for that pair;
   the display shows them in that order.

   **For lotteries: convention is `p` = probability of the good branch
   (A), `(1−p)` = probability of the bad branch (B). Pick `p` so the
   choice between C and the lottery is genuinely difficult — *don't*
   default to a fixed number.** Compute the designed-reward
   indifference point `p* = (return(C) − return(B)) / (return(A) −
   return(B))`, then move slightly off it (usually onto the
   C-preferring side) so the reward gives a small-but-real preference
   that a human might intuitively reverse. The exact `(1−p)` depends
   on the magnitudes in the chosen trajectories — sometimes very
   small, sometimes not. Justify the chosen `p` in the selection
   file. If posting two lotteries, prefer two *different shapes*
   (different C/A/B or a different trade-off) over two values of `p`
   on the same shape.

2. **Post every rendered comparison from the emitted payload.** The
   render writes a ready-to-post bundle to
   `workspaces/<WORKSPACE>/.repair_history/_ui_payload/`. Read its
   `manifest.json` and post one comparison per entry in `cards`, in
   order (pairs first, then lotteries). Post **only** the comparisons —
   no feature key, overall summary, pattern paragraph, or misalignment
   orientation. Each card carries this round's `content_file`, `viz_file`
   (the exact `--viz` argument — `src` URLs, the `preferred` badge for
   pairs, and `role`/`probability` for lotteries already filled in),
   `label`, and `layout`:
   ```bash
   # pair card: post the render's sentences verbatim.
   python -m web_ui.cli post --type comparison \
       --label "<card.label>" \
       --content-file "<card.content_file>" \
       --viz "$(cat <card.viz_file>)" \
       --why "<1 sentence naming the trade-off between the two trajectories — what the preferred one gains and gives up. Don't mention perspectives or metrics.>"

   # lottery card: you author --content (the content_file is empty); add --layout lottery.
   python -m web_ui.cli post --type comparison \
       --label "<card.label>" --layout lottery \
       --content "<2–3 sentences on how Trajectory A (good branch), Trajectory B (bad branch), and certain Trajectory C differ. Be QUANTITATIVE: state the actual characteristic values (with units) from the table, not qualitative interpretations like 'safe', 'smooth', or 'aggressive'. Don't mention perspectives or metrics.>" \
       --viz "$(cat <card.viz_file>)" \
       --why "<1 sentence naming the risk-attitude trade-off between C and the lottery. Don't mention perspectives or metrics.>"
   ```
   Omit `--viz` only when a card's `viz_file` is null (no `viz.py` in the
   workspace). Don't hand-build viz JSON or re-derive the `src` paths —
   they're already in `viz_file` and are correct for this round.

3. **Always close with the three-option next-step question** posted to
   the web UI:
   ```bash
   python -m web_ui.cli ask --prompt "What would you like to do next?" \
       --options '[
         {"label":"Edit the reward function","description":"Modify the reward to reflect what you actually want, based on the discordances above."},
         {"label":"Propose more discordances","description":"Propose more pairs or lotteries from the trajectories already collected."},
         {"label":"End session","description":"Keep the current reward function and close the tool."}
       ]'
   ```
   Act on the answer per `repair-preferences/SKILL.md` Step 4.

The render script anonymizes trajectory identity — each is shown as
"Trajectory A" / "Trajectory B" (or "C" / "A" / "B" inside a lottery)
with no round, step count, or policy class. Do not re-introduce any of
that information in the website posts.

Skipping the posted comparisons, the per-comparison `--why`, or the
closing `ask` is not optional — they are the core deliverable of this
skill. The user should never have to ask for any of them separately.

For the transition message between rounds and the preferences view, use
plain language framed as outcomes:

```bash
python -m web_ui.cli post --type status --content "Collected enough evidence. Surfacing the most telling disagreements now."
```

Do not say "running /repair-preferences" or list internal commands.

## After the closing question

The user's answer drives the next action per `repair-preferences/SKILL.md`
Step 4 (edit the reward / propose more discordances / end session). Don't append a separate summary of which files exist or how
to re-run individual skills.
