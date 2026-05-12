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

Treat the user as someone who will never read the codebase. Don't list
file names, directories, internal scripts, or rollout counts. Talk about
what the system is **doing** in plain language ("collecting trajectories
that probe a particular reward term", "checking whether the evidence is
rich enough"). Save filenames for tool calls.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with the stakeholder's view." Whether
something is actually right is the user's call. (See `CLAUDE.md` for
the full tone rules.)

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-round hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, tell the user to run `/repair-setup` first.

`WORKSPACE_DIR = workspaces/<WORKSPACE>`.

## Preconditions

- `<WORKSPACE_DIR>/env_config.py`, `<WORKSPACE_DIR>/designed_reward_fn.py`,
  `<WORKSPACE_DIR>/gather_evidence/trajectory_features.py`, and
  `<WORKSPACE_DIR>/candidate_performance_metrics/performance_metrics.py`
  all exist. If any is missing, tell the user to run `/repair-setup` or
  `/repair-init` first (in user-friendly terms — don't list paths).

## The autonomous loop

Run up to **3 rounds per invocation**. After each round, Claude
evaluates the evidence collected so far and decides:

- **Run another round** if the evidence has clear gaps (see "Sufficiency
  rubric" below). Author a new round that targets the gap.
- **Stop and proceed to `/repair-preferences`** when the evidence is
  rich enough, or when the 3-round cap is reached.

Communicate the decision in one short sentence each time
("Collecting one more round — the proximity term hasn't been stress-tested
yet."). Don't dump per-round statistics on the user.

## Per-round procedure

### Step 1 — figure out the next round number

```
import glob, re, os
existing = glob.glob(f"{WORKSPACE_DIR}/gather_evidence/data_collection_policy_round_*.py")
nums = [int(re.search(r"round_(\d+)\.py$", p).group(1)) for p in existing]
k = (max(nums) + 1) if nums else 1
```

### Step 2 — author the round's policies

Write `<WORKSPACE_DIR>/gather_evidence/data_collection_policy_round_{k}.py`.

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

How many policies per round? 3-5 for round 1 (broad coverage, including
the required maximizer); 1-3 for later rounds (focused on a specific
gap — but still re-include the maximizer under the current reward,
since the reward may have changed since the prior round).

### Step 3 — roll out + featurize

```
python -m runtime.run_round --workspace workspaces/<WORKSPACE> --round K
```

This writes trajectories and features under
`<WORKSPACE_DIR>/gather_evidence/round_{K}_trajectories/` and
`.../round_{K}_trajectory_features/`. You don't need to surface any of
this to the user — just confirm the round completed.

### Step 4 — evaluate sufficiency

Read every `round_*_trajectory_features/*.json` written so far (across all
rounds in this workspace) and aggregate per-policy feature means. Use the
rubric below.

## Sufficiency rubric

The evidence is **sufficient** when *all* of the following hold:

1. **Every reward term has been exercised by at least one trajectory.**
   Read `designed_reward_fn.py` to enumerate the terms. For each, look in the feature
   aggregates for a trajectory where that term clearly fired. Examples:
   - a binary failure-state term needs a trajectory whose corresponding
     feature equals 1.0
   - a per-step penalty needs a trajectory whose fraction-of-steps-firing
     feature is meaningfully above zero
   - a sparse terminal-bonus term needs at least one trajectory that
     reaches the bonus condition
   If a term is silent across **all** trajectories, the evidence is
   incomplete — design a round targeting it.

2. **There is meaningful spread across rollouts** on the key features.
   "Meaningful spread" ≈ (max − min) / typical-scale > 0.3 on at least
   half the features. If every policy collapses to the same behavior, add
   policies that diverge.

3. **`find_informative_pairs.py --top-overall 5 --margin 0.05`** returns
   at least 3 distinct disagreeing pairs at margin ≥ 0.05 (you can
   probe this without showing the user — count `PAIR ` headers in
   stdout). If fewer than 3, the rollouts didn't generate enough
   diverging behavior.

If any of (1)/(2)/(3) fails AND we haven't hit the 3-round cap, run
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

## When the loop ends

When the rubric is satisfied (or the 3-round cap is reached), continue
immediately into `/repair-preferences` — do **not** stop and ask the
user. `/repair-preferences` is a separate skill but you can execute its
instructions inline. You must produce its **full deliverable** at the
end of this turn. Follow `repair-preferences/SKILL.md` precisely:

1. Pull up to 15 disagreements with
   `find_informative_pairs.py --workspace workspaces/<WORKSPACE> --top-overall 15 --format preference-labels`.
2. Reprint **every** returned pair **verbatim** inside fenced code
   blocks (feature key first, then each pair in the order the script
   returned them), with **two summary lines directly under each pair**,
   in this order:
   - `**Why they disagree:**` — 1–2 sentences naming the trade-off
     the reward made, framed as trade-offs, not as verdicts.
   - `**What you might change in the reward function to flip this preference:**`
     — 1–2 sentences proposing a concrete, specific edit (a weight to
     bump, a term to add, a gating condition, etc., referencing the
     actual variable/constant names from `designed_reward_fn.py`) that
     would make the designed reward prefer the stakeholder's side on
     that pair. Phrase as an option, not a directive.
3. A short overall-pattern paragraph + 1–2 possible reward tweaks
   ("if you didn't intend X over Y, one option is …"), never framed
   as "the reward is wrong."
4. **Always** close with an `AskUserQuestion` offering three options:
   (a) run another round of evidence, (b) modify the reward yourself,
   or (c) ask Claude to modify the reward to satisfy the surfaced
   preferences.

The script anonymizes trajectory identity — each is shown as just
"Trajectory A" / "Trajectory B" with no round, step count, or policy
class. Do not re-introduce any of that information in your text.

Skipping the verbatim pairs, either per-pair summary line, or the
closing `AskUserQuestion` is not optional — they are the core
deliverable of this skill. The user should never have to ask for any
of them separately.

Frame the transition to the user as outcomes:

> "Collected enough evidence. Surfacing the most telling disagreements
> between your reward function and other stakeholder perspectives now."

Do not say "running /repair-preferences" or list internal commands.

## After `/repair-preferences` returns

The end-of-turn message is whatever `/repair-preferences` produced
(verbatim pairs + per-pair interpretations + overall pattern + the
closing `AskUserQuestion`). Don't append a summary of which files
exist or how to re-run individual skills. Act on the user's answer per
`repair-preferences/SKILL.md` Step 5.
