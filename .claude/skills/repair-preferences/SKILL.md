---
name: repair-preferences
description: Show up to 5 trajectory pairs plus up to 2 lottery comparisons (certain trajectory C vs a p-probability lottery over A and B) as preference labels on the web UI. For each comparison: features, scores from the designed reward and the alternative-perspective metrics, the preference label each induces, a short explanation of why another perspective disagrees, and per-trajectory visualization paths. Ends with an overall interpretation of likely reward flaws.
---

# /repair-preferences — show preference-label disagreements on the web UI

## ⚠ CORE CONTRACT (do not skip)

**Every comparison the script returns — up to 5 PAIR blocks and up to 2
LOTTERY blocks — MUST be posted to the web UI via `python -m web_ui.cli
post --type comparison`, each with its own `--why` interpretation
line.** The chat / terminal output the agent emits is **not** the
deliverable; the website is. The render script's stdout is consumed by
the agent, parsed, and forwarded to the UI — not pasted into the chat.

Before any comparison is posted, the response on the web UI must
include — in this order:

1. **Overall-pattern paragraph + 1–2 possible reward tweaks**, posted
   via `post --type markdown`. Framed as "the reward *appears* to
   prioritize X over Y" — never as "the reward is wrong".
2. **Orientation line**, posted via `post --type status`, telling the
   user the comparisons below show where the designed reward may be
   misaligned (and that they'll be asked what to do next). Keep
   *may be misaligned* — not "wrong".
3. **Feature key**, posted via `post --type feature_key`. Renders in
   the right-hand sidebar.

Then the comparisons (pairs first, then lotteries), each with its
`--why` line.

Finally, **always** close with the four-option `ask` (edit / add a
feature / view more / finish and submit). The skill has not delivered
its core output until that question is on the website.

## Tone for messages to the user

The user will never read the codebase. Don't include file names,
internal scripts, or boilerplate paths in anything posted to the
website. **Never name a metric by its internal identifier**
(e.g. `performance_metric_<short_name>`). Refer to "the candidate
performance metric" or describe the perspective from the docstring
(e.g. "the safety reviewer's view", "the operations perspective").
Do not use the word "stakeholder" in anything posted to the user — say
"other perspectives" or describe the specific perspective.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with this perspective's view." Whether
something is actually right is the user's call — the system surfaces
evidence, not verdicts. The same applies to the overall interpretation:
say "the reward *appears* to prioritize X over Y," not "the reward is
wrong about Y." (See `CLAUDE.md` for the full tone rules.)

### Short status updates

While the skill is running, post **one-sentence** status messages at
key transitions (e.g. "Analysing the collected trajectories.",
"Selecting the most informative comparisons.", "Rendering the
visualizations.") via `python -m web_ui.cli post --type status`. The
website automatically shows a "Claude is thinking" spinner whenever
no question is pending, so you don't need to repeat that idea — just
post a single-sentence label when something interesting changes.

## Web UI is the only user-facing channel

The web UI was launched by `/repair-setup`. Every user-visible message
here goes through `python -m web_ui.cli post`; every question goes
through `python -m web_ui.cli ask`. **Do not** call the
`AskUserQuestion` tool. See `CLAUDE.md` → "Web UI — the user-facing
channel" for the reference.

If the server isn't running, bring it up first:
`python -m web_ui.cli launch`.

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-preferences hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, post a message asking the user to run `/repair-setup`
first and stop. Set the UI header:

```bash
python -m web_ui.cli set workspace=<WORKSPACE> status=running
```

## Preconditions

- At least one round of evidence has been collected for this workspace.
  If none, post a message asking the user to run `/repair-round` first
  and stop.

## Procedure

### 1. Select the comparisons — you choose them, not an algorithm

Comparison selection is **Claude-driven**. You read every collected
trajectory (summarised by the same features the user sees on the
website), look at the designed reward and every alternative-perspective
metric, and pick the comparisons most useful for the user to see. The
legacy algorithmic selector lives in `find_informative_pairs.py` and is
no longer used for this step.

Two kinds of comparisons are available:

- **Pairs** (`{a, b}`): a head-to-head A-vs-B comparison.
- **Lottery comparisons** (`{c, a, b, p}`): a certain trajectory C vs a
  `p`-vs-`(1−p)` lottery over A and B. Use when you want to expose
  curvature or risk-attitude disagreements that a head-to-head can't.

Pick the **top 5 most informative pairs** and the **top 2 most
informative lotteries**. Favor diversity — don't repeat the same
trade-off twice; each comparison should illuminate a distinct way the
reward might be misaligned.

**1a. Dump the trajectory table.**

```bash
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python select_pairs.py --workspace workspaces/<WORKSPACE> --mode dump
```

This writes `workspaces/<WORKSPACE>/.repair_history/_trajectory_table.txt`:
one row per trajectory, with a stable id (`T001`, …), its total under
the designed reward, every alternative-perspective metric, and every
feature.

**1b. Read the table and reason about it.** Read that file with the Read
tool. Each row is a trajectory encoded as features — the same encoding
the user reads — which is what lets you reason about what happened.

For **pairs**, identify head-to-head comparisons where the designed
reward ranks the two trajectories oppositely to another perspective the
user plausibly cares about, or that otherwise best expose how the
reward trades off the outcomes that matter.

For **lotteries**, identify triples (C, A, B) where: A is a clearly
good outcome under some perspective's lens, B is a clearly bad outcome
under that lens, and C is a midpoint that is not obviously good or
bad. By convention let `p` be the probability of the **good** branch
(A) and `(1−p)` the probability of the **bad** branch (B). Then pick
a `p ∈ [0, 1]` (free real, not restricted to round values) such that
the comparison between `return(C)` and `p·return(A) + (1−p)·return(B)`
under the designed reward is most informative about whether the
reward is too risk-seeking or too risk-averse relative to that
perspective.

**Pick `p` so the choice between C and the lottery is genuinely
difficult — *don't* default to a fixed number.** "Genuinely difficult"
means: the designed reward gives a clear, definite preference, but a
reasonable human reading the lottery could plausibly disagree.

How to find such a `p` for a given `(C, A, B)`:

1. Compute the designed-reward indifference point
   `p* = (return(C) − return(B)) / (return(A) − return(B))`. At this
   `p`, the reward sees C and the lottery as tied — uninformative.
2. Move slightly off `p*` so the reward picks a side with a small but
   real margin. The interesting region is usually on the C-preferring
   side — designed says "prefer C," but the human's intuition might
   still pick the lottery because the bad branch is rare or the good
   branch is much more aligned with their stated objective.
3. The right `(1−p)` is whatever makes that side genuinely tempting —
   often quite small (a few percent or less) when B carries a large
   negative reward, but it depends entirely on the magnitudes in the
   trajectories you picked. Don't hardcode 5%, 0.5, or any other
   round number. Compute it.
4. Always justify the chosen `p` in the selection file by explaining
   what makes *this specific* `p` the hard one for the human.

If you want to surface two lotteries, prefer two *different shapes*
(different C/A/B trajectories, or a fundamentally different trade-off)
over two values of `p` on the same shape — sweeping `p` on one shape
is a single question rendered twice.

You **must justify** your choice of `p` in the selection file —
explain in one sentence why this specific `p` makes the disagreement
legible.

For both kinds, favor diversity across perspectives, behaviors, and
reward terms. Use the internal `src` column only to judge diversity;
never reveal round, policy, or step counts to the user. Aim for up to
5 distinct pairs and up to 2 distinct lotteries; fewer is fine if you
can't find that many genuinely informative ones.

**1c. Write the selection and render it.** Write a selection file to
`workspaces/<WORKSPACE>/.repair_history/_selection.json`:

```json
{
  "pairs": [
    {"a": "T012", "b": "T034"},
    {"a": "T100", "b": "T077"}
  ],
  "lotteries": [
    {"c": "T020", "a": "T005", "b": "T088", "p": 0.5,
     "justification": "p=0.5 puts the expected reward of the lottery just above C, exposing whether the safety reviewer would still pick C to avoid B's downside"}
  ]
}
```

Then render:

```bash
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python select_pairs.py --workspace workspaces/<WORKSPACE> --mode render \
       --selection workspaces/<WORKSPACE>/.repair_history/_selection.json
```

The render step orients A = the reward-preferred side for pairs, lists
the other perspectives that prefer B, prints each pair in the
preference-label format, and for each comparison the workspace's
`viz.py` saves per-trajectory visualizations to
`workspaces/<WORKSPACE>/viz/round{snap_id}/{pair,lottery}{rank}/…` and
prints those paths.

For lotteries it also computes `return(C)` and `p·return(A) +
(1−p)·return(B)` under the designed reward, prints the lottery as a
question (certain trajectory C vs the lottery), and lists the other
perspectives whose preference disagrees with the reward's.

The render also writes a **ready-to-post UI payload** to
`workspaces/<WORKSPACE>/.repair_history/_ui_payload/` (printed at the end
of its stdout as `[ui_payload] …`): a `manifest.json` listing every card
in render order, one `*.content.txt` per card (the exact block to post,
already stripped of the trailing visualizations list), one `*.viz.json`
per card (the `--viz` argument, with this round's visuals — `src` URLs,
`preferred`, and lottery `role`/`probability` — already filled in), and a
`feature_key.json`. The bundle is regenerated on every render, so its
visuals always belong to the round just rendered. You supply only the
per-round judgment text (the overall-pattern paragraph and each card's
`--why`); you do **not** hand-build viz JSON, content files, or the
feature key any more. Use it in Step 2.

The render's stdout (and these payload files) are the source the agent
forwards to the web UI in Step 2; nothing is copy-pasted into the chat.

(If the table holds fewer than 3 trajectories that form any useful
disagreeing comparison, jump to Step 3 and post a status message saying
there's not yet enough evidence to surface diverse disagreements.)

### 2. Forward the deliverable to the web UI

This is the load-bearing step. Carry out all of the following before
moving on.

**2a. Overall pattern paragraph** (post first, before any comparison):

```bash
python -m web_ui.cli post --type markdown --content "<your overall pattern + 1–2 possible reward tweaks, ≤ 6 lines, framed as 'the reward *appears* to prioritize X over Y — if that wasn't intended, one option is …'. Never 'the reward is wrong about X.'>"
```

**2b. Orientation status:**

```bash
python -m web_ui.cli post --type status --content "View the trajectory comparisons below to note where the designed reward function may be misaligned. You'll be asked what to do next."
```

**2c. Feature key in the right sidebar.** Post the feature key the render
already emitted — don't reconstruct it:

```bash
python -m web_ui.cli post --type feature_key \
    --features-file workspaces/<WORKSPACE>/.repair_history/_ui_payload/feature_key.json
```

**2d. Each comparison** (pairs first, in render order; then lotteries).
Read
`workspaces/<WORKSPACE>/.repair_history/_ui_payload/manifest.json` and
post one comparison per entry in its `cards` list, **in order**. Every
card already carries this round's `content_file`, `viz_file` (the exact
`--viz` argument, with the round-specific `src` URLs, `preferred` badge,
and lottery `role`/`probability` already filled in), `label`, and
`layout`. You add **only** the `--why` judgment line:

```bash
# pair card (layout is null):
python -m web_ui.cli post --type comparison \
    --label "<card.label>" \
    --content-file "<card.content_file>" \
    --viz "$(cat <card.viz_file>)" \
    --why "<1–2 sentences naming the trade-off the reward made. For pairs: look at the largest entries in the Δ (B−A) column — those features drive the disagreement. Phrase as: 'the reward *appears* to prioritize <reward-term> over <other-perspective-priority>, scoring A higher because <feature shift>; this perspective would have chosen the opposite trade-off because <their priority>.'>"

# lottery card (layout is "lottery"): add --layout lottery
python -m web_ui.cli post --type comparison \
    --label "<card.label>" --layout lottery \
    --content-file "<card.content_file>" \
    --viz "$(cat <card.viz_file>)" \
    --why "<1–2 sentences naming the risk-attitude or curvature trade-off between C and the lottery. Phrase as: 'the reward *appears* to treat the lottery as worth more than C (i.e. it is risk-seeking between A's upside and B's downside relative to this perspective's view); this perspective prefers C because <their priority — usually downside aversion or upside skepticism>.'>"
```

Pass `--layout <card.layout>` only when the card's `layout` is non-null
(lotteries). Omit `--viz` only when a card's `viz_file` is null (the
workspace has no `viz.py`, so no visuals were produced). The `--why`
line is the one piece you author per card; it differs each round, just
as the visuals do.

Don't name a metric by its internal id; either say "the candidate
performance metric" or refer to the perspective ("the safety
reviewer's view", etc.). Do not use the word "stakeholder". Do not
re-introduce policy class names or round numbers — the payload
intentionally hides them. Do not edit the `content_file` — post it
exactly as the render produced it.

### 3. Always close by asking the user what to do next

Use `python -m web_ui.cli ask` with exactly these four options (or
wording very close to them):

```bash
python -m web_ui.cli ask --prompt "What would you like to do next?" \
    --options '[
      {"label":"Edit the reward function","description":"Modify the reward to reflect what you actually want, based on the disagreements above."},
      {"label":"Add another trajectory feature","description":"Add a new feature to the trajectory summary so the comparisons show more detail."},
      {"label":"View more comparisons","description":"Surface additional pairs or lotteries from the evidence already collected."},
      {"label":"Finish and submit (not reversible)","description":"Lock in the current reward function as the final answer. This ends the session."}
    ]'
```

Read `value` from the returned JSON.

### 4. Act on the answer

- **Edit the reward function** → ask whether the user wants to make the
  edit themselves or have Claude make it:

  ```bash
  python -m web_ui.cli ask --prompt "Who should make the reward edit?" \
      --options '[{"label":"You (Claude) make it","description":"I describe the change in the next message; Claude applies it and reshows the file."},
                  {"label":"I will make it myself","description":"I edit the file directly and continue when ready."}]'
  ```

  - *Claude makes the edit:* first post the reward-edit tip box (below)
    to the web UI, then ask the user to describe the change via a
    long-text question, apply the edit with `Edit`, post the new
    source as a code message, and re-score: re-run this skill
    (trajectories on disk are unchanged, so it just re-scores under
    the new reward) or suggest `/repair-verify`.
  - *User edits it themselves:* post a status message saying the
    system is waiting; when they're ready they can run `/repair-verify`
    to re-check the previously-surfaced pairs, **or** `/repair-round`
    to gather fresh evidence under the new reward.

- **Add another trajectory feature** → ask the user, via a long-text
  question, what feature they want. Edit
  `gather_evidence/trajectory_features.py`, re-run
  `runtime.compute_features` over every existing round so all
  trajectories carry the new feature, then re-present the evidence
  (Steps 1–3) with the feature visible.

- **View more comparisons (pairs or lotteries)** → behaviour depends on
  whether this is the *first* "View more" click in a row or the
  *second* consecutive one (since the last reward edit, feature add,
  or fresh /repair-round).

  **Second consecutive click → hand off to /repair-round.** Once
  another batch of within-evidence comparisons has already been
  surfaced, the existing trajectories are largely exhausted of
  distinct disagreements. Don't re-select from disk a third time —
  instead invoke `Skill(skill="repair-round")`. The reward function
  is unchanged, so the new round's policies will see the *same*
  designed reward that produced the disagreements the user is still
  trying to resolve; that's the intended effect ("run another round
  without modifying the reward"). Reset the within-a-row counter
  whenever the user edits the reward, adds a feature, or a /repair-
  round happens for any other reason.

  **First click → re-select within existing evidence (this step's
  default).** Re-run Step 1, but with one crucial extra constraint:
  **treat every previously-shown trajectory as one whose designed-
  reward ranking the user has now agreed with**. By pressing "view
  more comparisons", the user has implicitly confirmed that the
  designed reward got it right on everything you've shown so far —
  re-showing those same rankings (or ones that depend on them)
  wastes their time.

  Concretely:
  - Walk through every previously-saved snapshot
    (`workspaces/<WORKSPACE>/.repair_history/pref_round_*.json`) and
    enumerate every trajectory that has appeared in a pair or lottery
    so far. Those are the "agreed" trajectories.
  - **Do not surface a new pair whose disagreement could already be
    inferred from the agreed set.** If the user already saw a pair
    "A ≻ B" along some axis (e.g. one outcome-class beats another by
    a particular feature) and agreed, don't surface another pair
    that re-instantiates the same axis with near-equivalent
    trajectories — the user has effectively endorsed that trade-off.
  - **Prefer pairs that introduce at least one trajectory the user
    hasn't seen yet**, and that surface a *new axis* of disagreement
    (a new perspective, a new feature, a finer-grained trade-off).
  - For lotteries the same idea applies: the user has agreed with the
    reward's preference between the trajectories used in past
    lotteries, so a new lottery should explore a different `(C, A, B,
    p)` shape — e.g. a different bad-event probability that probes
    risk-aversion at a different point on the curvature.

  Then present the new selection per Steps 2–3.

- **Finish and submit (not reversible)** → follow "Finish and submit"
  in `CLAUDE.md`: write the submission record, post a confirmation to
  the web UI (`post --type markdown --content "The reward function is
  submitted. Session complete."`) plus `set status=done`, and stop.

#### Reward-edit tip box (post when Claude is about to edit the reward)

```bash
python -m web_ui.cli post --type markdown --content "💡 **Tips for guiding a reward edit**
- Be concrete: name the term to change, and — if you can — the preference label you want for each shown pair (e.g. \"for pair 2, A ≻ B should become B ≻ A\").
- You can pin an edit to an outcome: e.g. \"increase this penalty until the preference for pair 2 flips,\" and Claude will solve for the smallest change that produces it."
```

## When the user edits the reward and asks you to recheck

Re-run this skill; trajectories on disk are unchanged, so the
disagreement finder simply re-scores them under the new reward and the
new comparisons get posted to the web UI.
