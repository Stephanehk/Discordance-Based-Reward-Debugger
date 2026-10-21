---
name: repair-preferences
description: Show up to 5 trajectory pairs plus up to 2 lottery comparisons (certain trajectory C vs a p-probability lottery over A and B) on the web UI. For each: the trajectory videos, a short "why they disagree" line, and one sentence per characteristic contrasting the two trajectories.
---

# /repair-preferences — show preference-label disagreements on the web UI

## ⚠ CORE CONTRACT (do not skip)

**Every comparison the script returns — up to 5 PAIR blocks and up to 2
LOTTERY blocks — MUST be posted to the web UI via `python -m web_ui.cli
post --type comparison`, each with its own `--why` interpretation
line.** The chat / terminal output the agent emits is **not** the
deliverable; the website is. The render script's stdout is consumed by
the agent, parsed, and forwarded to the UI — not pasted into the chat.

Post **only** the comparisons (pairs first, then lotteries), each with
its `--why` line. Do **not** post a feature key, overall summary,
pattern paragraph, or misalignment orientation.

Finally, **always** close with the three-option `ask` (edit /
propose more / end session). The skill has not delivered
its core output until that question is on the website.

## Tone for messages to the user

The user will never read the codebase. Don't include file names,
internal scripts, or boilerplate paths in anything posted to the
website. The alternative-perspective metrics are **under-the-hood
selection signals only** — never mention a perspective, metric, or
"stakeholder" in anything posted to the user.

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect.
Whether something is actually right is the user's call — the system
surfaces evidence, not verdicts. (See `CLAUDE.md` for the full tone
rules.)

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

**Informativeness is judged against the input reward function and its
suspected flaws.** Read `<WORKSPACE_DIR>/designed_reward_fn.py` (the
reward whose flaws this whole process exists to surface) and the reward
decomposition `<WORKSPACE_DIR>/.repair_history/_reward_analysis.md` (its
per-term breakdown and the exploit hypotheses `H1, H2, …`). A comparison
is informative to the extent that it exposes one of those suspected
flaws — a case where a specific reward term or exploit hypothesis drives
the designed reward to prefer a trajectory the user would plausibly
reject. Comparisons you can tie to a term or hypothesis in
`designed_reward_fn.py` outrank generic disagreements that don't point at
any suspected flaw.

Two kinds of comparisons are available:

- **Pairs** (`{a, b}`): a head-to-head A-vs-B comparison.
- **Lottery comparisons** (`{c, a, b, p}`): a certain trajectory C vs a
  `p`-vs-`(1−p)` lottery over A and B. Use when you want to expose
  curvature or risk-attitude disagreements that a head-to-head can't.

Pick the **top 5 most informative pairs** and the **top 2 most
informative lotteries**. Favor diversity — don't repeat the same
trade-off twice; each comparison should illuminate a distinct way the
reward might be misaligned.

**1a-pre. Refresh the reward decomposition.** Before selecting, ensure
`<WORKSPACE_DIR>/.repair_history/_reward_analysis.md` matches the current
reward hash; if the user edited the reward since it was written,
regenerate it following `prompts/reward_decomposition_prompt.md` (see
`CLAUDE.md` → "Reward introspection" for the reuse rule), so selection
reasons about the *current* reward. All of this reward-flaw reasoning
stays **internal** — the user-facing `--why` and `--content` still follow
the tone rules below (no term names, perspectives, or "exploit").

**1a. Dump the trajectory table.**

```bash
ENV=$(cat workspaces/<WORKSPACE>/.conda_env 2>/dev/null); if [ -n "$ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV"; fi; \
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
user plausibly cares about, or that otherwise best expose where the reward function is flawed and how the
reward trades off the outcomes that matter. **Prioritize pairs you can
tie to a specific reward term or exploit hypothesis `H#`** from the
decomposition: the strongest comparison is one where the reward prefers A
because a particular term fired (a proxy term, a farmable dense term, a
skipped sparse term) while B is the trajectory that actually achieves the
underlying outcome — read the proxy and true-outcome feature columns to
confirm the decoupling is real in the two trajectories you pick. A pair
that shows the reward preferring a proxy-gaming trajectory over a
genuine-success trajectory should outrank a generic metric disagreement.

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
    {"a": "T012", "b": "T034",
     "reward_basis": "H[n] [proxy term]: reward prefers A because the proxy term fired; B achieves the true outcome the proxy stands in for",
     "characteristics": ["<short_name_1>", "<short_name_2>", "<short_name_3>"]},
    {"a": "T100", "b": "T077",
     "characteristics": ["<short_name_1>", "<short_name_2>"]}
  ],
  "lotteries": [
    {"c": "T020", "a": "T005", "b": "T088", "p": 0.5,
     "justification": "p=0.5 puts the expected reward of the lottery just above C, exposing whether the safety reviewer would still pick C to avoid B's downside"}
  ]
}
```

`"reward_basis"` is an **optional internal** field (the renderer ignores
it; it is never shown to the user): one phrase naming the reward term or
exploit hypothesis `H#` that drives this pair's questionable preference.
Add it to every pair you can tie to the decomposition, and let it guide
the `characteristics` ordering — put the proxy and true-outcome features
that explain the preference first.

For each pair, **rank the characteristics** via the optional
`"characteristics"` list — the characteristic `short_name`s (the table's
feature columns), ordered most→least informative about *why the designed
reward's preference may be wrong for this pair*. The display lists them
in that order (any you omit follow in their default order). Put the
characteristics that best explain the questionable preference first.

Then render:

```bash
ENV=$(cat workspaces/<WORKSPACE>/.conda_env 2>/dev/null); if [ -n "$ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV"; fi; \
python select_pairs.py --workspace workspaces/<WORKSPACE> --mode render \
       --selection workspaces/<WORKSPACE>/.repair_history/_selection.json
```

The render step orients A = the reward-preferred side for pairs (so A is
"Designed reward function's preferred trajectory", B "the alternative trajectory"), prints one
sentence per characteristic contrasting the two, and for each comparison
the workspace's `viz.py` saves per-trajectory visualizations to
`workspaces/<WORKSPACE>/viz/round{snap_id}/{pair,lottery}{rank}/…` and
prints those paths.

For lotteries the render emits **no** user-facing content — you author a
short plain-English summary of how A, B, and C differ (see Step 2).
Perspective metrics drive selection under the hood but are never printed.

The render also writes a **ready-to-post UI payload** to
`workspaces/<WORKSPACE>/.repair_history/_ui_payload/` (printed at the end
of its stdout as `[ui_payload] …`): a `manifest.json` listing every card
in render order, one `*.content.txt` per card (for pairs, the exact
sentences to post; empty for lotteries), and one `*.viz.json` per card
(the `--viz` argument, with this round's visuals — `src` URLs,
`preferred`, and lottery `role`/`probability` — already filled in). The
bundle is regenerated on every render, so its visuals always belong to
the round just rendered. You supply each pair's `--why` line, and for
each lottery both its `--content` summary and its `--why` line; you do
**not** hand-build viz JSON. Use it in Step 2.

The render's stdout (and these payload files) are the source the agent
forwards to the web UI in Step 2; nothing is copy-pasted into the chat.

(If the table holds fewer than 3 trajectories that form any useful
disagreeing comparison, jump to Step 3 and post a status message saying
there's not yet enough evidence to surface diverse disagreements.)

### 2. Forward the deliverable to the web UI

This is the load-bearing step. Post **only** the comparisons — no
feature key, overall summary, pattern paragraph, or misalignment
orientation.

**Each comparison** (pairs first, in render order; then lotteries).
Read
`workspaces/<WORKSPACE>/.repair_history/_ui_payload/manifest.json` and
post one comparison per entry in its `cards` list, **in order**. Every
card already carries this round's `content_file`, `viz_file` (the exact
`--viz` argument, with the round-specific `src` URLs, `preferred` badge,
and lottery `role`/`probability` already filled in), `label`, and
`layout`:

```bash
# pair card (layout is null): post the render's sentences verbatim.
python -m web_ui.cli post --type comparison \
    --label "<card.label>" \
    --content-file "<card.content_file>" \
    --viz "$(cat <card.viz_file>)" \
    --why "<1 sentence naming the trade-off between the two trajectories — what the designed reward function's preferred trajectory gains and what it gives up. Don't mention perspectives or metrics.>"

# lottery card (layout is "lottery"): you author --content (the content_file is empty).
python -m web_ui.cli post --type comparison \
    --label "<card.label>" --layout lottery \
    --content "<2–3 sentences on how Trajectory A (good branch), Trajectory B (bad branch), and the certain Trajectory C differ. Be QUANTITATIVE: state the actual characteristic values (with units) from the table; do NOT use qualitative interpretations like 'safe', 'smooth', or 'aggressive'. Don't mention perspectives or metrics.>" \
    --viz "$(cat <card.viz_file>)" \
    --why "<1 sentence naming the risk-attitude trade-off between the certain trajectory C and the lottery. Don't mention perspectives or metrics.>"
```

For pairs, post `--content-file` exactly as the render produced it. For
lotteries, supply your own `--content` summary (the render leaves the
lottery content empty). Pass `--layout lottery` only for lottery cards.
Omit `--viz` only when a card's `viz_file` is null (the workspace has no
`viz.py`, so no visuals were produced).

Never mention a perspective, metric, or "stakeholder" — they are
under-the-hood signals only. Do not re-introduce policy class names or
round numbers — the payload intentionally hides them. Do not edit a
pair's `content_file` — post it exactly as the render produced it.

### 3. Always close by asking the user what to do next

Use `python -m web_ui.cli ask` with exactly these three options (or
wording very close to them):

```bash
python -m web_ui.cli ask --prompt "What would you like to do next?" \
    --options '[
      {"label":"Edit the reward function","description":"Modify the reward to reflect what you actually want, based on the discordances above."},
      {"label":"Propose more discordances","description":"Propose more pairs or lotteries from the trajectories already collected."},
      {"label":"End session","description":"Keep the current reward function and close the tool."}
    ]'
```

Read `value` from the returned JSON.

### 4. Act on the answer

- **Edit the reward function** → ask whether the user wants to make the
  edit themselves or have Claude make it:

  ```bash
  python -m web_ui.cli ask --prompt "Who should make the reward edit?" \
      --options '[{"label":"You (Claude) make it","description":"I describe the change in the next message; Claude applies it and reshows the file."},
                  {"label":"I will make it myself","description":"I edit the file directly and press a button when done."}]'
  ```

  - *Claude makes the edit:* first post the reward-edit tip box (below)
    to the web UI, then ask the user to describe the change via a
    long-text question, apply the edit with `Edit`, post the new
    source as a code message, and re-score: re-run this skill
    (trajectories on disk are unchanged, so it just re-scores under
    the new reward) or suggest `/repair-verify`.
  - *User edits it themselves:* name the file, then wait for the
    user's signal that the edit is done (the button is that signal):

    ```bash
    python -m web_ui.cli post --type markdown --content "Edit \`workspaces/<WORKSPACE>/designed_reward_fn.py\` in the repo, then press the button below."
    python -m web_ui.cli cta --prompt "Waiting for your reward edit." --label "Done editing" --value "done"
    ```

    Then continue exactly as in the Claude branch: post the new source
    as a code message and re-score.

- **Propose more discordances** → behaviour depends on
  whether this is the *first* "Propose more" click in a row or the
  *second* consecutive one (since the last reward edit or fresh
  /repair-round).

  **Second consecutive click → hand off to /repair-round.** Once
  another batch of within-evidence comparisons has already been
  surfaced, the existing trajectories are largely exhausted of
  distinct disagreements. Don't re-select from disk a third time —
  instead invoke `Skill(skill="repair-round")`. The reward function
  is unchanged, so the new round's policies will see the *same*
  designed reward that produced the disagreements the user is still
  trying to resolve; that's the intended effect ("run another round
  without modifying the reward"). Reset the within-a-row counter
  whenever the user edits the reward or a /repair-round happens for any other reason.

  **First click → re-select within existing evidence (this step's
  default).** Re-run Step 1, but with one crucial extra constraint:
  **treat every previously-shown trajectory as one whose designed-
  reward ranking the user has now agreed with**. By pressing "Propose
  more discordances", the user has implicitly confirmed that the
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

- **End session** → follow "End session"
  in `CLAUDE.md`: write the submission record, post a confirmation to
  the web UI (`post --type markdown --content "Session ended. The
  current reward function is kept."`) plus `set status=done`, and stop.

#### Reward-edit tip box (post when Claude is about to edit the reward)

```bash
python -m web_ui.cli post --type markdown --content "💡 **Tips for guiding a reward edit**
- Be concrete: name the term to change, and — if you can — which discordances you agree with (e.g. \"I agree with discordances 1, 2, and 4\").
- You can pin an edit to an outcome: e.g. \"increase this penalty until the ranking in discordance 2 flips,\" and Claude will solve for the smallest change that produces it."
```

## When the user edits the reward and asks you to recheck

Re-run this skill; trajectories on disk are unchanged, so the
disagreement finder simply re-scores them under the new reward and the
new comparisons get posted to the web UI.
