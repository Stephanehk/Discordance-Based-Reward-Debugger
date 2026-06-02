---
name: repair-verify
description: Check whether the updated reward function resolves the disagreements surfaced by past /repair-preferences runs. For every saved pair and lottery comparison, recompute the designed reward under the current reward function and report on the web UI whether the reward now agrees with that perspective. Posts any comparisons that still disagree. Use after the user edits the reward and wants to confirm the fix.
---

# /repair-verify — confirm the reward fix addresses past disagreements

Whenever `/repair-preferences` runs, it saves a snapshot of the
surfaced trajectory pairs and lottery comparisons (which trajectories,
the probability `p` for lotteries, which perspective, which side that
perspective preferred). This skill loads every past snapshot,
recomputes the designed reward on each pair and lottery under the
**current** reward function, and reports which comparisons the new
reward now agrees with — and which still disagree — by posting to the
web UI.

## Tone for messages to the user

The user will never read the codebase. Don't post internal file names,
snapshot paths, or script names. Don't name metrics by their internal
identifiers (say "the candidate performance metric" or describe the
perspective). Don't use the word "stakeholder" — say "other
perspectives" or describe the specific perspective. Speak in terms of
outcomes: "every past disagreement is resolved" or "1 pair still
disagrees — here it is".

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** A pair being "resolved" means the reward now agrees
with that perspective on that pair — not that the reward is "right."
Say "the updated reward now agrees with this perspective" or "appears
to align with the surfaced preference," not "the reward is correct now."
Whether the reward is actually right is the user's call. (See
`CLAUDE.md` for the full tone rules.)

## Web UI is the only user-facing channel

The web UI was launched by `/repair-setup`. Every user-visible message
this skill emits goes through `python -m web_ui.cli post`; every
question goes through `python -m web_ui.cli ask`. **Do not** call the
`AskUserQuestion` tool, and do not narrate to the terminal beyond
brief operational status. See `CLAUDE.md` → "Web UI — the user-facing
channel" for the reference.

If the server isn't running, bring it up first:
`python -m web_ui.cli launch`.

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-verify hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, post a message asking the user to run `/repair-setup`
first and stop. Set the UI header:

```bash
python -m web_ui.cli set workspace=<WORKSPACE> status=running
```

## What to do

### 1. Run the verifier

```bash
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python find_informative_pairs.py --workspace workspaces/<WORKSPACE> --check-snapshots
```

The script:
- loads every previously saved snapshot,
- re-scores both trajectories in every pair, and `return(C)` vs
  `p·return(A) + (1−p)·return(B)` in every lottery, under the current
  reward function,
- reports the resolved / unresolved counts separately for pairs and
  lotteries,
- pretty-prints any pairs and lotteries that still disagree in the
  same preference-labels and lottery-question formats as
  `/repair-preferences`.

The output is for the agent's analysis only — don't paste it raw into
the chat or the web UI. The skill forwards the relevant parts to the
web UI in the next steps.

If no snapshots exist yet, post a status message saying the system has
nothing to verify against yet and recommend the user look for
disagreements first:

```bash
python -m web_ui.cli post --type status --content "There's nothing to verify yet — no comparisons have been surfaced. Run /repair-round to collect some first."
```

…and stop.

### 2. Post the summary

Post a short summary to the web UI naming the resolved / unresolved
counts in plain language:

```bash
python -m web_ui.cli post --type markdown --content "**Verification result** — N of M previously-surfaced disagreements are now resolved by your reward edit; K still disagree."
```

### 3a. If everything is resolved

Post a clear status saying so:

```bash
python -m web_ui.cli post --type status --content "Every past disagreement is now resolved by your reward function."
```

…and skip to Step 5.

### 3b. If some comparisons still disagree

Repost each unresolved comparison to the web UI exactly as
`/repair-preferences` would post a freshly-surfaced one — including
the feature key (right sidebar), each comparison's preformatted block,
its `--why` interpretation, and (where applicable) inline viz from the
captured-frame mp4s / saved pngs.

For each unresolved pair / lottery:

```bash
# Pair (still disagrees): mark the reward-preferred side with preferred:true.
python -m web_ui.cli post --type comparison \
    --label "PAIR k/N still disagrees" \
    --content-file <path to the still-disagreeing block from the script's output> \
    --viz '[{"src":"/workspace/<WORKSPACE>/viz/round{i}/pair{j}/A.mp4","caption":"Trajectory A","preferred":true},
            {"src":"/workspace/<WORKSPACE>/viz/round{i}/pair{j}/B.mp4","caption":"Trajectory B"}]' \
    --why "<1–2 sentences naming the trade-off the reward made.>"

# Lottery (still disagrees): use --layout lottery with role + probability.
python -m web_ui.cli post --type comparison \
    --label "LOTTERY k/M still disagrees" \
    --layout lottery \
    --content-file <path to the still-disagreeing block> \
    --viz '[{"src":"/workspace/<WORKSPACE>/viz/round{i}/lottery{j}/A.mp4","caption":"Trajectory A","role":"A","probability":<p>},
            {"src":"/workspace/<WORKSPACE>/viz/round{i}/lottery{j}/B.mp4","caption":"Trajectory B","role":"B","probability":<1-p>},
            {"src":"/workspace/<WORKSPACE>/viz/round{i}/lottery{j}/C.mp4","caption":"Certain (C)","role":"C"}]' \
    --why "<1–2 sentences naming the risk-attitude or curvature trade-off between C and the lottery.>"
```

After every unresolved comparison is posted, add a short overall
interpretation (≤ 4 lines) that names the pattern in the remaining
disagreement(s) and proposes 1 concrete reward tweak:

```bash
python -m web_ui.cli post --type markdown --content "<pattern + 1 concrete tweak>"
```

### 4. (skipped — flows into 5)

### 5. Always ask whether to continue

Use `python -m web_ui.cli ask` with three options:

```bash
python -m web_ui.cli ask --prompt "Where to next?" \
    --options '[
      {"label":"Look for new disagreements","description":"Re-analyse existing trajectories under the current reward, surfacing up to 5 pairs plus up to 2 lotteries (and a new snapshot)."},
      {"label":"Collect more evidence first","description":"Roll out a fresh round of policies before re-analysing — useful when existing trajectories feel exhausted or your edits warrant new adversarial policies."},
      {"label":"Finish and submit (not reversible)","description":"Lock in the current reward as the final answer. This ends the session."}
    ]'
```

Read `value` from the returned JSON.

### 6. Act on the user's answer

- **Look for new disagreements** → run `/repair-preferences` (follow
  its full contract — overall pattern, feature key sidebar, comparisons
  with `--why` lines, closing four-option question — all posted to the
  web UI).
- **Collect more evidence first** → run `/repair-round`, which itself
  auto-continues into the `/repair-preferences` deliverable.
- **Finish and submit (not reversible)** → follow "Finish and submit"
  in `CLAUDE.md`: write the submission record, post a confirmation to
  the web UI (`post --type markdown --content "The reward function is
  submitted. Session complete."` and `set status=done`), and stop.
- "Other" / decline → just stop.
