---
name: repair-verify
description: Check whether the updated reward function resolves the disagreements surfaced by past /repair-preferences runs. For every saved pair, recompute the designed reward and report whether the reward now agrees with the stakeholder. Prints any pairs that still disagree. Use after the user edits the reward and wants to confirm the fix.
---

# /repair-verify — confirm the reward fix addresses past disagreements

Whenever `/repair-preferences` runs, it saves a snapshot of the
surfaced trajectory pairs (which trajectories, which stakeholder, which
side the stakeholder preferred). This skill loads every past snapshot,
recomputes the designed reward on each pair under the **current**
reward function, and reports which pairs the new reward now agrees
with — and which still disagree.

## Tone for messages to the user

The user will never read the codebase. Don't list internal file names,
snapshot paths, or script names. Don't name metrics by their internal
identifiers (say "the candidate performance metric" or describe the
stakeholder). Speak in terms of outcomes: "every past disagreement is
resolved" or "1 pair still disagrees — here it is".

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** A pair being "resolved" means the reward now agrees
with the stakeholder on that pair — not that the reward is "right."
Say "the updated reward now agrees with this stakeholder" or "appears
to align with the surfaced preference," not "the reward is correct now."
Whether the reward is actually right is the user's call. (See
`CLAUDE.md` for the full tone rules.)

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-verify hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, tell the user to run `/repair-setup` first.

## What to do

### 1. Run the verifier

```
source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh && conda activate llm && \
python find_informative_pairs.py --workspace workspaces/<WORKSPACE> --check-snapshots
```

The script:
- loads every previously saved top-5 snapshot,
- re-scores both trajectories in every pair under the current reward
  function,
- reports the resolved / unresolved counts,
- pretty-prints any pairs that still disagree (in the same preference-
  labels format as `/repair-preferences`).

If no snapshots exist yet, the script will say so; in that case, tell
the user the system has nothing to verify against yet and recommend
they look for disagreements first.

### 2. If there are unresolved pairs, reprint them in your text response

Reprint each unresolved pair verbatim inside its own fenced code block,
with a 1–2 sentence "why they disagree" summary printed **directly
under that pair** (interleaved — not collected at the end). Follow the
same format as `/repair-preferences` Step 2. The script anonymizes
trajectory identity entirely (just "Trajectory A" / "Trajectory B");
do not re-introduce round, step count, or policy class.

After all unresolved pairs (each followed by its own "why" summary),
add a short (≤ 4 lines) overall interpretation that names the pattern
in the remaining disagreement(s) and proposes 1 concrete reward tweak.

### 3. If all past disagreements are resolved

Say so in plain language:

> "Every past disagreement is now resolved by your reward function."

### 4. Always ask whether to continue

Whether all resolved or some remain, end the response by asking the
user whether they want to repeat the analysis to find *new* trajectory
pairs under the updated reward. Use the `AskUserQuestion` tool with
three options:

- **Look for new disagreements** — runs the preference-labels analysis
  again on the existing trajectories; produces a fresh top-5 (and a new
  snapshot) under the current reward.
- **Collect more evidence first** — runs a fresh round of data-
  collection policies before re-analyzing. Use this when the trajectories
  on disk feel exhausted or when the user's reward edits should be
  tested against *new* adversarial policies.
- **Finish and submit (not reversible)** — finalize the current reward
  as the submission. See "Finish and submit" in `CLAUDE.md`.
- (Implicit "Other" lets the user say "no, I'm done", etc.)

Phrase the question in user-friendly language ("Would you like me to
look for more disagreements, or first collect more trajectory data?").
Do **not** name `/repair-preferences` or `/repair-round` in the prompt.

### 5. Act on the user's answer

- "Look for new disagreements" → run `/repair-preferences` (the skill,
  not just the command — follow its full contract including per-pair
  summaries and overall interpretation).
- "Collect more evidence first" → run `/repair-round`, which will
  itself auto-continue into `/repair-preferences`.
- "Finish and submit (not reversible)" → follow "Finish and submit" in
  `CLAUDE.md`: write the submission record and stop.
- "Other" / decline → just stop.
