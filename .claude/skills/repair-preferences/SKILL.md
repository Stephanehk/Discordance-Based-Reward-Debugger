---
name: repair-preferences
description: Show the 15 most informative trajectory pairs as preference labels. For each pair: features of both trajectories, scores from the designed reward and the candidate performance metric, the A≻B / B≻A label each induces, and a short explanation of why the stakeholder disagrees. Ends with an overall interpretation of likely reward flaws.
---

# /repair-preferences — show preference-label disagreements

## ⚠ CORE CONTRACT (do not skip)

**Up to 15 pretty-printed pairs — every disagreement returned by the
script — MUST appear in your text response, verbatim, with TWO summary
lines printed directly under each pair (interleaved, not in a single
block after the last pair):** a `**Why they disagree:**` line and a
`**What you might change in the reward function to flip this
preference:**` line. The bash tool result is invisible to the user —
only what you type into the response is. If your response does not
contain the pairs themselves (feature key + every PAIR block returned
by the script, copied unchanged) with **both** per-pair summary lines
attached to each one, the skill has failed regardless of how good the
overall interpretation is.

A correct response looks like:

````
```
<feature key, copied verbatim>
```

```
<PAIR 1/15 block, copied verbatim — header through feature table.
 Do not change any text or numbers.>
```
**Why they disagree:** <1–2 sentences naming the trade-off the reward
made, the feature deltas that drove it, and why this stakeholder would
have chosen oppositely>
**What you might change in the reward function to flip this preference:**
<1–2 sentences naming a concrete edit — which weight to change, which
term to add, remove, or gate — that would cause the designed reward to
prefer the stakeholder's side on this pair. Framed as an option, not a
verdict.>

```
<PAIR 2/15 block, copied verbatim>
```
**Why they disagree:** ...
**What you might change in the reward function to flip this preference:** ...

...

```
<PAIR 15/15 block, copied verbatim>
```
**Why they disagree:** ...
**What you might change in the reward function to flip this preference:** ...

<overall pattern + 1–2 possible reward tweaks, ≤ 6 lines, framed as
"the reward *appears* to prioritize X over Y" — never as "the reward
is wrong">

<AskUserQuestion call: another round of evidence / modify reward
yourself / Claude modifies the reward>
````

Do not paraphrase the pairs. Do not lump the per-pair summary lines
into a single section after the last pair — both summary lines go
directly beneath the pair they explain. Do not skip the feature key.
Do not skip either summary line on any pair. Do not skip the closing
`AskUserQuestion` — the user must always be offered a next step.

The script anonymizes trajectory identity in the printed header — each
trajectory is shown as just "Trajectory A" / "Trajectory B" with no
round, step count, or policy class. Do not re-introduce any of that
information in your text. The user should read each pair on the
merits of its features, not by recognizing where it came from.

## Tone for messages to the user

The user will never read the codebase. Don't list file names, internal
scripts, or boilerplate paths. **Never name a metric by its internal
identifier** (e.g. `performance_metric_<short_name>`). Refer to "the
candidate performance metric" or describe the stakeholder perspective
from the docstring (e.g. "the safety reviewer's view", "the operations
stakeholder").

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with the stakeholder's view." Whether
something is actually right is the user's call — the system surfaces
evidence, not verdicts. The same applies to your overall interpretation:
say "the reward *appears* to prioritize X over Y," not "the reward is
wrong about Y." (See `CLAUDE.md` for the full tone rules.)

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-preferences hopper`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, tell the user to run `/repair-setup` first.

## Preconditions

- At least one round of evidence has been collected for this workspace.
  If none, tell the user to run `/repair-round` first.

## Procedure

### 1. Pull the 15 most informative disagreements

```
python find_informative_pairs.py --workspace workspaces/<WORKSPACE> \
       --top-overall 15 --format preference-labels
```

The script applies dedup across (stakeholder, policy_A_class,
policy_B_class) triples, so the candidates already cover a variety of
stakeholders and policy pairings. Read the full output — it is the
deliverable.

(If the script returned fewer than 15 candidates, just show whatever
came back. If it returned fewer than 3, jump to Step 3 and tell the
user there's not yet enough evidence to surface diverse disagreements.)

### 2. Reprint every returned pair verbatim, interleaved with per-pair summaries

See the CORE CONTRACT above. This is the load-bearing step.

1. Start with the feature key inside a fenced code block.
2. Then for **every** pair returned by the script, in the order the
   script returned them:
   a. Print the pair's `PAIR …` block **verbatim** inside its own
      fenced code block — header through the feature table. Do not
      change any text or numbers, including the `PAIR k/N` header.
   b. Immediately after that code block (still in the same response,
      no other content in between), write a `**Why they disagree:**`
      line followed by 1–2 sentences. Look at the largest entries in
      the `Δ (B−A)` column — those are the features driving the
      disagreement. **Frame the explanation as a trade-off the reward
      made.** Phrase as: "the reward *appears* to prioritize
      <reward-term> over <stakeholder-priority>, scoring A higher
      because <feature shift>; this stakeholder would have chosen the
      opposite trade-off because <their priority>."
   c. On the very next line, write a
      `**What you might change in the reward function to flip this preference:**`
      line followed by 1–2 sentences. Read the body of
      `designed_reward_fn.py` and name a **concrete, specific edit** —
      bump a weight, lower a threshold, add a new term, gate a term on
      a feature — that would cause the designed reward to prefer the
      stakeholder's side on this specific pair. Mention the variable
      or constant name from the reward source so the suggestion is
      actionable. Phrase as an option, not a directive: "one option is
      to …" or "if you wanted B to win this pair, raise <CONST> from X
      to Y."

Do not lump the per-pair summary lines into a single section at the
end. Both summary lines go directly under the pair they explain — the
"Why they disagree" first, then "What you might change …" on the
following line, in that order.

Don't name a metric by its internal id; either say "the candidate
performance metric" or refer to the stakeholder perspective ("the
safety reviewer's view", "the operations stakeholder", etc.). Do not
re-introduce policy class names — the script intentionally hides them.
Do not truncate, summarise, or paraphrase any pair. If the script
saved a snapshot path on the last line, you can omit that — it's an
internal detail.

### 3. Overall interpretation — name the trade-off, not the verdict

After every pair (each followed by its own two summary lines), write a
short (≤ 6 lines) interpretation that:
- names the **pattern** common to the disagreements
- points to the **reward term** or **implicit priority** that drove
  the trade-off (by referencing what the reward credits/penalizes —
  not by file path or variable name)
- suggests 1–2 concrete tweaks the user *could* make to close the gap

**Stay descriptive, never prescriptive.** Don't say "the reward is
wrong about X" or "the reward should be Y." Say "the reward *appears*
to prioritize X over Y — if that wasn't intended, one option is …".
Whether the trade-off is acceptable is the user's call.

### 4. Always close by asking the user what to do next

Use `AskUserQuestion` with exactly these three options (or wording very
close to them):

- **Run another round of evidence** — "the pairs above weren't
  informative; collect more focused evidence and try again."
- **I'll modify the reward myself** — user will edit
  `designed_reward_fn.py` and call back when ready.
- **Modify the reward for me** — Claude proposes and applies edits
  targeted at the disagreements above.

Phrase the question in user-friendly language ("How would you like to
proceed?" or "Where to next?"). Do **not** mention `/repair-round`,
`/repair-preferences`, or any other command name in the question text
itself — but do feel free to act on the answer by running the relevant
skill.

Act on the answer:
- "Another round of evidence" → continue inline into `/repair-round`'s
  rubric (and re-enter this skill when its rubric is satisfied).
- "I'll modify the reward myself" → tell the user the system will wait;
  remind them that when they're done they can run `/repair-verify` to
  re-check the previously-surfaced pairs, **or skip that step and run
  `/repair-round` directly** to gather fresh evidence under the new
  reward (verification only re-scores the saved pairs; a new round will
  surface fresh disagreements that may have emerged from the edit).
- "Modify the reward for me" → propose specific edits to
  `designed_reward_fn.py` that address the disagreements above, apply
  them with `Edit`, show the diff back to the user, then suggest either
  `/repair-verify` (re-check the past pairs) **or** `/repair-round`
  (skip verification and gather fresh evidence under the new reward).

## When the user edits the reward and asks you to recheck

Re-run this skill; trajectories on disk are unchanged, so the
disagreement finder simply re-scores them under the new reward.
