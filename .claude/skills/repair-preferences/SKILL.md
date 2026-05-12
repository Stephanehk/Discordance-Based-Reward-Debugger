---
name: repair-preferences
description: Show the 5 most informative trajectory pairs as preference labels. For each pair: features of both trajectories, scores from the designed reward and the candidate performance metric, the A≻B / B≻A label each induces, and a short explanation of why the stakeholder disagrees. Ends with an overall interpretation of likely reward flaws.
---

# /repair-preferences — show preference-label disagreements

## ⚠ CORE CONTRACT (do not skip)

**Five pretty-printed pairs — chosen by you from a larger candidate
pool — MUST appear in your text response, verbatim, with a "why they
disagree" summary printed directly under each pair (interleaved, not
in a single block after the last pair).** The bash tool result is
invisible to the user — only what you type into the response is. If
your response does not contain the pairs themselves (feature key + 5
PAIR blocks, copied unchanged from the script output except for the
`PAIR k/N` header renumbering) with a per-pair "why" attached to each
one, the skill has failed regardless of how good the overall
interpretation is.

A correct response looks like:

````
```
<feature key, copied verbatim>
```

```
<PAIR 1/5 block, copied verbatim — header through feature table.
 Renumber only the `PAIR k/N` header; everything else is unchanged.>
```
**Why they disagree:** <1–2 sentences naming the trade-off the reward
made, the feature deltas that drove it, and why this stakeholder would
have chosen oppositely>

```
<PAIR 2/5 block, copied verbatim>
```
**Why they disagree:** ...

...

```
<PAIR 5/5 block, copied verbatim>
```
**Why they disagree:** ...

<overall pattern + 1–2 possible reward tweaks, ≤ 6 lines, framed as
"the reward *appears* to prioritize X over Y" — never as "the reward
is wrong">

<AskUserQuestion call: another round of evidence / modify reward
yourself / Claude modifies the reward>
````

Do not paraphrase the pairs. Do not lump the "why" summaries into a
single section after the last pair — each "why" goes directly beneath
its pair. Do not skip the feature key. Do not skip the closing
`AskUserQuestion` — the user must always be offered a next step.

## Tone for messages to the user

The user will never read the codebase. Don't list file names, internal
scripts, or boilerplate paths. **Never name a metric by its internal
identifier** (e.g. `neighbor_throughput`, `performance_metric_safety`).
Refer to "the candidate performance metric" or describe the stakeholder
perspective from the docstring (e.g. "the traffic-operations stakeholder").

**Never call any behavior, reward, or trajectory "correct" or
"incorrect."** At most, frame things as *possibly* correct/incorrect or
"appears aligned/misaligned with the stakeholder's view." Whether
something is actually right is the user's call — the system surfaces
evidence, not verdicts. The same applies to your overall interpretation:
say "the reward *appears* to prioritize X over Y," not "the reward is
wrong about Y." (See `CLAUDE.md` for the full tone rules.)

## Resolve the workspace

The skill takes the workspace name as `args` (e.g. `/repair-preferences merge`).
If `args` is empty, read `.current_workspace` at the repo root; if that
doesn't exist, tell the user to run `/repair-setup` first.

## Preconditions

- At least one round of evidence has been collected for this workspace.
  If none, tell the user to run `/repair-round` first.

## Procedure

### 1. Pull a candidate pool of disagreements

Don't ask the script for just 5 pairs — it tends to surface the same
loudest disagreement under multiple slots. Instead, pull a larger pool
and choose from it yourself:

```
python find_informative_pairs.py --workspace workspaces/<WORKSPACE> \
       --top-overall 15 --format preference-labels
```

The script applies dedup across (stakeholder, policy_A_class,
policy_B_class) triples, so the 15 candidates already cover a variety
of stakeholders and policy pairings. Read the full output — that's the
pool you'll pick from.

(If the script returned fewer than 5 candidates total, the pool is
genuinely thin; jump to Step 3 and tell the user there's not yet
enough evidence to surface diverse disagreements.)

### 2. **You** pick the 5 most informative, most diverse pairs

Out of the 15 candidates, choose the 5 you'll surface to the user.
Optimize for **diversity** and for **trade-off illumination** — not for
raw rank-product score:

- **Stakeholder coverage.** Try to feature different stakeholder
  perspectives (safety, comfort, fleet, highway-operator, etc.). Avoid
  showing 5 pairs all judged by the same metric.
- **Policy-pair diversity.** Avoid showing the same two policy
  classes compared 3+ times in slightly different rollouts. One or two
  rollouts of any one policy-pair is enough.
- **Trade-off axis.** Prefer pairs where the disagreement highlights a
  *trade-off the reward made*. Read the body of the reward function
  before picking — what does it credit, what does it penalize, and
  which terms are heavier than others? When the reward weighs term X
  heavily and term Y lightly, the most informative disagreements are
  the ones where a stakeholder would have made the *opposite* call.
  Phrase these in the "why" line as: "the reward appears to prioritize
  X over Y; this stakeholder would prioritize Y."
- **Different driver features.** Look at the `Δ (B−A)` columns: pairs
  where *different* features drive the disagreement are more diverse
  than five pairs all driven by the same single delta.
- **Reward-term coverage.** The reward has multiple terms (crash,
  speed, slow penalty, unsafe-distance, jam, etc.). Try to surface
  disagreements that probe different terms.

If the pool itself is degenerate (all 15 candidates collapse onto the
same axis), that is itself a finding — say so in the overall
interpretation in Step 4 and offer to collect more focused evidence.

### 3. Reprint the 5 chosen pairs verbatim, interleaved with per-pair "why" summaries

See the CORE CONTRACT above. This is the load-bearing step.

1. Start with the feature key inside a fenced code block.
2. Then for **each** of the 5 pairs you chose, in your chosen order:
   a. Print the pair's `PAIR …` block **verbatim** inside its own
      fenced code block — header through the feature table. Renumber
      the header from `PAIR k/15` to `PAIR k/5` (or similar) — that's
      the only allowed edit; do not change any other text or numbers.
   b. Immediately after that code block (still in the same response,
      no other content in between), write a `**Why they disagree:**`
      line followed by 1–2 sentences. Look at the largest entries in
      the `Δ (B−A)` column — those are the features driving the
      disagreement. **Frame the explanation as a trade-off the reward
      made.** Phrase as: "the reward *appears* to prioritize
      <reward-term> over <stakeholder-priority>, scoring A higher
      because <feature shift>; this stakeholder would have chosen the
      opposite trade-off because <their priority>."

Do not lump the per-pair "why" summaries into a single section at the
end. Each one goes directly under the pair it explains.

Don't name a metric by its internal id; either say "the candidate
performance metric" or refer to the stakeholder perspective ("the
safety regulator's view", "the traffic-operations view", etc.). Do
not truncate, summarise, or paraphrase any pair. If the script saved
a snapshot path on the last line, you can omit that — it's an internal
detail.

### 4. Overall interpretation — name the trade-off, not the verdict

After all 5 pairs (each followed by its own "why" summary), write a
short (≤ 6 lines) interpretation that:
- names the **pattern** common to the 5 disagreements
- points to the **reward term** or **implicit priority** that drove
  the trade-off (by referencing what the reward credits/penalizes —
  not by file path or variable name)
- suggests 1–2 concrete tweaks the user *could* make to close the gap

**Stay descriptive, never prescriptive.** Don't say "the reward is
wrong about X" or "the reward should be Y." Say "the reward *appears*
to prioritize X over Y — if that wasn't intended, one option is …".
Whether the trade-off is acceptable is the user's call.

### 5. Always close by asking the user what to do next

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
  remind them they can run `/repair-verify` after editing to recheck.
- "Modify the reward for me" → propose specific edits to
  `designed_reward_fn.py` that address the disagreements above, apply
  them with `Edit`, show the diff back to the user, then suggest
  running `/repair-verify`.

## When the user edits the reward and asks you to recheck

Re-run this skill; trajectories on disk are unchanged, so the
disagreement finder simply re-scores them under the new reward.
