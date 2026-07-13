# Prompt: decompose the designed reward and hypothesize its flaws

> **Task:** Read the designed reward function and produce an internal
> term-by-term decomposition of it, plus a numbered list of hypotheses
> for how optimizing it could diverge from what the user actually wants.
> This analysis is what the later stages use to target their probes and
> their comparison selection at the reward's *suspected* flaws.

This is an **internal** artifact — never shown to the user, never posted
to the web UI. Do not use the word "exploit", term names, or hypothesis
ids in anything the user sees; the web-UI tone rules still apply.

## Context to use

- The filled reward-design template (the comment block at the top of
  `designed_reward_fn.py`) — the outcomes the user said they care about.
- The body of `designed_reward_fn.py` — what the reward *actually*
  credits and penalizes, term by term.

## Output

Write `.repair_history/_reward_analysis.md` (workspace-relative). It must
contain, in order:

1. **A hash stamp as the first line**, so later stages can tell when the
   analysis is stale relative to the current reward:

   ```
   <!-- reward-hash: sha1:xxxxxxxxxxxx -->
   ```

   Compute the hash from the repo root (`llm` active):

   ```bash
   python -c "import hashlib;print('sha1:'+hashlib.sha1(open('workspaces/<name>/designed_reward_fn.py','rb').read()).hexdigest()[:12])"
   ```

2. **One entry per reward term** — each additive or multiplicative piece
   of the returned value. For each term give:
   - its name,
   - its trigger condition (when it fires),
   - its sign and rough scale / magnitude,
   - the behavior that maximizes it.

3. **A numbered list of exploit hypotheses** `H1, H2, …` — concrete ways
   optimizing the reward could diverge from the user's true intent.
   Each hypothesis names the term it concerns and, in one phrase, the
   behavior that would expose the flaw. Generate them by working through
   the lenses in "How to hypothesize flaws" below. Generate between 2-8 hypotheses.

## How to hypothesize flaws

Work through these lenses on the decomposed reward; each tends to surface
a different kind of flaw. You should think about all of the flaws.

- **Rank inversion — a more successful trajectory scoring lower than a
  less successful one.** Look for a way the reward would total *higher*
  for a trajectory that does worse at the user's actual goal than for one
  that does better. This is the most decisive kind of flaw: name the
  successful behavior the reward under-credits and the inferior behavior
  it over-credits, and the term responsible.
- **Goodharting / proxy decoupling.** Wherever a term rewards a *proxy*
  for the goal (a distance, a contact, a position, a count) rather than
  the goal itself, hypothesize the behavior that drives the proxy term
  high while the true outcome never happens. Maximizing the measure stops
  tracking the thing it was meant to measure.
- **Farming a dense term.** A term that pays out every step (or
  repeatedly) can be accumulated by behavior the user never envisioned —
  loitering where it pays, oscillating to re-trigger it, stalling instead
  of finishing. Hypothesize the loop that racks it up without progress.
- **Non-obvious trade-offs between terms.** Where two terms pull in
  different directions, work out the exchange rate the reward implies
  (how much of term A it will sacrifice for a unit of term B). Flag any
  rate a reasonable person, told it explicitly, would likely reject.
- **Reasonable-disagreement cases.** Independent of any single term, look
  for situations where the reward expresses a definite preference between
  two trajectories that a thoughtful person could plausibly rank the
  other way.
- **Missing penalty / specification gap.** Outcomes the user would
  clearly want *avoided* but that no term penalizes: side effects,
  norm-violations, jitter, near-misses with a bad state. Hypothesize a
  trajectory that achieves a listed goal while doing one of these
  un-penalized bad things.
- **Untested regions.** Parts of the state/action space the reward's
  behavior is simply unspecified or extrapolated into. Hypothesize where
  it might be silently wrong because nothing pins it down there.

## Guidance

- Enumerate terms from the code body, not from the template — the
  template says what the user *wanted*; the body says what the reward
  *does*. Hypotheses live in the gap between the two.
- Prefer hypotheses you could actually instantiate with a policy: "drive
  term X high while outcome Y never happens" is more useful than a vague
  worry.
- Each hypothesis should name the term(s) involved and, where it applies,
  point at the contrast that would demonstrate it (the successful-but-
  lower-ranked trajectory, the proxy-vs-outcome pair, the questionable
  trade-off). That contrast is what `/repair-round` tries to produce and
  `/repair-preferences` tries to surface.
- The hypothesis ids `H1, H2, …` are referenced by later stages, so keep
  them stable within a given reward version. Regenerate the whole file
  (new ids included) when the reward changes.
