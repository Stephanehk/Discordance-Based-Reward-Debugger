<!-- reward-hash: sha1:a57ea09195fb -->

# Reward decomposition — metaworld_drawer_open (telescoping endpoint reward)

INTERNAL. Never shown to the user.

The returned value per step is

    r = OPEN_WEIGHT * (openness(next) - openness(obs))
        + CAGE_WEIGHT * (cage(next) - cage(obs))

with `OPEN_WEIGHT = 10.0`, `CAGE_WEIGHT = 0.5`. Both terms are per-step
*differences* of potentials, so summed over an episode they telescope to a
function of the endpoints only:

    return(tau) = OPEN_WEIGHT * (openness(final) - openness(initial))
                + CAGE_WEIGHT * (cage(final) - cage(initial))

Since the drawer starts shut (openness(initial) ~= 0), the return is essentially
`10 * openness(final) + 0.5 * (cage(final) - cage(initial))`.

## Terms

### T1 — Opening term
- **Form:** `OPEN_WEIGHT * (openness(next) - openness(obs))`, telescopes to
  `10 * openness(final)` (initial ~= 0).
- **openness(o):** `clip(1 - ||handle - target|| / 0.2, 0, 1)`; handle = obs[4:7],
  target (fully-open pose) = obs[39:42]. 0 = shut, 1 = fully open.
- **Sign/scale:** dominant term, 0 .. +10 over an episode.
- **Maximized by:** ending the episode with the handle exactly at the fully-open
  target, regardless of *how* or *when* it got there — path, timing, effort, and
  whether the drawer was ever released are all invisible; only the final handle
  position is read.

### T2 — Caging term (bounded guidance)
- **Form:** `CAGE_WEIGHT * (cage(next) - cage(obs))`, telescopes to
  `0.5 * (cage(final) - cage(initial))`.
- **cage(o):** XY-emphasized hand-to-handle proximity via dm_control `tolerance`,
  0 (far) .. 1 (hand at handle); normalized by the hand-to-handle error at reset.
- **Sign/scale:** bounded to +/- 0.5 over a whole episode — an order of magnitude
  smaller than T1 and unable to outweigh it.
- **Maximized by:** ending with the hand on the handle (cage(final) ~= 1) having
  started far from it. Only the *net change* pays, so hovering/dwelling at the
  handle earns nothing — the potential-shaping fixes the classic idle exploit.

### Not scored at all
- The **action** (movement magnitude, jerk, finger chatter) — nothing reads it.
- **Timing / dwell** — when the drawer opens, and how long it stays open, do not
  matter; only the final handle position.
- **How the handle is moved** — a proper finger grasp vs. shoving the drawer with
  the back of the hand are indistinguishable to the reward.

## Exploit hypotheses

- **H1 — T1/action, effort & smoothness are free.** Because no term reads the
  action, a violent, jerky, full-magnitude yank that ends the drawer fully open
  earns the *same* ~10 as a smooth, gentle open. Instantiate: a slam-open policy
  (max-magnitude, jittery, finger-chatter) that still finishes open, contrasted
  with a gentle grip-and-pull that finishes open. Decouples final openness (proxy
  for "drawer opened well") from motion quality.
- **H2 — T1 endpoint-only, timing & dwell are free.** The telescoped return reads
  only the final openness, so opening the drawer in the *last few steps* (after
  idling most of the episode) scores identically to opening it immediately and
  holding it open the whole time; likewise a drawer that is open only at the final
  instant scores like one open throughout. Instantiate: a dawdle-then-open-late
  policy that reaches full openness only at the end. Decouples final openness from
  speed and from sustained-open.
- **H3 — T1 release/creep penalty vs. task achieved.** T1 reads the *final* handle
  position and a released drawer creeps back a few cm, so a policy that opens the
  drawer and then withdraws the arm loses a slice of the opening credit relative to
  one that stays clamped on the handle to the end — even though both *did open* the
  drawer. Instantiate: open-then-retreat vs. the clamped maximizer. Surfaces a
  reasonable-disagreement / rank case (letting go after opening is arguably fine).
- **H4 — T1 proxy, no grasp required.** T1 credits only the handle's final
  position and T2 (grasp guidance) is bounded to 0.5, so a policy can shove the
  drawer open with the back of an open hand — never grasping the handle — and
  still collect the full opening credit. Instantiate: a no-grasp shove-open policy
  contrasted with a proper grip-and-open. Decouples "handle reached target" from
  "handle was grasped and drawn open."
