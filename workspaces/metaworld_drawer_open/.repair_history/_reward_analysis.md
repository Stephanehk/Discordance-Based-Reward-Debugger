<!-- reward-hash: sha1:aa233acd765a -->

# Reward decomposition — metaworld_drawer_open

`reward(step) = 5 * (T1_caging + T2_opening)`, evaluated on `next_obs` at
**every** step and summed over the 150-step episode. The action is never
read. Max ≈ 10/step ≈ 1500/episode.

## T1 — caging (hand-near-handle)

- **Formula:** `tolerance(|| (handle - hand) * [3,3,1] ||, bounds 0..0.01,
  margin = || (handle_closed - hand_start) * [3,3,1] ||, long_tail)`.
- **Trigger:** fires every step; = 1.0 when the hand is within 1 cm
  (XY-weighted) of the handle, decaying smoothly to ~0.1 at the
  start-of-episode error and further beyond.
- **Sign / scale:** positive, 0 → 5 per step (0 → 750 per episode).
- **Maximized by:** parking the end-effector on the handle and keeping it
  there for all 150 steps. Nothing about the fingers, the gripper command,
  or drawer motion enters this term.
- **Note:** XY error is weighted 3× vs Z, so Z misalignment is cheap.

## T2 — opening (handle-near-fully-open-target)

- **Formula:** `tolerance(|| handle - target ||, bounds 0..0.02,
  margin 0.2 m, long_tail)`.
- **Trigger:** fires every step; = 1.0 once the handle is within 2 cm of
  the fully-open pose, ≈0.13 with the drawer shut (19 cm away).
- **Sign / scale:** positive, ~0.67 → 5 per step (~100 → 750 per episode).
- **Maximized by:** getting the handle to the fully-open pose as **early**
  as possible and leaving it there — the term is paid per step, so total
  credit ≈ (steps spent open) × 5.
- **Note:** it is a *state* term. It never asks who moved the drawer, nor
  whether it is open at the end — only where the handle is right now.

## Exploit hypotheses

- **H1 — T1 proxy decoupling (hover-at-handle).** Park the hand on the
  handle with the fingers wide open and never pull. T1 pays ~5/step for
  150 steps (~750) while the drawer never moves. The reward cannot tell
  "gripping the handle" from "hovering a millimetre off it".

- **H2 — T1/T2 rank inversion (hovering beats a slow success).** A
  never-opens hover run banks ≈750 (T1) + ≈100 (T2 at the shut distance)
  ≈ 850. A run that reaches and opens the drawer only late in the episode
  spends most steps with *both* terms low, and can total **less** than the
  hover run that never opened the drawer at all. Contrast: successful-but-
  slow open vs. never-opens hover.

- **H3 — T1 punishes letting go (release-after-success).** After the
  drawer is fully open, T1 still pays 5/step for keeping the hand glued to
  the handle. A run that opens the drawer and then withdraws the arm —
  clearing the workspace, a natural way to end — scores strictly *lower*
  than one that opens it and stays clamped on the handle, though the end
  states are equally open. Contrast: open-then-retreat vs. open-then-camp.

- **H4 — T2 farming without a durable outcome (open, then undo).** T2 is
  banked per step and the final state is never checked. A run that opens
  the drawer early, dwells there, then pushes it shut again before the
  horizon keeps nearly all its T2 credit and can outscore a run that ends
  fully open but got there later. Contrast: opens-then-closes vs. ends-open.

- **H5 — missing penalty (violence / effort / jitter).** No term reads the
  action, the handle speed, or the command jerk. Yanking the drawer open at
  maximum speed, chattering the commands, and thrashing the arm are all
  free. A smooth open and a violent open with identical positions score
  identically. Contrast: gentle open vs. slam open.

- **H6 — no grasp requirement (hook / shove instead of grip).** The gripper
  action dimension is unrewarded and finger aperture is not read. Opening
  the drawer by hooking or shoving it with fingers splayed scores exactly
  the same as opening it with a proper grip. Contrast: gripped open vs.
  shoved open.

- **H7 — T1 Z-tolerance gap (hover above the handle).** Z error is weighted
  1× against XY's 3×, so the hand can float several centimetres *above* the
  handle, never touching it, and still collect most of T1. Contrast:
  hovering above the handle vs. hand on the handle.

- **H8 — over-run reward past the goal.** T2 saturates within 2 cm of the
  target and the reward never penalizes pulling *past* it. A policy that
  keeps yanking the drawer beyond its fully-open pose (or wrenches the
  handle off-axis) loses nothing, as long as it stays inside the tolerance
  band. Contrast: stop-at-open vs. keep-pulling.
