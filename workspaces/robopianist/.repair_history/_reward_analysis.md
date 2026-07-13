<!-- reward-hash: sha1:c597aa3cdfee -->

# Reward decomposition — RoboPianist (Twinkle Twinkle, two Shadow hands)

The designed reward is the **unweighted sum of five per-step terms**, scored
from the post-step state (`next_obs`). Per-step range is roughly `[~ -0.1, 3]`.

## Terms

### T1 — key_press  (0 … 1, positive)
Two halves, each worth 0.5:
- **press half** `0.5 * mean_over_goal_keys( tolerance(goal - depth, band 0.05) )`
  — fires when the keys the song wants *now* are pushed to full depth. Gaussian
  shaping gives **partial credit for partial depth**. If no key is required this
  step, this half is 0.
- **no-false-press half** `0.5 * (1 - [any off-goal key is activated])` — a
  **binary** flag: full 0.5 whenever *zero* wrong keys are down, 0 the instant
  *any* wrong key is down, regardless of how many.
- Maximized by: pressing exactly the required keys to full depth and nothing
  else. But the no-false-press half alone is maximized by **pressing nothing**.

### T2 — sustain  (0 … 1, positive)
`tolerance(sustain_target - sustain_activation, band 0.05)`. Fires when the pedal
state matches the song's per-step sustain target. Twinkle Twinkle's debug track
rarely calls for sustain, so the target is ~0 most steps → **holding the pedal
up (off) already scores near 1 almost every step.**

### T3 — energy  (≤ 0, penalty)
`-0.005 * (right_hand_power + left_hand_power)`. Small per-step penalty on total
actuator mechanical power. Coefficient is tiny (5e-3) → **weak**; only extreme
thrashing makes it material. Maximized (toward 0) by not moving.

### T4 — fingering  (0 … 1, positive)
`mean_over_active_fingers( tolerance(||fingertip - assigned_key_point||, band
0.01 m) )`. Rewards each assigned fingertip being within ~1 cm of its target
key's contact point. **This is proximity, not pressing** — a fingertip hovering
1 cm above a key scores full credit without the key ever sounding.

### T5 — forearm  (0 or 0.5, positive)
`0.5 * [forearms are NOT colliding]`. Binary: 0.5 for clear forearms, 0 while
they touch. Maximized by keeping the two hands apart — trivially satisfied by
**staying off the keyboard**.

## Net observation
A hand held **still in a neutral hover** over the keys collects: T1 press ≈ 0,
T1 no-false-press = 0.5, T2 ≈ 1, T3 ≈ 0, T4 = up to 1 (if fingertips sit near
their assigned keys), T5 = 0.5 → **≈ 2–3 per step while producing no music at
all.** A genuine performance has to *additionally* earn the T1 press half but
risks losing the binary no-false-press 0.5 and the collision 0.5, and spends
energy. The reward therefore only weakly separates "plays the song" from "does
nothing gracefully."

## Exploit hypotheses

- **H1 — key_press / rank inversion (T1, T4, T5).** A still neutral hover that
  plays *zero* notes (musical F1 ≈ 0) scores near a real performer because it
  banks the no-false-press 0.5, the sustain ≈1, the forearm 0.5, and (if
  fingertips rest near their keys) the fingering proximity — the reward
  under-credits actually playing and over-credits doing nothing.

- **H2 — no-false-press binary granularity (T1).** Pressing *one* wrong key and
  pressing *thirty* wrong keys both zero out the same single 0.5 term — the
  reward cannot distinguish a tiny slip from mashing the whole keyboard. Drive
  many simultaneous false presses; the penalty is identical to one.

- **H3 — fingering proximity proxy (T4).** Hover assigned fingertips within ~1 cm
  of their keys without ever depressing them: T4 → high while the keys never
  sound (activation stays 0), decoupling the proximity proxy from real presses.

- **H4 — sustain decoupling (T2).** Hold the sustain pedal fully down every step
  regardless of the song: the pedal is used independently of what the music
  calls for, and on a track that rarely sustains this *costs* reward where a
  human would expect pedaling to be neutral or helpful.

- **H5 — weak energy / unpenalized jerk (T3).** The 5e-3 power penalty is far too
  small to discourage violent, high-jerk thrashing; a policy can oscillate
  actuators to their extremes every step (utterly unplayable, unsafe motion)
  while the reward barely registers it.

- **H6 — forearm binary / avoid-by-abstaining (T5).** The collision term is a
  flat 0.5 with no severity scaling, and is trivially maxed by never bringing
  the hands together — i.e. by not playing. Conversely a policy that must cross
  hands to play the song is penalized the same 0.5 for a light brush as for a
  hard clash.
