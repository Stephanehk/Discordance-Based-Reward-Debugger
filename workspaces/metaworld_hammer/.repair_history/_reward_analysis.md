<!-- reward-hash: sha1:a1528b9d5353 -->

# Reward decomposition — metaworld_hammer

`reward_t = (2 * reward_grab + 6 * reward_in_place) * reward_quat`, evaluated
fresh at every step and summed over the 150-step episode. Overridden to a flat
`10.0` on any step where `nail_slide > 0.09` **and** the shaped sum is already
`> 5.0`.

Episode geometry (seed 0, representative): hammer starts on the table at
`[0.01, 0.44, 0.00]`; the nail head sticks out at `[0.24, 0.64, 0.11]`; the
target — the pose the nail head occupies once fully driven — is
`[0.24, 0.74, 0.11]`. So driving the nail means pushing it 0.10 m in `+y`.
The reward's model of the hammer head is `hammer_body + [0.16, 0.06, 0.0]`,
a **fixed offset that ignores the hammer's rotation**.

## Terms

### T1 — `reward_grab` (weight 2, range [0, 2])
- **Fires when:** the two finger pads bracket the hammer handle in `y` (each pad
  within 0.015–0.02 m of it), the tool-center-point closes in on the handle in
  `x-z` (within 0.01 m), and the gripper command `act[3]` is positive. The
  high-density variant returns `(caging_and_gripping + caging) / 2`, so **pure
  caging with no squeeze still pays half**.
- **Sign / scale:** positive, up to 2.0 per step (300 over an episode).
- **Maximized by:** parking the fingers around the handle and squeezing. It
  checks *finger pose and gripper command only* — never that the hammer left the
  table, moved, or was used.

### T2 — `reward_in_place` (weight 6, range [0, 6]) — the dominant term
- **Fires when:** `0.1 * (computed_head_z > 0.02)` plus `0.9 *
  tolerance(||computed_head − target||, bounds=(0, 0.02), margin=0.2,
  long_tail)`.
- **Sign / scale:** positive, up to 6.0 per step (900 over an episode). The
  long-tail tolerance is **extremely generous**: at 0.10 m from the target it
  still returns ~0.41; at 0.05 m, ~0.83.
- **Maximized by:** holding the hammer head as close to the *driven-in* nail pose
  as possible, **for as many steps as possible**. It pays for a static pose, not
  for the nail moving. Concretely: hammer head pressed against a half-driven nail
  (~0.05 m short of target) yields ~5.1 from this term + ~2.0 from T1 ≈ **7.1 per
  step**, and holding that for 150 steps totals ≈ 1065 — **more than the ≈ 829
  a scripted expert earns by fully driving the nail**, because the 10/step
  override only starts at the (late) moment of success.

### T3 — `reward_quat` (multiplier, range [0, 1])
- **Fires when:** the hammer's quaternion is near `[1, 0, 0, 0]` (lying flat);
  `max(1 − ||q − ideal|| / 0.4, 0)`, reaching 0 at roughly a 46° tilt.
- **Sign / scale:** multiplicative gate on the entire shaped sum.
- **Maximized by:** keeping the hammer level. Note it is **1.0 for a hammer lying
  untouched on the table** — it credits an idle scene as much as a perfectly
  wielded tool.

### T4 — success override (flat 10, gated)
- **Fires when:** `nail_slide > 0.09` **and** the shaped sum `> 5.0`.
- **Sign / scale:** replaces the shaped sum with 10.0 per step, i.e. only ~25%
  above the shaped ceiling of 8.
- **Maximized by:** driving the nail in **early** and then continuing to hold the
  hammer hard against it, because the gate re-evaluates every step: let go of the
  hammer after finishing and the shaped sum collapses below 5, and the 10/step
  stops.

## Exploit hypotheses

- **H1 — Grasp proxy without the tool ever being wielded (T1).** Cage the handle
  on the table and squeeze, never lifting the hammer. Banks up to 2.0/step for an
  episode in which nothing happens. Proxy = finger pose + gripper command; true
  outcome = the hammer actually being lifted and used.

- **H2 — Rank inversion by under-driving the nail (T2 vs T4).** Drive the nail
  only *partway* (~0.04–0.06 m, deliberately staying below the 0.09 success line)
  and hold the hammer head pressed there for the rest of the episode. Estimated
  ≈ 7/step × 150 ≈ 1065, **beating a full, genuine success** (≈ 829). The reward
  strictly prefers the trajectory that refuses to finish the task.

- **H3 — Dense farming with zero task progress (T2).** Grasp the hammer, carry
  its head up against the *un-driven* nail head (0.10 m from target) and hold.
  Yields ≈ 4.8/step ≈ 720 — the large majority of the expert's return — while the
  nail never moves at all.

- **H4 — Punished for putting the tool down after succeeding (T4 gate).** Fully
  drive the nail, then release the hammer and retract the arm to a neutral pose.
  Because the 10/step override is re-gated on `shaped > 5` every step, cleanly
  disengaging after success collapses the reward to near zero, while mashing the
  hammer into the already-driven nail keeps paying 10/step.

- **H5 — Genuine success with the wrong tool earns nothing (T1/T2/T4).** Push the
  nail all the way in with the gripper/arm directly, leaving the hammer on the
  table. The user's actual goal (nail driven into the wall) is fully achieved, but
  T1 and T2 never fire and the override is gated behind `shaped > 5`, so the
  reward is ≈ 0. The reward pays for *holding a hammer near a spot*, not for the
  nail going in.

- **H6 — No penalty for violence, effort, or jitter (missing penalty).** Slam the
  hammer at maximum command magnitude with hard repeated strikes and heavy
  jitter. Nothing in the reward penalizes control effort, impact force, or
  destructive side effects, so this scores at least as well as a gentle,
  controlled drive.

- **H7 — Orientation-blind head model (T2 vs T3).** The head position is a fixed
  offset from the hammer body, ignoring rotation, and T3 tolerates tilts up to
  ~46°. A hammer carried rotated can place the *computed* head near the target
  while the real head points elsewhere — and, conversely, the flatness multiplier
  is at its maximum for a hammer nobody has touched.

- **H8 — The gate can deny credit for a real success (T4).** Sink the nail with a
  fast whack that leaves the grip loose or the hammer tilted, so the shaped sum is
  below 5 on the step the nail passes 0.09. The nail is genuinely driven in but
  the 10/step override never triggers — success goes uncredited.
