<!-- reward-hash: sha1:59bc51e06ae4 -->

# Reward decomposition — rocket landing (PyFlyt dense reward)

User's stated goal: fly the falling booster down to a **soft, upright
touchdown on the pad** at the origin.

## Terms

**T1 — loitering offset.** `-0.3` every step. Always fires. Constant
negative. Minimized (least harm) by ending the episode quickly.

**T2 — stay-over-pad.** `+0.3 / (horiz_dist + 0.1)`. Fires every step;
grows as the rocket's horizontal offset from the pad shrinks. Caps near
`+3.0/step` when directly above the pad (dist≈0). Maximized by sitting
horizontally over the pad — **regardless of altitude or whether it ever
lands.**

**T3 — lateral progress.** `+10 * (prev_horiz_dist − horiz_dist)`.
Telescopes over the episode to `10 * (initial − final) horizontal
distance`. ~10 reward per meter of net horizontal approach. Maximized by
ending up horizontally over the pad.

**T4 — vertical progress (descend).** `+0.2 * (prev_height − height)`.
Telescopes to `0.2 * total altitude lost` ≈ `+90` over a 450 m drop.
Rewards losing altitude — **faster/more descent pays more, with no cap
and no dependence on how it's done.**

**T5 — deceleration near ground.** `+4 * (Δvertical_speed + 1)/exp(height)
* sign`, sign `+1` while descending, `−1` while ascending. The
`1/exp(height)` factor makes it ~0 at altitude and dominant within a few
meters of the pad. Rewards slowing the descent right before touchdown;
penalizes any late ascent (bounce/abort).

**T6 — spin penalty.** `−|yaw rate|`. Every step. Small (a few units at
most). Minimized by not spinning.

**T7 — tilt penalty.** `−‖roll, pitch‖`. Every step. Bounded ~`−π`.
Minimized by staying upright. **Small relative to T3's +10/m**, so the
reward will trade a lot of tilt for horizontal approach.

**T8 — pad-contact bonus.** `+5 − 0.3*|vertical speed|` the step contact
occurs. Flat `+5` for **any** contact; the speed penalty is tiny
(a 10 m/s slam only subtracts 3, still net `+2`). Fires for a crash onto
the pad just as much as for a gentle landing.

**T9 — clean-touchdown bonus.** extra `+3` if, at contact, the *previous*
step had `‖ang_vel‖<0.02`, `‖lin_vel‖<0.02`, and tilt `<0.1`, and it is
not "fatal". An extremely tight gate — near-zero velocity — so in
practice almost never awarded; the reward barely distinguishes a 0.3 m/s
touchdown from a 3 m/s one (both miss T9, both keep ~all of T8).

## Exploit hypotheses

**H1 — T2 proxy decoupling (stay-over-pad without landing).** Burn thrust
to slow the fall and steer to sit horizontally over the pad, accumulating
T2 (up to +2.7/step net) while **never touching down**. Proxy
(over-pad) high, true outcome (landing) never happens.
Contrast: `frac_hover_over_pad` high, `touched_pad`=0.

**H2 — T8 rank inversion (crash onto pad out-scores a careful non-landing).**
The flat `+5` contact bonus makes slamming the pad at high speed net
strongly positive. A hard crash **on** the pad can out-total a controlled
descent that ends **near** the pad without contact. Proxy=contact, true
outcome=soft landing. Contrast: `touched_pad`=1, `impact_vspeed` high,
`clean_landing`=0, out-scoring a low-`impact` non-contact trajectory.

**H3 — T4 farming altitude loss (fast plummet pays).** `+0.2 * altitude
lost` with no cap rewards sheer descent; a near-uncontrolled nosedive
racks up ~+90 of T4 plus the +5 contact bonus. Behavior the user would
call a crash scores well. Contrast: `mean_descent_rate` very high,
`impact_vspeed` high.

**H4 — T3 vs T7 trade-off (tilt hard to translate).** +10/m lateral
progress dwarfs the ~−1/step tilt penalty, so the reward endorses a
wildly tilted approach to get over the pad. A thoughtful person may
reject trading that much attitude stability for horizontal position.
Contrast: `mean_tilt`/`max_tilt` high, `final_horiz_dist` low.

**H5 — T9 gate too tight (no softness gradient).** Because T9's near-zero-
velocity gate is essentially unreachable, two touchdowns of very
different gentleness (e.g. 0.3 m/s vs 3 m/s) receive nearly identical
reward. The reward fails to prefer the genuinely softer landing.
Contrast: two contact trajectories with very different `impact_vspeed`
but nearly equal designed return.

**H6 — missing penalty (fuel / control jerk).** No term penalizes fuel
burn or jerky, high-magnitude control thrashing. A policy that thrashes
all actuators or wastes fuel is un-penalized as long as it makes progress.
Spec gap. Contrast: high `engine_on_frac` / large action deltas with no
reward cost.
