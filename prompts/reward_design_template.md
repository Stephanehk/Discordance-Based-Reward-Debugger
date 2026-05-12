# Reward design template

Please fill in the three sections below. Be as concrete as you can — list
specific outcomes, not vague goals. Examples are encouraged.

---

## 1. What outcomes should the agent try to achieve?

List objectives that describe what successful behavior looks like. For
each, provide a description or example of the outcome.

> Example:
> - **Reach the goal location.** Final position is within 0.5 units of the
>   goal coordinate.
> - **Reach the goal quickly.** Fewer timesteps used is better.

(your list here)

---

## 2. What outcomes should the agent avoid?

List outcomes that the agent should avoid. For each, provide a description
or example of the outcome.

> Example:
> - **Collisions with obstacles.** Any timestep where the agent overlaps
>   an obstacle bounding box.
> - **Excessive control effort.** Sustained max-magnitude actions.

(your list here)

---

## 3. How should the agent trade off between different objectives or outcomes?

Describe when it is acceptable to prioritize one objective over another.

> Example:
> - Reaching the goal is more important than minimizing control effort —
>   it's fine to use lots of energy if it means succeeding.
> - But never trade safety for speed: a single collision should outweigh
>   any speed benefit.

(your trade-off rules here)
