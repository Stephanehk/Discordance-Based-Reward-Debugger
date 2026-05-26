# Reward design task

You are designing a reward function for the public-health authority in this
pandemic-control environment. Each simulated day you choose a regulation
stage in 0..4 (0 = no restrictions, 4 = strictest lockdown). Your reward
function must encode the criteria below.

## Criteria

1. When the critical-case proportion is below 0.01, the regulation stage
   should be 0. As the critical-case proportion rises, the regulation stage
   should rise in proportion.

2. Spend as little time as possible at the strictest lockdown (stage 4).

3. Avoid rapidly changing the regulation stage. A trajectory whose
   critical-case rate changes by 0.001 while the regulation stage stays
   constant is preferable to one whose regulation stage rises or falls while
   the critical-case rate does not change at all.

4. Over a single step, a change in critical cases of 0.01 with no change in
   regulation stage is exactly as desirable as a change in critical cases of
   0.005 together with an increase in regulation stage from 0 to 1.

## Starting point

You will be shown an initial reward function that does **not** yet satisfy
these criteria. Your job, using the `/repair-*` skills, is to surface its
flaws and revise it until it does.
