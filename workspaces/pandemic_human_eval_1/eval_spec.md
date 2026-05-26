# Reward design task

You are designing a reward function for the public-health authority in this
pandemic-control environment. Each simulated day you choose a regulation
stage in 0..4 (0 = no restrictions, 4 = strictest lockdown). Your reward
function must encode the criteria below.

## Criteria

1. When the critical proportion is less than 0.01, lockdown regulation
   should be zero. Higher critical rates are always worse than lower
   critical rates.

2. Spend as little time in lockdown stage 4 as possible.

3. Avoid rapidly changing lockdown levels; a change in critical rates of
   0.001 but a constant lockdown stage is preferable to an increasing or
   decreasing lockdown stage but no change in critical rates. (Keeping
   the lockdown stage from fluctuating is worth a small increase in
   critical cases.)

4. For a single step, a change in critical cases of 0.01 and no change in
   lockdown regulation is equally as desirable as a change in critical
   cases of 0.005 and an increase from lockdown regulation 0 to 1.
   (Keeping critical rates low is more important than keeping lockdown
   rates low.)

## Starting point

You will be shown an initial reward function that does **not** yet satisfy
these criteria. Your job, using the `/repair-*` skills, is to surface its
flaws and revise it until it does.
