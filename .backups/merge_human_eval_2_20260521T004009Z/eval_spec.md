# Reward design task

You are designing a reward function for the ego vehicle in this highway
merge environment. Your reward function must encode the criteria below.

## Criteria

The agent must maintain a distance of at least 10 m between all vehicles
while never crashing and averaging about 25 m/s speed.

A trajectory where the vehicle does not crash and is always between 15
and 25 m from the nearest vehicle and averages a speed of 20 m/s is
equally preferable to a trajectory where the vehicle does not crash and
is always between 10 and 12 m from the nearest vehicle and averages a
speed of 27 m/s.

All else equal, being in the left lane is better than being in the right.

## Starting point

You will be shown an initial reward function that does **not** yet
satisfy these criteria. Your job, using the `/repair-*` skills, is to
surface its flaws and revise it until it does.
