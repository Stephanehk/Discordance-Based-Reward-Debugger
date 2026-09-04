"""Designed reward for the robosuite Wipe task (Panda arm, OSC_POSE control).

The agent controls a Panda arm holding a wiping pad and must clean small
"dirt" markers off a table over a 500-step episode.

------------------------------------------------------------------------
OBSERVATION — 1-D float32 vector of length (proprio_dim + 10)
------------------------------------------------------------------------
The leading `proprio_dim` entries are robosuite's flattened Wipe observation
(condensed dirt-field summary + Panda proprioception). The trailing 10 channels
carry per-step reward components; index them off the END of the vector:

  next_obs[-10] = total_wiped         - cumulative # markers wiped so far
  next_obs[ -9] = new_wiped           - # markers wiped on this single step
  next_obs[ -8] = mean_distance       - mean distance (m) from the wiper to the
                                        remaining dirt
  next_obs[ -7] = has_gripper_contact - 1.0 iff the wiper is touching the table
  next_obs[ -6] = ncon                - # of active MuJoCo contacts
  next_obs[ -5] = total_force_ee      - max end-effector force magnitude (N)
  next_obs[ -4] = mean_ee_acc         - mean |end-effector acceleration|
  next_obs[ -3] = collision_or_qlim   - 1.0 if the arm collided with the table
                                        OR hit a joint limit this step
  next_obs[ -2] = num_markers         - constant: total dirt markers spawned
  next_obs[ -1] = horizon             - constant: episode horizon (500 steps)

------------------------------------------------------------------------
ACTION — Box(low=-1.0, high=1.0, shape=(6,), float32)
------------------------------------------------------------------------
OSC_POSE end-effector control (no gripper DoF — the wiping pad is fixed):
  act[0:3] - end-effector position delta (dx, dy, dz), scaled to ±0.05 m/axis
  act[3:6] - end-effector orientation delta as an axis-angle rotation
             (rx, ry, rz), scaled to ±0.5 rad/axis

Fill in the body. Signature must be unchanged.
"""


def designed_reward_fn(obs, act, next_obs) -> float:
    raise NotImplementedError("Fill in designed_reward_fn before running /repair-init.")
