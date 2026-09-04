"""Designed reward for Meta-World hammer-v3.

A Sawyer robot arm must grasp a hammer and use it to drive a nail into a wall.
The episode runs for a fixed horizon of 150 steps.

This reward pays for the nail going into the wall. The dense terms are written
so that they cannot be collected by holding a pose: nail credit is paid on the
*increment* of nail travel, so it telescopes to the final depth no matter how
long the arm lingers, and the reach/carry guidance is potential-based, so its
episode total depends only on the first and last state. What remains that an
agent can actually increase is finishing the nail, finishing it sooner, and
doing so without thrashing.

Observation (58-dim Box). The reward only ever receives (obs, act, next_obs):
  [0:3]   hand (end-effector) position (x, y, z), meters
  [3]     gripper aperture (open > 0)
  [4:7]   hammer position (x, y, z), meters
  [7:11]  hammer orientation quaternion (w, x, y, z)
  [11:14] nail-head position (x, y, z), meters
  [14:18] nail orientation quaternion (w, x, y, z)
  [18:36] previous timestep's copy of indices [0:18]
  [36:39] task goal position (driven-in nail-head target), meters
  [39:42] target position used by the reward (driven nail-head pose), meters
  [42:45] hand position at the start of the episode (x, y, z), meters
  [45:48] tool-center-point (between the fingers) position (x, y, z), meters
  [48:51] left gripper pad position (x, y, z), meters
  [51:54] right gripper pad position (x, y, z), meters
  [54:57] hammer position at the start of the episode (x, y, z), meters
  [57]    nail slide displacement (meters); 0 when flush, grows as the nail is
          driven in

Action (4-dim Box, low=-1, high=1):
  [0] delta x of the end-effector
  [1] delta y of the end-effector
  [2] delta z of the end-effector
  [3] gripper command (>0 closes the fingers, <0 opens them)
"""

import numpy as np

HEAD_OFFSET = np.array([0.16, 0.06, 0.0])  # meters; hammer head relative to hammer body
NAIL_TARGET_DEPTH = 0.09  # meters of nail travel that counts as driven in

W_PROGRESS = 8.0  # paid out across the nail's travel, in full once it is driven in
W_SUCCESS = 8.0  # one-off, on the step the nail crosses the driven-in line
C_TIME = 0.02  # per step, charged until the nail is in
C_EFFORT = 0.005  # per step, on the squared arm command
W_REACH = 0.2  # shaping potential: fingers to the hammer handle
W_CARRY = 0.6  # shaping potential: hammer head to the driven-in nail pose


def _nail_depth(state):
    """How far the nail has travelled into the wall, in meters."""
    return float(state[57])


def _is_driven(state):
    """Whether the nail is all the way into the wall."""
    return _nail_depth(state) >= NAIL_TARGET_DEPTH


def _potential(state):
    """Guidance potential: 0 once the nail is in, else how far the arm is from driving it.

    Applied as a difference between successive states, so an episode's total
    guidance depends only on where it starts and ends -- lingering in a
    well-placed pose accumulates nothing.
    """
    if _is_driven(state):
        return 0.0
    hammer = state[4:7]
    to_handle = float(np.linalg.norm(state[45:48] - hammer))
    to_nail = float(np.linalg.norm((hammer + HEAD_OFFSET) - state[39:42]))
    return -(W_REACH * to_handle + W_CARRY * to_nail)


def designed_reward_fn(obs, act, next_obs) -> float:
    """Reward one transition: nail progress, completion, promptness, effort, guidance."""
    state, next_state = np.asarray(obs, dtype=np.float64), np.asarray(next_obs, dtype=np.float64)
    action = np.asarray(act, dtype=np.float64)

    # Nail travel, credited on the increment and capped at the driven-in depth:
    # over an episode this sums to how far the nail actually went in, whatever
    # route the arm took and however long it held its pose afterwards.
    depth = min(_nail_depth(state), NAIL_TARGET_DEPTH)
    next_depth = min(_nail_depth(next_state), NAIL_TARGET_DEPTH)
    reward = W_PROGRESS * (next_depth - depth) / NAIL_TARGET_DEPTH

    # Driving the nail home, paid once, on the step it goes in.
    if _is_driven(next_state) and not _is_driven(state):
        reward += W_SUCCESS

    # Until then, every step costs something.
    if not _is_driven(next_state):
        reward -= C_TIME

    # Wrenching the arm around is never free.
    reward -= C_EFFORT * float(np.mean(np.square(action[:3])))

    # Dense guidance toward grasping the hammer and bringing its head to the nail.
    reward += _potential(next_state) - _potential(state)

    return float(reward)
