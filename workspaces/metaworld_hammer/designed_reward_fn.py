"""Designed reward for Meta-World hammer-v3.

The agent controls a Sawyer robot arm that must grasp a hammer and use it to
drive a nail into a wall.

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

Fill in the body. The signature must be unchanged.
"""


"""Designed reward for hammer.

Rewards (a) the gripper grasping/caging the hammer and (b) the hammer head
reaching the nail target, modulated by how well the hammer is oriented; on task
success (nail driven in) the reward is set to its maximum, provided the shaped
reward is already high. Function of the resulting state `next_obs` and the action
that produced it (see env_config.OBS_DESCRIPTION for the layout).
"""

import numpy as np

_HAMMER_HANDLE_LENGTH = 0.14


def _tolerance(x, lower, upper, margin, value_at_margin=0.1):
    if lower <= x <= upper:
        return 1.0
    if margin <= 0:
        return 0.0
    d = (lower - x if x < lower else x - upper) / margin
    scale = (1.0 / value_at_margin - 1.0) ** 0.5
    return 1.0 / ((d * scale) ** 2 + 1.0)


def _hamacher(a, b):
    denom = a + b - a * b
    return (a * b) / denom if denom > 0 else 0.0


def _grasp_reward(action, obj_pos, left_pad, right_pad, tcp, init_tcp, obj_init,
                  obj_radius=0.015, pad_success_thresh=0.02, xz_thresh=0.01):
    pad_y = np.array([left_pad[1], right_pad[1]])
    pad_to_obj = np.abs(pad_y - obj_pos[1])
    pad_to_objinit = np.abs(pad_y - obj_init[1])
    caging_margin = np.abs(pad_to_objinit - pad_success_thresh)
    caging_lr = [
        _tolerance(float(pad_to_obj[i]), obj_radius, pad_success_thresh, float(caging_margin[i]))
        for i in range(2)
    ]
    caging_y = _hamacher(caging_lr[0], caging_lr[1])

    xz = [0, 2]
    caging_xz_margin = float(np.linalg.norm(obj_init[xz] - init_tcp[xz])) - xz_thresh
    caging_xz = _tolerance(
        float(np.linalg.norm(tcp[xz] - obj_pos[xz])), 0.0, xz_thresh, caging_xz_margin
    )

    gripper_closed = min(max(0.0, float(action[-1])), 1.0)
    caging = _hamacher(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = _hamacher(caging, gripping)
    # high-density variant
    return (caging_and_gripping + caging) / 2.0


def _quat_reward(quat):
    ideal = np.array([1.0, 0.0, 0.0, 0.0])
    error = float(np.linalg.norm(quat - ideal))
    return max(1.0 - error / 0.4, 0.0)


def _in_place_reward(hammer_head, target):
    pos_error = target - hammer_head
    a, b = 0.1, 0.9
    lifted = hammer_head[2] > 0.02
    return a * float(lifted) + b * _tolerance(float(np.linalg.norm(pos_error)), 0.0, 0.02, 0.2)


def designed_reward_fn(obs, act, next_obs):
    s = np.asarray(next_obs, dtype=np.float64)

    hand = s[0:3]
    hammer = s[4:7]
    quat = s[7:11]
    target = s[39:42]
    init_tcp = s[42:45]
    tcp = s[45:48]
    left_pad = s[48:51]
    right_pad = s[51:54]
    obj_init = s[54:57]
    nail = float(s[57])

    hammer_head = hammer + np.array([0.16, 0.06, 0.0])
    hammer_threshed = hammer.copy()
    if abs(hammer[0] - hand[0]) < _HAMMER_HANDLE_LENGTH / 2.0:
        hammer_threshed[0] = hand[0]

    reward_quat = _quat_reward(quat)
    reward_grab = _grasp_reward(act, hammer_threshed, left_pad, right_pad, tcp, init_tcp, obj_init)
    reward_in_place = _in_place_reward(hammer_head, target)

    shaped = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
    success = nail > 0.09
    if success and shaped > 5.0:
        shaped = 10.0
    # Dense reward for the true task outcome: how far the nail is actually
    # driven into the wall (obs[57]). Added outside the orientation gate so
    # that genuinely sinking the nail is valued above merely hovering the
    # hammer head in the target band with good grasp/orientation.
    reward = shaped + 20.0 * max(0.0, nail)
    return float(reward)
