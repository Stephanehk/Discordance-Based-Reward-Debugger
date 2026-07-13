"""Environment configuration for the Meta-World `hammer-v3` task.

A Sawyer arm must grasp a hammer and use it to drive a nail into a wall.
The episode runs for a fixed horizon.

Observation
-----------
The base Meta-World observation (39-dim) covers the current and previous hand
pose, gripper state, the two tracked objects' poses, and the task goal. Because
the reward and any trajectory analysis only ever receive `(obs, act, next_obs)`
(never the simulator's internal state), `make_env` appends a block of additional
world-frame physical quantities to the observation so they are available
downstream. See OBS_DESCRIPTION for the exact layout.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import metaworld

CAPTURE_FRAMES = True

_ENV_NAME = "hammer-v3"
_BENCH = metaworld.MT1(_ENV_NAME, seed=0)
_TASK = _BENCH.train_tasks[0]

OBS_DESCRIPTION = """\
Box(shape=(58,), dtype=float64). Concatenation of the Meta-World hammer
observation and a block of world-frame physical quantities:
  [0:3]   hand (end-effector) position (x, y, z), meters
  [3]     gripper aperture (open>0)
  [4:7]   hammer position (x, y, z), meters
  [7:11]  hammer orientation quaternion (w, x, y, z)
  [11:14] nail-head position (x, y, z), meters
  [14:18] nail orientation quaternion (w, x, y, z)
  [18:36] previous timestep's copy of indices [0:18]
  [36:39] task goal position (the driven-in nail-head target), meters
  [39:42] target position used by the reward (driven nail-head pose), meters
  [42:45] hand position at the start of the episode (x, y, z), meters
  [45:48] tool-center-point (between the fingers) position (x, y, z), meters
  [48:51] left gripper pad position (x, y, z), meters
  [51:54] right gripper pad position (x, y, z), meters
  [54:57] hammer position at the start of the episode (x, y, z), meters
  [57]    nail slide displacement (meters); 0 when flush, grows as the nail is
          driven in
"""

ACTION_DESCRIPTION = """\
Box(low=-1, high=1, shape=(4,), dtype=float32). A delta end-effector control:
  [0] delta x of the end-effector
  [1] delta y of the end-effector
  [2] delta z of the end-effector
  [3] gripper command (>0 closes the fingers, <0 opens them)
"""

MAX_EPISODE_STEPS = 150

_N_BASE = 39
_AUX = 19  # target(3)+init_tcp(3)+tcp(3)+left_pad(3)+right_pad(3)+obj_init(3)+nail(1)


class _StateAugmentWrapper(gym.Wrapper):
    """Append world-frame physical state consumed by the reward / needed to
    judge task progress but absent from the base observation."""

    def __init__(self, env):
        super().__init__(env)
        n = _N_BASE + _AUX
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64
        )

    def _aug(self, obs):
        u = self.env.unwrapped
        target = np.asarray(u._target_pos, dtype=np.float64).reshape(3)
        init_tcp = np.asarray(u.init_tcp, dtype=np.float64).reshape(3)
        tcp = np.asarray(u.tcp_center, dtype=np.float64).reshape(3)
        left_pad = np.asarray(u.get_body_com("leftpad"), dtype=np.float64).reshape(3)
        right_pad = np.asarray(u.get_body_com("rightpad"), dtype=np.float64).reshape(3)
        obj_init = np.asarray(u.obj_init_pos, dtype=np.float64).reshape(3)
        nail = np.asarray([float(u.data.joint("NailSlideJoint").qpos)], dtype=np.float64)
        return np.concatenate(
            [np.asarray(obs, dtype=np.float64).reshape(-1),
             target, init_tcp, tcp, left_pad, right_pad, obj_init, nail]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._aug(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._aug(obs), reward, terminated, truncated, info


def make_env(render_mode: str | None = None):
    cls = _BENCH.train_classes[_ENV_NAME]
    env = cls(render_mode=render_mode) if render_mode is not None else cls()
    env.set_task(_TASK)
    return _StateAugmentWrapper(env)
