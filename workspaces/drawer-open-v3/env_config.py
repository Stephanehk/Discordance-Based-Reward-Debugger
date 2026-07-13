"""Environment configuration for the Meta-World `drawer-open-v3` task.

A Sawyer arm must pull a drawer open to a target handle position. The
episode runs for a fixed horizon.

Observation
-----------
The base Meta-World observation (39-dim) describes the current and previous
hand pose, gripper state, and the two manipulable objects' poses, plus the
task goal. Because the reward and any trajectory analysis only ever receive
`(obs, act, next_obs)` (never the env's internal simulator state), `make_env`
appends a small block of additional physical quantities to the end of the
observation so they are available downstream:

  index  0:39   base Meta-World observation
                  [0:3]   hand position
                  [3]     gripper aperture
                  [4:7]   object-1 (drawer handle) position
                  [7:11]  object-1 orientation (quat)
                  [11:14] object-2 position
                  [14:18] object-2 orientation (quat)
                  [18:36] previous-step copy of [0:18]
                  [36:39] task goal position
  index 39:42   target handle position (drawer fully-open pose)
  index 42:45   initial hand position at episode start

All positions are in meters in the world frame.
"""

import os

# MuJoCo offscreen rendering on headless nodes.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import metaworld

CAPTURE_FRAMES = True

_ENV_NAME = "drawer-open-v3"
_BENCH = metaworld.MT1(_ENV_NAME, seed=0)
_TASK = _BENCH.train_tasks[0]

OBS_DESCRIPTION = """\
Box(shape=(45,), dtype=float64). Concatenation of the Meta-World drawer-open
observation and a small block of world-frame physical quantities:
  [0:3]   hand (end-effector) position (x, y, z), meters
  [3]     gripper aperture (open>0)
  [4:7]   drawer handle position (x, y, z), meters
  [7:11]  drawer handle orientation quaternion (w, x, y, z)
  [11:14] second tracked object position (unused by this task), meters
  [14:18] second tracked object orientation quaternion
  [18:36] previous timestep's copy of indices [0:18]
  [36:39] task goal position (x, y, z), meters
  [39:42] target handle position when the drawer is fully open (x, y, z), meters
  [42:45] hand position at the start of the episode (x, y, z), meters
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
_AUX = 6  # target_pos(3) + init_tcp(3)


class _StateAugmentWrapper(gym.Wrapper):
    """Append world-frame state (target handle pose, initial hand pose) that the
    reward consumes but which is not present in the base observation."""

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
        return np.concatenate([np.asarray(obs, dtype=np.float64).reshape(-1), target, init_tcp])

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
