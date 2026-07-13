"""Environment configuration for the Gymnasium-Robotics `PointMaze_UMaze-v3`
task (dense reward).

A force-controlled point mass must reach a goal cell in a C-shaped maze. The
goal and start cells are sampled each episode.

Observation
-----------
The base env returns a dict {observation, achieved_goal, desired_goal}. To
conform to the array-based `(obs, act, next_obs)` pipeline, `make_env` flattens
it into an 8-vector (see OBS_DESCRIPTION).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

CAPTURE_FRAMES = True

OBS_DESCRIPTION = """\
Box(shape=(8,), dtype=float64), a flattened point-maze observation:
  [0:2] ball position (x, y) in the maze frame (meters)
  [2:4] ball velocity (vx, vy)
  [4:6] achieved goal = ball position (x, y)  [same as 0:2]
  [6:8] desired goal position (x, y)
The maze cells are ~1 m; the walls of the C-shaped maze are not encoded in the
observation.
"""

ACTION_DESCRIPTION = """\
Box(low=-1, high=1, shape=(2,), dtype=float32): linear force applied to the
ball, (force_x, force_y).
"""

MAX_EPISODE_STEPS = 300


class _FlattenDictWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )

    @staticmethod
    def _flat(obs):
        return np.concatenate([
            np.asarray(obs["observation"], dtype=np.float64).reshape(-1),
            np.asarray(obs["achieved_goal"], dtype=np.float64).reshape(-1),
            np.asarray(obs["desired_goal"], dtype=np.float64).reshape(-1),
        ])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flat(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flat(obs), reward, terminated, truncated, info


def make_env(render_mode: str | None = None):
    kwargs = {"reward_type": "dense", "max_episode_steps": MAX_EPISODE_STEPS}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return _FlattenDictWrapper(gym.make("PointMaze_UMaze-v3", **kwargs))
