"""Roll a policy out in the env and collect trajectories."""
from __future__ import annotations

import os

import numpy as np

from runtime.env_loader import env_reset, env_step, load_env_config


def _try_render_frame(env):
    """Return one rgb_array frame from env.render() or None if unavailable."""
    try:
        f = env.render()
    except Exception:
        return None
    if f is None:
        return None
    return np.asarray(f)


def _save_mp4(path: str, frames: list, fps: int = 4) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps)


def rollout_one_episode(env, policy, max_steps: int,
                         record_video_path: str | None = None) -> list:
    """Return a trajectory: list of (obs, action, next_obs) tuples.

    If `record_video_path` is given and `env.render()` returns frames, an
    mp4 is written to that path covering every step that was rolled out
    (initial frame + one frame per env.step()). The frames captured here
    are the actual trajectory; no replay is involved.
    """
    traj = []
    frames: list | None = [] if record_video_path else None
    obs = env_reset(env)
    if frames is not None:
        f = _try_render_frame(env)
        if f is not None:
            frames.append(f)
    for _ in range(max_steps):
        action = policy.act(obs)
        next_obs, _reward, done, _info = env_step(env, action)
        traj.append((obs, action, next_obs))
        if frames is not None:
            f = _try_render_frame(env)
            if f is not None:
                frames.append(f)
        obs = next_obs
        if done:
            break
    if record_video_path and frames:
        _save_mp4(record_video_path, frames)
    return traj


def rollout_policy(env, policy, n_episodes: int, max_steps: int) -> list[list]:
    """Return a list of trajectories, one per episode."""
    return [rollout_one_episode(env, policy, max_steps) for _ in range(n_episodes)]


def default_max_steps() -> int:
    return load_env_config()["max_episode_steps"]
