"""Roll a policy out in the env and collect trajectories."""
from __future__ import annotations

from runtime.env_loader import env_reset, env_step, load_env_config


def rollout_one_episode(env, policy, max_steps: int) -> list:
    """Return a trajectory: list of (obs, action, next_obs) tuples."""
    traj = []
    obs = env_reset(env)
    for _ in range(max_steps):
        action = policy.act(obs)
        next_obs, _reward, done, _info = env_step(env, action)
        traj.append((obs, action, next_obs))
        obs = next_obs
        if done:
            break
    return traj


def rollout_policy(env, policy, n_episodes: int, max_steps: int) -> list[list]:
    """Return a list of trajectories, one per episode."""
    return [rollout_one_episode(env, policy, max_steps) for _ in range(n_episodes)]


def default_max_steps() -> int:
    return load_env_config()["max_episode_steps"]
