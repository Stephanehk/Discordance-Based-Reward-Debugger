"""Env loading + gym/gymnasium compatibility shims."""
from __future__ import annotations

from typing import Any, Tuple


def load_env():
    """Build the env via the user's env_config.make_env()."""
    from env_config import make_env
    return make_env()


def load_env_config() -> dict:
    import env_config
    return {
        "obs_description": getattr(env_config, "OBS_DESCRIPTION", ""),
        "action_description": getattr(env_config, "ACTION_DESCRIPTION", ""),
        "max_episode_steps": getattr(env_config, "MAX_EPISODE_STEPS", 1000),
    }


def env_reset(env) -> Any:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def env_step(env, action) -> Tuple[Any, float, bool, dict]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
    elif len(out) == 4:
        obs, reward, done, info = out
        done = bool(done)
    else:
        raise RuntimeError(f"Unexpected env.step return arity: {len(out)}")
    return obs, float(reward), done, info
