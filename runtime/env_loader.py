"""Env loading + gym/gymnasium compatibility shims."""
from __future__ import annotations

import inspect

from typing import Any, Tuple


def load_env(*, render_mode: str | None = None, seed: int | None = None):
    """Build the env via the user's env_config.make_env().

    Passes `render_mode` and/or `seed` to make_env only if its signature
    accepts them — preserves backward compat with workspaces whose make_env
    takes no kwargs.
    """
    from env_config import make_env
    try:
        params = inspect.signature(make_env).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {}
    if render_mode is not None and "render_mode" in params:
        kwargs["render_mode"] = render_mode
    if seed is not None and "seed" in params:
        kwargs["seed"] = seed
    return make_env(**kwargs)


def capture_frames_enabled() -> bool:
    """True if the active workspace opts into per-step frame capture."""
    import env_config
    return bool(getattr(env_config, "CAPTURE_FRAMES", False))


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
