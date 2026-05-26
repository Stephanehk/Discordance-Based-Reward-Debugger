"""CLI: roll out every policy in ``eval_scripts/<env>/policies.py`` *in
parallel* across CPU cores and save the resulting trajectories under
``eval_scripts/<env>/trajectory/``.

Drop-in faster alternative to ``runtime.run_eval_rollout`` for slow envs
(notably the agent-based pandemic simulator). Each ``(policy, episode)``
pair runs in its own worker process; the env is built fresh inside the
worker, and workers are seeded per task so the random baselines (and any
stochastic env transitions) are distinct *and* reproducible.

Usage:
    python -m runtime.run_eval_rollout_parallel \\
        --eval-dir eval_scripts/pandemic_human_eval_1 \\
        --env-workspace workspaces/pandemic_human_eval_1 \\
        --episodes-per-policy 10 --workers 24
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import multiprocessing as mp
import os
import sys

from runtime.workspace import activate
from runtime.env_loader import load_env_config


# Globals populated by ``_init_worker`` in each spawned worker process.
_WORKER_POLICIES_PATH: str = ""
_WORKER_OUTPUT_DIR: str = ""


def _load_policies_from_file(path: str) -> dict[str, type]:
    """Return ``{class_name: class}`` from a free-standing policies file."""
    spec = importlib.util.spec_from_file_location("eval_policies", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "POLICIES"):
        classes = list(mod.POLICIES)
    else:
        classes = [
            cls for name, cls in vars(mod).items()
            if inspect.isclass(cls)
            and name.startswith("DataCollectionPolicy")
            and cls.__module__ == mod.__name__
        ]
    return {c.__name__: c for c in classes}


def _init_worker(env_workspace: str, policies_path: str, output_dir: str) -> None:
    """Pool initializer: activate the env workspace and remember where the
    eval policies + output dir live, so ``_rollout_one`` can reach them."""
    global _WORKER_POLICIES_PATH, _WORKER_OUTPUT_DIR
    activate(env_workspace)
    _WORKER_POLICIES_PATH = policies_path
    _WORKER_OUTPUT_DIR = output_dir


def _build_env(seed: int):
    """Build the env, seeding it per task when the workspace's ``make_env``
    accepts a ``seed`` kwarg."""
    import env_config
    try:
        params = inspect.signature(env_config.make_env).parameters
    except (TypeError, ValueError):
        params = {}
    if "seed" in params:
        return env_config.make_env(seed=seed)
    if hasattr(env_config, "_seed_counter"):
        import itertools
        env_config._seed_counter = itertools.count(seed)
    return env_config.make_env()


def _instantiate(policy_cls, env, episode_idx: int):
    """Pass ``seed=episode_idx`` through if the policy's ``__init__`` accepts it."""
    try:
        params = inspect.signature(policy_cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    if "seed" in params:
        return policy_cls(env, seed=episode_idx)
    return policy_cls(env)


def _rollout_one(task):
    """Worker entrypoint: roll out a single (policy, episode) and save it."""
    policy_cls_name, episode_idx, seed, max_steps = task

    from runtime.rollout import rollout_one_episode
    from runtime.trajectory_io import save_trajectory

    classes = _load_policies_from_file(_WORKER_POLICIES_PATH)
    if policy_cls_name not in classes:
        raise RuntimeError(
            f"Policy class {policy_cls_name!r} not found in {_WORKER_POLICIES_PATH}"
        )
    policy_cls = classes[policy_cls_name]

    env = _build_env(seed)
    try:
        policy = _instantiate(policy_cls, env, episode_idx)
        traj = rollout_one_episode(env, policy, max_steps)
        path = os.path.join(_WORKER_OUTPUT_DIR,
                            f"{policy_cls_name}__traj_{episode_idx:03d}.pkl")
        save_trajectory(path, traj, {
            "policy_class": policy_cls_name,
            "rollout_index": episode_idx,
            "n_steps": len(traj),
            "seed": seed,
        })
    finally:
        try:
            env.close()
        except Exception:
            pass
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True,
                   help="Directory containing policies.py and the trajectory/ "
                        "output dir (e.g. eval_scripts/pandemic_human_eval_1).")
    p.add_argument("--env-workspace", required=True,
                   help="Workspace whose env_config defines the env "
                        "(e.g. workspaces/pandemic_human_eval_1).")
    p.add_argument("--episodes-per-policy", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override MAX_EPISODE_STEPS from env_config.")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of worker processes (default: min(#tasks, CPU count)).")
    args = p.parse_args()

    # Resolve absolutes BEFORE activate() chdirs us.
    eval_dir_abs = os.path.abspath(args.eval_dir)
    policies_path = os.path.join(eval_dir_abs, "policies.py")
    output_dir = os.path.join(eval_dir_abs, "trajectory")

    if not os.path.exists(policies_path):
        raise SystemExit(f"Policies file not found: {policies_path}")

    # Resolve the env workspace in the parent (lets us print + read max_steps).
    ws = activate(args.env_workspace)
    print(f"[eval_rollout_parallel] env workspace = {ws}")
    print(f"[eval_rollout_parallel] eval dir      = {eval_dir_abs}")

    cfg = load_env_config()
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_episode_steps"]
    n_per = args.episodes_per_policy

    classes = _load_policies_from_file(policies_path)
    if not classes:
        raise SystemExit(f"No policy classes found in {policies_path}")
    policy_names = list(classes.keys())

    os.makedirs(output_dir, exist_ok=True)

    # One task per (policy, episode); deterministic, unique seeds.
    tasks = []
    seed = 0
    for name in policy_names:
        for i in range(n_per):
            tasks.append((name, i, seed, max_steps))
            seed += 1

    workers = args.workers if args.workers is not None else min(len(tasks), os.cpu_count() or 1)
    print(f"[eval_rollout_parallel] policies={len(policy_names)}  "
          f"episodes/policy={n_per}  tasks={len(tasks)}  workers={workers}  "
          f"max_steps={max_steps}")

    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(ws, policies_path, output_dir),
    ) as pool:
        for path in pool.imap_unordered(_rollout_one, tasks):
            done += 1
            print(f"[eval_rollout_parallel]   ({done}/{len(tasks)}) wrote {os.path.basename(path)}")

    print(f"[eval_rollout_parallel] wrote {len(tasks)} trajectories to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
