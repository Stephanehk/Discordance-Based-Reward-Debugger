"""CLI: roll out every policy in round K *in parallel* across CPU cores, then
compute features.

This is a drop-in faster alternative to ``runtime.run_round`` for slow
environments (e.g. the agent-based pandemic simulator, where a single episode
takes ~a minute). Instead of rolling out every (policy, episode) pair serially
in one process, it fans them out over a process pool — ideal for a many-core
SLURM node.

Usage:
    python -m runtime.run_round_parallel --workspace workspaces/<name> --round K
    python -m runtime.run_round_parallel --workspace workspaces/<name> --round K \
        --rollouts-per-policy 3 --workers 16

Each (policy, episode) pair runs in its own worker, which builds its own env.
Workers seed the env per task (when ``make_env`` accepts a ``seed`` argument)
so outbreaks are distinct *and* reproducible across runs. The output layout is
byte-for-byte the same as ``runtime.run_round`` — trajectories under
``round_{K}_trajectories/`` and features under ``round_{K}_trajectory_features/``.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import itertools
import multiprocessing as mp
import os
import sys

from runtime.workspace import add_workspace_arg, activate
from runtime.env_loader import load_env_config
from runtime.compute_features import compute_features_for_round
from runtime.run_round import discover_policies, pick_rollouts_per_policy


def _init_worker(workspace: str) -> None:
    """Pool initializer: put the workspace on sys.path and chdir into it so
    `import env_config`, the policy package, and relative save paths resolve
    inside each spawned worker process."""
    activate(workspace)


def _build_env(seed: int):
    """Build the env, seeding it per task when the workspace's make_env supports
    a `seed` kwarg; otherwise fall back to whatever seeding make_env does."""
    import env_config

    try:
        params = inspect.signature(env_config.make_env).parameters
    except (TypeError, ValueError):
        params = {}

    if "seed" in params:
        return env_config.make_env(seed=seed)
    # Generic fallback: nudge a module-level seed counter if the workspace
    # exposes one, so forked/spawned workers don't all replay the same seed.
    if hasattr(env_config, "_seed_counter"):
        env_config._seed_counter = itertools.count(seed)
    return env_config.make_env()


def _rollout_one(task):
    """Worker entrypoint: roll out a single episode and save its trajectory.

    `task` = (round_k, policy_cls_name, episode_idx, seed, max_steps).
    Returns the path written (or raises, which the pool surfaces to the parent).
    """
    round_k, policy_cls_name, episode_idx, seed, max_steps = task

    from runtime.rollout import rollout_one_episode
    from runtime.trajectory_io import save_trajectory

    env = _build_env(seed)
    try:
        mod = importlib.import_module(
            f"gather_evidence.data_collection_policy_round_{round_k}"
        )
        policy = getattr(mod, policy_cls_name)(env)
        traj = rollout_one_episode(env, policy, max_steps)
        out_dir = f"gather_evidence/round_{round_k}_trajectories"
        path = os.path.join(out_dir, f"{policy_cls_name}__traj_{episode_idx:03d}.pkl")
        save_trajectory(path, traj, {
            "round": round_k,
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
    add_workspace_arg(p)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--rollouts-per-policy", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override MAX_EPISODE_STEPS from env_config.")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of worker processes (default: min(#tasks, CPU count)).")
    args = p.parse_args()

    ws = activate(args.workspace)
    print(f"[run_round_parallel] workspace={ws}")
    k = args.round
    cfg = load_env_config()
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_episode_steps"]
    n_per = pick_rollouts_per_policy(max_steps, args.rollouts_per_policy)

    os.makedirs(f"gather_evidence/round_{k}_trajectories", exist_ok=True)

    policy_classes = discover_policies(k)

    # One task per (policy, episode); each gets a unique, deterministic seed.
    tasks = []
    seed = 0
    for cls in policy_classes:
        for i in range(n_per):
            tasks.append((k, cls.__name__, i, seed, max_steps))
            seed += 1

    workers = args.workers if args.workers is not None else min(len(tasks), os.cpu_count() or 1)
    print(f"[run_round_parallel] round={k}  policies={len(policy_classes)}  "
          f"rollouts/policy={n_per}  tasks={len(tasks)}  workers={workers}  "
          f"max_steps={max_steps}")

    # 'spawn' gives each worker a clean interpreter, so the env's process-global
    # simulator state (e.g. pandemic_simulator globals) never leaks across tasks.
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(workers, initializer=_init_worker, initargs=(ws,)) as pool:
        for path in pool.imap_unordered(_rollout_one, tasks):
            done += 1
            print(f"[run_round_parallel]   ({done}/{len(tasks)}) wrote {path}")

    print(f"[run_round_parallel] computing features for round {k} ...")
    n_feat = compute_features_for_round(k)
    print(f"[run_round_parallel] wrote {n_feat} feature files. done.")


if __name__ == "__main__":
    sys.exit(main())
