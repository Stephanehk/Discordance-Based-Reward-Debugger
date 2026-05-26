"""CLI: roll out every policy in ``eval_scripts/<env>/policies.py`` and save
the resulting trajectories under ``eval_scripts/<env>/trajectory/``.

This is the eval analogue of ``runtime.run_round``. The policies live
outside the workspace (they are part of the evaluation harness), but the
env still comes from a workspace's ``env_config.make_env()``. Pass the env
source via ``--env-workspace`` (e.g. ``workspaces/merge_human_eval_1``).

Usage:
    python -m runtime.run_eval_rollout \\
        --eval-dir eval_scripts/merge_human_eval_1 \\
        --env-workspace workspaces/merge_human_eval_1 \\
        --episodes-per-policy 10

Each policy is re-instantiated between episodes so any per-episode state
(timestep counters, monotone ratchets, seeded RNGs) starts fresh; if a
policy's ``__init__`` accepts a ``seed`` keyword, the episode index is
passed in so the random baseline is reproducible.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys

from runtime.workspace import activate
from runtime.env_loader import load_env, load_env_config
from runtime.rollout import rollout_one_episode
from runtime.trajectory_io import save_trajectory


def _load_policies_from_file(path: str) -> list[type]:
    """Load policy classes from a free-standing ``policies.py`` file."""
    if not os.path.exists(path):
        raise SystemExit(f"Policies file not found: {path}")
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
    if not classes:
        raise SystemExit(f"No policy classes found in {path}")
    return classes


def _instantiate(policy_cls, env, episode_idx: int):
    """Instantiate a policy, threading the episode index in as ``seed`` if accepted."""
    try:
        params = inspect.signature(policy_cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    if "seed" in params:
        return policy_cls(env, seed=episode_idx)
    return policy_cls(env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True,
                   help="Directory containing policies.py and the trajectory/ "
                        "output dir (e.g. eval_scripts/merge_human_eval_1).")
    p.add_argument("--env-workspace", required=True,
                   help="Workspace whose env_config defines the env "
                        "(e.g. workspaces/merge_human_eval_1).")
    p.add_argument("--episodes-per-policy", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override MAX_EPISODE_STEPS from env_config.")
    args = p.parse_args()

    # Resolve paths to absolutes *before* activate() chdirs us.
    eval_dir_abs = os.path.abspath(args.eval_dir)
    policies_path = os.path.join(eval_dir_abs, "policies.py")
    output_dir = os.path.join(eval_dir_abs, "trajectory")

    ws = activate(args.env_workspace)
    print(f"[eval_rollout] env workspace = {ws}")
    print(f"[eval_rollout] eval dir      = {eval_dir_abs}")

    policy_classes = _load_policies_from_file(policies_path)
    cfg = load_env_config()
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_episode_steps"]
    n_per = args.episodes_per_policy

    os.makedirs(output_dir, exist_ok=True)
    print(f"[eval_rollout] policies={len(policy_classes)}  "
          f"episodes/policy={n_per}  max_steps={max_steps}")

    env = load_env()
    try:
        for policy_cls in policy_classes:
            cls_name = policy_cls.__name__
            print(f"[eval_rollout]   rolling out {cls_name} ...")
            for i in range(n_per):
                policy = _instantiate(policy_cls, env, episode_idx=i)
                traj = rollout_one_episode(env, policy, max_steps)
                path = os.path.join(output_dir, f"{cls_name}__traj_{i:03d}.pkl")
                save_trajectory(path, traj, {
                    "policy_class": cls_name,
                    "rollout_index": i,
                    "n_steps": len(traj),
                    "eval_dir": eval_dir_abs,
                    "env_workspace": ws,
                })
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(f"[eval_rollout] wrote {len(policy_classes) * n_per} trajectories "
          f"to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
