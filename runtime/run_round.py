"""CLI: roll out every policy in round K and compute features.

Usage:
    python -m runtime.run_round --workspace workspaces/<name> --round K
    python -m runtime.run_round --workspace workspaces/<name> --round K --rollouts-per-policy 30
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys

from runtime.workspace import add_workspace_arg, activate
from runtime.env_loader import load_env, load_env_config, capture_frames_enabled
from runtime.rollout import rollout_one_episode
from runtime.trajectory_io import save_trajectory, video_path_for
from runtime.compute_features import compute_features_for_round


def discover_policies(round_k: int) -> list[type]:
    mod_name = f"gather_evidence.data_collection_policy_round_{round_k}"
    mod = importlib.import_module(mod_name)
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
        raise RuntimeError(f"No policy classes found in {mod_name}")
    return classes


MAX_TOTAL_ROLLOUTS = 100


def pick_rollouts_per_policy(max_steps: int, override: int | None) -> int:
    if override is not None:
        return override
    if max_steps <= 200:
        return 30
    if max_steps <= 1000:
        return 20
    return 10


def cap_rollouts(n_policies: int, n_per: int) -> int:
    """Clamp so that n_policies * n_per <= MAX_TOTAL_ROLLOUTS."""
    if n_policies <= 0:
        return n_per
    capped = min(n_per, max(1, MAX_TOTAL_ROLLOUTS // n_policies))
    return capped


def main():
    p = argparse.ArgumentParser()
    add_workspace_arg(p)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--rollouts-per-policy", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override MAX_EPISODE_STEPS from env_config.")
    args = p.parse_args()

    ws = activate(args.workspace)
    print(f"[run_round] workspace={ws}")
    k = args.round
    cfg = load_env_config()
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_episode_steps"]
    n_per = pick_rollouts_per_policy(max_steps, args.rollouts_per_policy)

    out_dir = f"gather_evidence/round_{k}_trajectories"
    os.makedirs(out_dir, exist_ok=True)

    policy_classes = discover_policies(k)
    n_per = cap_rollouts(len(policy_classes), n_per)
    capture = capture_frames_enabled()
    print(f"[run_round] round={k}  policies={len(policy_classes)}  "
          f"rollouts/policy={n_per}  total={len(policy_classes) * n_per}  "
          f"(cap={MAX_TOTAL_ROLLOUTS})  max_steps={max_steps}  "
          f"capture_frames={capture}")

    env = load_env(render_mode=("rgb_array" if capture else None))
    try:
        for policy_cls in policy_classes:
            policy = policy_cls(env)
            cls_name = policy_cls.__name__
            print(f"[run_round]   rolling out {cls_name} ...")
            for i in range(n_per):
                traj_path = os.path.join(out_dir, f"{cls_name}__traj_{i:03d}.pkl")
                video_path = video_path_for(traj_path) if capture else None
                traj = rollout_one_episode(
                    env, policy, max_steps, record_video_path=video_path,
                )
                save_trajectory(traj_path, traj, {
                    "round": k,
                    "policy_class": cls_name,
                    "rollout_index": i,
                    "n_steps": len(traj),
                })
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(f"[run_round] computing features for round {k} ...")
    n_feat = compute_features_for_round(k)
    print(f"[run_round] wrote {n_feat} feature files. done.")


if __name__ == "__main__":
    sys.exit(main())
