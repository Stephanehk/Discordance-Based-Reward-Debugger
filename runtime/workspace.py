"""Workspace resolution: each environment lives in its own workspace directory.

A workspace is a directory containing the env-specific files (env_config.py,
designed_reward_fn.py, gather_evidence/, candidate_performance_metrics/).
The boilerplate in runtime/ + find_informative_pairs.py is env-agnostic and
operates against whichever workspace is selected.

The CLI entrypoints accept --workspace PATH. activate(path) prepends the
workspace to sys.path so `import env_config` and friends resolve, and
chdirs into it so the existing relative read/write paths in trajectory_io
and compute_features land in the right place.
"""
from __future__ import annotations

import argparse
import os
import sys


def add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Path to the workspace directory (e.g. workspaces/merge). "
             "Defaults to the contents of .current_workspace at the repo root.",
    )


def resolve_workspace(path: str | None) -> str:
    if path is None:
        marker = os.path.join(_repo_root(), ".current_workspace")
        if not os.path.exists(marker):
            raise SystemExit(
                "No --workspace given and no .current_workspace file found. "
                "Pass --workspace workspaces/<name> or run /repair-setup first."
            )
        with open(marker) as f:
            path = f.read().strip()
    # Bare names like "merge" resolve to workspaces/merge under the repo root.
    if os.sep not in path and "/" not in path:
        path = os.path.join(_repo_root(), "workspaces", path)
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise SystemExit(f"Workspace directory does not exist: {abs_path}")
    return abs_path


def activate(path: str | None) -> str:
    """Resolve, register on sys.path, and chdir into the workspace. Returns abs path."""
    abs_path = resolve_workspace(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
    os.chdir(abs_path)
    return abs_path


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
