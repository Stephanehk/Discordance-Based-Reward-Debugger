"""Trajectory pickle save/load."""
from __future__ import annotations

import os
import pickle
import glob
import re
from typing import Iterator


def video_path_for(traj_path: str) -> str:
    """Sibling video path for a trajectory pickle.

    `gather_evidence/round_{k}_trajectories/X.pkl`
        →
    `gather_evidence/round_{k}_trajectory_videos/X.mp4`
    """
    d, fname = os.path.split(traj_path)
    parent, sub = os.path.split(d)
    new_sub = re.sub(r"trajectories$", "trajectory_videos", sub)
    base = os.path.splitext(fname)[0]
    return os.path.join(parent, new_sub, base + ".mp4")


def save_trajectory(path: str, trajectory: list, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"trajectory": trajectory, "metadata": metadata}, f)


def load_trajectory(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_round_trajectory_files(round_k: int) -> Iterator[str]:
    pattern = f"gather_evidence/round_{round_k}_trajectories/*.pkl"
    yield from sorted(glob.glob(pattern))


def iter_all_trajectory_files() -> Iterator[str]:
    pattern = "gather_evidence/round_*_trajectories/*.pkl"
    yield from sorted(glob.glob(pattern))


def list_rounds() -> list[int]:
    import re
    dirs = glob.glob("gather_evidence/round_*_trajectories")
    nums = []
    for d in dirs:
        m = re.search(r"round_(\d+)_trajectories$", d)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)
