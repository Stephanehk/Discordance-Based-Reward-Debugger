"""Trajectory pickle save/load."""
from __future__ import annotations

import os
import pickle
import glob
from typing import Iterator


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
