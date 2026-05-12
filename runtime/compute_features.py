"""Batch-apply Trajectory2Features to a round's trajectories."""
from __future__ import annotations

import json
import os
from runtime.trajectory_io import iter_round_trajectory_files, load_trajectory


def features_path_for(traj_path: str) -> str:
    """gather_evidence/round_K_trajectories/foo.pkl -> .../round_K_trajectory_features/foo.json"""
    d, fname = os.path.split(traj_path)
    d2 = d.replace("_trajectories", "_trajectory_features")
    base = os.path.splitext(fname)[0]
    return os.path.join(d2, base + ".json")


def compute_features_for_round(round_k: int) -> int:
    from gather_evidence.trajectory_features import Trajectory2Features

    featurizer = Trajectory2Features()
    names = list(featurizer.feature_names)
    n_written = 0
    for traj_path in iter_round_trajectory_files(round_k):
        payload = load_trajectory(traj_path)
        feats = featurizer.get_traj_features(payload["trajectory"])
        feats = [float(x) for x in feats]
        if len(feats) != len(names):
            raise RuntimeError(
                f"feature length mismatch in {traj_path}: "
                f"{len(feats)} values vs {len(names)} names"
            )
        out_path = features_path_for(traj_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"features": feats, "names": names, "source": traj_path}, f, indent=2)
        n_written += 1
    return n_written
