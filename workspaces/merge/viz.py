"""Trajectory visualization for the merge workspace.

/repair-preferences calls `visualize_trajectory` per shown trajectory.
Frames are captured during the original rollout (see runtime/rollout.py +
runtime/run_round*.py; opt-in via `CAPTURE_FRAMES = True` in env_config),
and saved as an mp4 sibling to the trajectory pickle:

    gather_evidence/round_{k}_trajectories/X.pkl
        →   gather_evidence/round_{k}_trajectory_videos/X.mp4

This module simply copies that sidecar mp4 to the per-comparison
visualization path — no replay, no re-renderering. If the sidecar is
missing (e.g. rollouts predate the frame-capture feature), the call
raises and the caller surfaces the failure.
"""

import os
import shutil

from runtime.trajectory_io import video_path_for


VIZ_EXT = ".mp4"


def visualize_trajectory(trajectory, metadata, out_path, source_path=None, **_):
    if source_path is None:
        raise RuntimeError(
            "merge viz needs the trajectory's source .pkl path to locate the "
            "captured-frames sidecar; select_pairs is expected to supply it."
        )
    sidecar = video_path_for(source_path)
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"no captured-frames video at {sidecar}; this trajectory predates "
            f"frame capture — re-run /repair-round to record fresh evidence."
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(sidecar, out_path)
