"""Trajectory visualization: copy the captured-frames mp4 recorded during the
rollout (CAPTURE_FRAMES=True in env_config) to the per-comparison path."""

import os
import shutil

from runtime.trajectory_io import video_path_for

VIZ_EXT = ".mp4"


def visualize_trajectory(trajectory, metadata, out_path, source_path=None, **_):
    if source_path is None:
        raise RuntimeError(
            "viz needs the trajectory's source .pkl path to locate the "
            "captured-frames sidecar."
        )
    sidecar = video_path_for(source_path)
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"no captured-frames video at {sidecar}; re-run /repair-round to "
            f"record fresh evidence with frame capture enabled."
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(sidecar, out_path)
