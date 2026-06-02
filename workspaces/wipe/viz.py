"""Trajectory visualization for the robosuite Wipe workspace.

Frames are captured during the original rollout (env_config opts in via
``CAPTURE_FRAMES = True``) and stored as an mp4 sibling to the
trajectory pickle:

    gather_evidence/round_{k}_trajectories/X.pkl
        →   gather_evidence/round_{k}_trajectory_videos/X.mp4

``visualize_trajectory`` simply copies that sidecar into the per-
comparison visualization path — no replay, no re-rendering.
"""

import os
import shutil

from runtime.trajectory_io import video_path_for


VIZ_EXT = ".mp4"


def visualize_trajectory(trajectory, metadata, out_path, source_path=None, **_):
    if source_path is None:
        raise RuntimeError(
            "wipe viz needs the trajectory's source .pkl path to locate "
            "the captured-frames sidecar; select_pairs is expected to "
            "supply it."
        )
    sidecar = video_path_for(source_path)
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"no captured-frames video at {sidecar}; this trajectory "
            f"predates frame capture — re-run /repair-round to record "
            f"fresh evidence."
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(sidecar, out_path)
