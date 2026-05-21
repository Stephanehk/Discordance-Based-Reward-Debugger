"""Baseline trajectory display (single-trajectory analogue of select_pairs.py).

The reward-repairer *method* surfaces trajectory **pairs** where the designed
reward and an independent stakeholder metric disagree. The **baseline** is a
different process: it builds one policy that maximizes the designed reward,
rolls it out, and shows the human a handful of those trajectories on their own
— encoded in feature space, with each trajectory's total under the designed
reward. There are no stakeholder performance metrics and no pairs; the human
inspects what the reward actually incentivizes and decides whether to edit it.

This script is the rendering half of that step. It deliberately does NOT load
`candidate_performance_metrics/performance_metrics.py` (the baseline has none),
so it works in a baseline workspace that contains only env_config,
designed_reward_fn, and trajectory_features.

Two modes:

  --mode dump
      Load every trajectory, compute its designed-reward total and its
      features, assign each a stable id (T001, T002, ...), and write a
      human-legible table to .baseline_history/_trajectory_table.txt (plus a
      machine pool .baseline_history/_trajectory_pool.json mapping ids to
      paths). Claude reads the table to pick a small, feature-diverse set
      (2-5) to surface.

  --mode render --selection <file.json>
      Read a selection file of the form
          {"trajectories": ["T003", "T012", ...]}
      and render each selected trajectory as an anonymized block — its total
      under the designed reward and a feature-value table under a shared
      feature key — then append the shown set to a snapshot under
      .baseline_history/ for provenance.

Usage:
    python show_trajectories.py --workspace baseline_workspaces/<name> --mode dump
    python show_trajectories.py --workspace baseline_workspaces/<name> --mode render \
           --selection baseline_workspaces/<name>/.baseline_history/_selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime

from runtime.workspace import add_workspace_arg, activate
from runtime.trajectory_io import (
    iter_all_trajectory_files,
    iter_round_trajectory_files,
    load_trajectory,
)

# Reuse the designed-reward loader, the feature-on-disk reader, and the
# feature-key printer from the (metrics-free) parts of the algorithmic
# selector so the displayed features match what every other view shows.
# We never call its load_records / _load_performance_metrics — the baseline
# has no stakeholder metrics.
import find_informative_pairs as fip


HISTORY_DIR = ".baseline_history"
TABLE_PATH = os.path.join(HISTORY_DIR, "_trajectory_table.txt")
POOL_PATH = os.path.join(HISTORY_DIR, "_trajectory_pool.json")


@dataclass
class TrajRecord:
    path: str
    metadata: dict
    designed_reward: float
    features: list[float]
    feature_names: list


def _src_tag(metadata: dict) -> str:
    """Internal-only provenance tag (round + rollout) for diversity reasoning.

    Shown to Claude in the table so it can pick a varied subset; it is never
    rendered into a user-facing trajectory block (render mode is anonymized),
    and the skill instructs Claude not to surface it.
    """
    rnd = metadata.get("round", "?")
    idx = metadata.get("rollout_index", "?")
    return f"r{rnd}:{idx}"


def _load_records(rounds: list[int] | None) -> list[TrajRecord]:
    if rounds is None:
        paths = list(iter_all_trajectory_files())
    else:
        paths = []
        for k in rounds:
            paths.extend(iter_round_trajectory_files(k))
        paths.sort()
    if not paths:
        raise SystemExit(
            "No trajectories found. Roll out the reward-maximizing policy first "
            "(run /baseline-evidence)."
        )

    designed_reward_fn = fip._load_designed_reward_fn()
    records: list[TrajRecord] = []
    for path in paths:
        data = load_trajectory(path)
        traj = data["trajectory"]
        if not traj:
            continue
        r_total = sum(float(designed_reward_fn(o, a, no)) for (o, a, no) in traj)
        feats, names = fip._read_features_for(path)
        records.append(TrajRecord(
            path=path,
            metadata=data.get("metadata", {}),
            designed_reward=r_total,
            features=feats or [],
            feature_names=names or [],
        ))
    records.sort(key=lambda r: r.path)  # stable ids across modes
    return records


def _assign_ids(records) -> list[str]:
    return [f"T{idx:03d}" for idx in range(1, len(records) + 1)]


def _do_dump(records) -> None:
    ids = _assign_ids(records)
    feature_names = records[0].feature_names if records else []

    lines: list[str] = []
    lines.append(f"TRAJECTORY TABLE — {len(records)} trajectories from the "
                 f"reward-maximizing policy")
    lines.append("")
    lines.append("Each row is one trajectory the reward-maximizing policy produced,")
    lines.append("summarised exactly as the human sees it (the features below) plus")
    lines.append("its total under the designed reward. Pick a small, feature-diverse")
    lines.append("set (2-5) that best shows what behavior maximizing this reward")
    lines.append("actually produces.")
    lines.append("")
    lines.append("Feature key (short_name — what it measures):")
    for short, desc in feature_names:
        lines.append(f"  {short:<18} {desc}")
    lines.append("")
    lines.append("'src' is internal provenance for your diversity reasoning only —")
    lines.append("never reveal round or rollout indices to the user.")
    lines.append("")

    feat_shorts = [s for s, _ in feature_names]
    header = ["id", "reward"] + feat_shorts + ["src"]
    widths = [6, 10] + [max(8, len(s)) for s in feat_shorts] + [12]
    lines.append("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    lines.append("  ".join("-" * w for w in widths))

    pool = {}
    for tid, rec in zip(ids, records):
        row = [tid.rjust(widths[0]), f"{rec.designed_reward:+.2f}".rjust(widths[1])]
        ci = 2
        for v in rec.features:
            row.append(f"{v:+.2f}".rjust(widths[ci]))
            ci += 1
        row.append(_src_tag(rec.metadata).rjust(widths[ci]))
        lines.append("  ".join(row))
        pool[tid] = {"path": rec.path}

    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(TABLE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(POOL_PATH, "w") as f:
        json.dump({"ids": ids, "pool": pool}, f, indent=2)

    print(f"[show_trajectories] wrote table for {len(records)} trajectories to {TABLE_PATH}")
    print(f"[show_trajectories] read that file, then write a selection JSON of the form")
    print(f'               {{"trajectories": ["T003", "T012", ...]}}  (pick 2-5)')
    print(f"               and run --mode render --selection <that file>.")


def _print_trajectory(rank: int, total: int, rec: TrajRecord) -> None:
    """Anonymized single-trajectory block: designed-reward total + feature table."""
    print()
    print("=" * 78)
    print(f"TRAJECTORY {rank}/{total}")
    print("-" * 78)
    print(f"  designed reward (total over the episode):  {rec.designed_reward:>+14.3f}")
    print()
    if rec.features and rec.feature_names:
        print(f"  Trajectory features (see feature key above):")
        print(f"    {'feature':28s}   {'value':>12s}")
        print(f"    {'-' * 46}")
        for (short, _desc), v in zip(rec.feature_names, rec.features):
            print(f"    {short:28s}   {v:>+12.2f}")
    else:
        print("  (no features on disk for this trajectory)")


def _save_snapshot(records, selected_ids, rec_by_id) -> str:
    import glob
    import re

    os.makedirs(HISTORY_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(HISTORY_DIR, "shown_round_*.json"))
    nums = [int(re.search(r"shown_round_(\d+)\.json", p).group(1)) for p in existing]
    snap_id = (max(nums) + 1) if nums else 1
    snap = {
        "snapshot_id": snap_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selection_method": "claude_selected_single_trajectories",
        "trajectories": [],
    }
    for rank, tid in enumerate(selected_ids, 1):
        rec = rec_by_id[tid]
        snap["trajectories"].append({
            "rank": rank,
            "traj_path": rec.path,
            "snapshot_reward": float(rec.designed_reward),
        })
    out_path = os.path.join(HISTORY_DIR, f"shown_round_{snap_id:03d}.json")
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=2)
    return out_path


def _do_render(records, selection_path: str) -> int:
    with open(selection_path) as f:
        selection = json.load(f)
    tids = selection.get("trajectories", [])
    if not tids:
        print("[show_trajectories] selection file contains no trajectories.")
        return 1

    ids = _assign_ids(records)
    rec_by_id = dict(zip(ids, records))
    for tid in tids:
        if tid not in rec_by_id:
            raise SystemExit(f"[show_trajectories] unknown trajectory id: {tid!r}")

    feature_names = records[0].feature_names if records else []
    fip._print_feature_key(feature_names)
    print(f"\n{len(tids)} trajectories from the reward-maximizing policy "
          f"(Claude-selected):")

    total = len(tids)
    for rank, tid in enumerate(tids, 1):
        _print_trajectory(rank, total, rec_by_id[tid])

    snap_path = _save_snapshot(records, tids, rec_by_id)
    print(f"\n[snapshot saved: {snap_path}]")
    return 0


def main():
    p = argparse.ArgumentParser()
    add_workspace_arg(p)
    p.add_argument("--mode", choices=["dump", "render"], required=True)
    p.add_argument("--selection", type=str, default=None,
                   help="Path to the selection JSON (render mode).")
    p.add_argument("--rounds", type=str, default=None,
                   help="Comma-separated round numbers; default = all.")
    args = p.parse_args()

    # Resolve the selection path against the original cwd before activate()
    # chdirs into the workspace.
    if args.selection:
        args.selection = os.path.abspath(args.selection)

    ws = activate(args.workspace)
    print(f"[show_trajectories] workspace={ws}")

    rounds = None
    if args.rounds:
        rounds = [int(x) for x in args.rounds.split(",")]

    records = _load_records(rounds)
    print(f"[show_trajectories] loaded {len(records)} trajectories "
          f"from rounds {sorted({r.metadata.get('round') for r in records})}")

    if args.mode == "dump":
        _do_dump(records)
        return 0
    if not args.selection:
        raise SystemExit("[show_trajectories] --mode render requires --selection <file>.")
    return _do_render(records, args.selection)


if __name__ == "__main__":
    sys.exit(main())
