"""Claude-driven trajectory-pair selection (replaces the algorithmic
selector in find_informative_pairs.py for the /repair-preferences step).

The old selector (find_informative_pairs.py, kept for reference) chose
pairs with a Mahalanobis-greedy criterion over feature deltas. This
script instead exposes the raw material — every trajectory summarised by
its features, its designed-reward total, and every stakeholder metric —
and lets Claude pick the pairs that will be most useful to show the human.
Claude reads the table, writes a small selection file naming the pairs it
wants, and this script renders them in the exact same preference-label
format and writes the same snapshot schema, so /repair-verify is
unaffected.

Two modes:

  --mode dump
      Load every trajectory, compute its designed-reward total and each
      stakeholder metric, assign each a stable id (T001, T002, ...), and
      write a human-legible table to .repair_history/_trajectory_table.txt
      (plus a machine pool .repair_history/_pair_pool.json mapping ids to
      paths). Claude reads the table to decide which pairs to surface.

  --mode render --selection <file.json>
      Read a selection file of the form
          {"pairs": [{"a": "T012", "b": "T034"}, ...]}
      and, for each pair, orient A = the higher-designed-reward side
      (the reward-preferred trajectory, matching the snapshot/verify
      contract), list the stakeholder perspectives that prefer B, print
      the preference-label block, and append the pair to a snapshot under
      .repair_history/ so /repair-verify can recheck it later.

Usage:
    python select_pairs.py --workspace workspaces/<name> --mode dump
    python select_pairs.py --workspace workspaces/<name> --mode render \
           --selection .repair_history/_selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from runtime.workspace import add_workspace_arg, activate

# Reuse the loading / printing / snapshot machinery from the algorithmic
# selector so the displayed pairs and the saved snapshot are byte-for-byte
# compatible with what /repair-preferences and /repair-verify expect.
import find_informative_pairs as fip


HISTORY_DIR = fip.HISTORY_DIR
TABLE_PATH = os.path.join(HISTORY_DIR, "_trajectory_table.txt")
POOL_PATH = os.path.join(HISTORY_DIR, "_pair_pool.json")


def _metric_tag(metric_name: str) -> str:
    """Short column label for a performance metric (drops the prefix)."""
    return metric_name.removeprefix("performance_metric_")


def _ordered_records(rounds):
    """Load records and sort by path so ids are stable across modes."""
    records = fip.load_records(rounds)
    records.sort(key=lambda r: r.path)
    return records


def _assign_ids(records) -> list[str]:
    return [f"T{idx:03d}" for idx in range(1, len(records) + 1)]


def _src_tag(metadata: dict) -> str:
    """Internal-only provenance tag (round + policy) for diversity reasoning.

    Shown to Claude in the table to help it avoid picking many near-identical
    pairs; it is never rendered into a user-facing pair (render mode is fully
    anonymised), and the skill instructs Claude not to surface it.
    """
    rnd = metadata.get("round", "?")
    pol = (metadata.get("policy") or metadata.get("policy_class") or "?")
    pol = str(pol).removeprefix("DataCollectionPolicy")
    return f"r{rnd}:{pol}"


def _do_dump(records, metrics) -> None:
    ids = _assign_ids(records)
    metric_names = [m[0] for m in metrics]
    metric_docs = {m[0]: m[2] for m in metrics}
    tags = {name: _metric_tag(name) for name in metric_names}
    feature_names = records[0].feature_names if records else []

    lines: list[str] = []
    lines.append(f"TRAJECTORY TABLE — {len(records)} trajectories")
    lines.append("")
    lines.append("Each row is one trajectory, summarised exactly as the human sees it")
    lines.append("(the features below), plus its total under the designed reward and")
    lines.append("each stakeholder metric. Pick pairs (by id) where the designed")
    lines.append("reward ranks the two trajectories oppositely to a stakeholder you")
    lines.append("think the human cares about, or that otherwise best expose how the")
    lines.append("reward trades off the outcomes that matter.")
    lines.append("")
    lines.append("Feature key (short_name — what it measures):")
    for short, desc in feature_names:
        lines.append(f"  {short:<18} {desc}")
    lines.append("")
    lines.append("Stakeholder metrics (higher = better by that stakeholder's lights):")
    for name in metric_names:
        lines.append(f"  [{tags[name]:<16}] {metric_docs[name]}")
    lines.append("")
    lines.append("'src' is internal provenance for your diversity reasoning only —")
    lines.append("never reveal round, policy, or step counts to the user.")
    lines.append("")

    # Header row.
    feat_shorts = [s for s, _ in feature_names]
    header = ["id", "reward"] + [tags[n] for n in metric_names] + feat_shorts + ["src"]
    widths = [6, 10] + [12] * len(metric_names) + [max(8, len(s)) for s in feat_shorts] + [16]
    lines.append("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    lines.append("  ".join("-" * w for w in widths))

    pool = {}
    for tid, rec in zip(ids, records):
        row = [tid.rjust(widths[0]), f"{rec.designed_reward:+.2f}".rjust(widths[1])]
        ci = 2
        for name in metric_names:
            row.append(f"{rec.metric_values[name]:+.2f}".rjust(widths[ci]))
            ci += 1
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

    print(f"[select_pairs] wrote table for {len(records)} trajectories to {TABLE_PATH}")
    print(f"[select_pairs] read that file, then write a selection JSON of the form")
    print(f'               {{"pairs": [{{"a": "T012", "b": "T034"}}, ...]}}')
    print(f"               and run --mode render --selection <that file>.")


def _do_render(records, metrics, selection_path: str) -> int:
    with open(selection_path) as f:
        selection = json.load(f)
    pairs = selection.get("pairs", [])
    if not pairs:
        print("[select_pairs] selection file contains no pairs.")
        return 1

    ids = _assign_ids(records)
    rec_by_id = dict(zip(ids, records))
    metric_items = [(m[0], m[2]) for m in metrics]  # (name, doc)

    feature_names = records[0].feature_names if records else []
    fip._print_feature_key(feature_names)
    print(f"\nTop {len(pairs)} preference-label disagreements (Claude-selected):")

    selected: list[tuple[float, fip.PairCandidate]] = []
    for entry in pairs:
        a_id, b_id = entry["a"], entry["b"]
        if a_id not in rec_by_id or b_id not in rec_by_id:
            raise SystemExit(f"[select_pairs] unknown trajectory id in selection: "
                             f"{a_id!r} / {b_id!r}")
        ra, rb = rec_by_id[a_id], rec_by_id[b_id]
        # Orient so A is the reward-preferred side (snapshot/verify contract).
        if rb.designed_reward > ra.designed_reward:
            ra, rb = rb, ra
        # A stakeholder disagrees when it scores B above A.
        disagreements = []
        for name, doc in metric_items:
            if rb.metric_values[name] > ra.metric_values[name]:
                disagreements.append({
                    "perspective": doc,
                    "metric_a": float(ra.metric_values[name]),
                    "metric_b": float(rb.metric_values[name]),
                })
        a_idx = records.index(ra)
        b_idx = records.index(rb)
        import numpy as np
        delta = (np.asarray(rb.features, dtype=float)
                 - np.asarray(ra.features, dtype=float))
        selected.append((0.0, fip.PairCandidate(
            a_idx=a_idx, b_idx=b_idx, delta=delta, disagreements=disagreements,
        )))

    total = len(selected)
    for rank, (score, cand) in enumerate(selected, 1):
        fip._print_pair_preference_labels(
            rank, total, records[cand.a_idx], records[cand.b_idx],
            cand.disagreements, score=score,
        )

    snap_path = _save_claude_snapshot(records, selected)
    print(f"\n[snapshot saved: {snap_path}]")
    return 0


def _save_claude_snapshot(records, selected) -> str:
    """Write a snapshot in the same schema find_informative_pairs.py uses,
    tagged as Claude-selected. /repair-verify reads it via --check-snapshots.
    """
    import glob
    import re
    from datetime import datetime

    os.makedirs(HISTORY_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(HISTORY_DIR, "pref_round_*.json"))
    nums = [int(re.search(r"pref_round_(\d+)\.json", p).group(1)) for p in existing]
    snap_id = (max(nums) + 1) if nums else 1
    snap = {
        "snapshot_id": snap_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selection_method": "claude_selected",
        "pairs": [],
    }
    for rank, (score, cand) in enumerate(selected, 1):
        rec_a = records[cand.a_idx]
        rec_b = records[cand.b_idx]
        snap["pairs"].append({
            "rank": rank,
            "score": float(score),
            "traj_a_path": rec_a.path,
            "traj_b_path": rec_b.path,
            "snapshot_reward_a": float(rec_a.designed_reward),
            "snapshot_reward_b": float(rec_b.designed_reward),
            "disagreements": cand.disagreements,
        })
    out_path = os.path.join(HISTORY_DIR, f"pref_round_{snap_id:03d}.json")
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=2)
    return out_path


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
    print(f"[select_pairs] workspace={ws}")

    rounds = None
    if args.rounds:
        rounds = [int(x) for x in args.rounds.split(",")]

    records = _ordered_records(rounds)
    print(f"[select_pairs] loaded {len(records)} trajectories "
          f"from rounds {sorted({r.metadata.get('round') for r in records})}")
    metrics = fip._load_performance_metrics()

    if args.mode == "dump":
        _do_dump(records, metrics)
        return 0
    if not args.selection:
        raise SystemExit("[select_pairs] --mode render requires --selection <file>.")
    return _do_render(records, metrics, args.selection)


if __name__ == "__main__":
    sys.exit(main())
