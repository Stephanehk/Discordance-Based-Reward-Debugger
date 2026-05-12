"""Find trajectory pairs where designed_reward_fn disagrees with a performance metric.

Usage:
    python find_informative_pairs.py --workspace workspaces/<name>
    python find_informative_pairs.py --workspace workspaces/<name> --top-k 5 --margin 0.0
    python find_informative_pairs.py --workspace workspaces/<name> --rounds 1,2
"""
from __future__ import annotations

import argparse
import glob
import importlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime

from runtime.workspace import add_workspace_arg, activate
from runtime.trajectory_io import (
    iter_all_trajectory_files,
    iter_round_trajectory_files,
    load_trajectory,
    list_rounds,
)
from runtime.compute_features import features_path_for


@dataclass
class TrajRecord:
    path: str
    metadata: dict
    trajectory: list           # list of (obs, action, next_obs)
    designed_reward: float
    metric_values: dict        # name -> float
    features: list[float]
    feature_names: list


def _load_designed_reward_fn():
    from designed_reward_fn import designed_reward_fn
    return designed_reward_fn


def _load_performance_metrics() -> list[tuple[str, callable, str]]:
    """Return [(name, fn, docstring), ...] in module order."""
    mod = importlib.import_module("candidate_performance_metrics.performance_metrics")
    metrics = []
    for name, obj in vars(mod).items():
        if inspect.isfunction(obj) and name.startswith("performance_metric_"):
            doc = (obj.__doc__ or "").strip().splitlines()[0] if obj.__doc__ else name
            metrics.append((name, obj, doc))
    if not metrics:
        raise RuntimeError("No performance_metric_* functions found.")
    return metrics


def _read_features_for(traj_path: str):
    feat_path = features_path_for(traj_path)
    if not os.path.exists(feat_path):
        return None, None
    with open(feat_path) as f:
        payload = json.load(f)
    return payload["features"], payload["names"]


def load_records(rounds: list[int] | None) -> list[TrajRecord]:
    if rounds is None:
        paths = list(iter_all_trajectory_files())
    else:
        paths = []
        for k in rounds:
            paths.extend(iter_round_trajectory_files(k))
        paths.sort()

    if not paths:
        raise RuntimeError("No trajectories found. Run /repair-round first.")

    designed_reward_fn = _load_designed_reward_fn()
    metrics = _load_performance_metrics()

    records: list[TrajRecord] = []
    for path in paths:
        data = load_trajectory(path)
        traj = data["trajectory"]
        if not traj:
            continue
        r_total = sum(float(designed_reward_fn(o, a, no)) for (o, a, no) in traj)
        m_vals = {name: float(fn(traj)) for (name, fn, _doc) in metrics}
        feats, names = _read_features_for(path)
        records.append(TrajRecord(
            path=path,
            metadata=data.get("metadata", {}),
            trajectory=traj,
            designed_reward=r_total,
            metric_values=m_vals,
            features=feats or [],
            feature_names=names or [],
        ))
    return records


def _ranks(values: list[float]) -> list[float]:
    """Average-rank, 1-indexed."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


class FindDisagreeingPairs:
    def __init__(self, records: list[TrajRecord], metric_docs: dict[str, str]):
        self.records = records
        self.metric_docs = metric_docs
        self._reward_ranks = _ranks([r.designed_reward for r in records])
        self._metric_ranks = {
            m: _ranks([r.metric_values[m] for r in records])
            for m in next(iter(records)).metric_values
        } if records else {}

    def disagreeing_for(self, metric_name: str, top_k: int, margin: float):
        n = len(self.records)
        if n < 2:
            return []
        r_rank = self._reward_ranks
        m_rank = self._metric_ranks[metric_name]
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                # disagreement score in normalized rank units
                dr = (r_rank[i] - r_rank[j]) / (n - 1)
                dm = (m_rank[i] - m_rank[j]) / (n - 1)
                if dr * dm >= 0:
                    continue
                score = abs(dr) * abs(dm)
                if score < margin * margin:
                    continue
                if dr > 0:
                    pref_reward, pref_metric = i, j
                else:
                    pref_reward, pref_metric = j, i
                candidates.append((score, pref_reward, pref_metric))
        candidates.sort(reverse=True)
        return candidates[:top_k]


def _print_pair(rec_pref_reward: TrajRecord, rec_pref_metric: TrajRecord,
                metric_name: str, metric_doc: str,
                rank: int | None = None, total: int | None = None,
                score: float | None = None):
    """Print one disagreeing pair, side-by-side, with a feature delta column.

    The feature key is printed once at the top of the run (see
    `_print_feature_key`), so per-feature rows here intentionally omit
    the inline description.
    """
    a_r = rec_pref_reward.designed_reward
    b_r = rec_pref_metric.designed_reward
    a_m = rec_pref_reward.metric_values[metric_name]
    b_m = rec_pref_metric.metric_values[metric_name]
    a_meta = rec_pref_reward.metadata
    b_meta = rec_pref_metric.metadata

    print()
    print("═" * 78)
    header = "PAIR"
    if rank is not None and total is not None:
        header = f"PAIR {rank}/{total}"
    print(header)
    print(f"Stakeholder: {metric_doc}")
    print("═" * 78)
    print()
    print(f"  Trajectory A:  {a_meta.get('policy_class', '?')}"
          f"   (round {a_meta.get('round', '?')}, "
          f"{a_meta.get('n_steps', len(rec_pref_reward.trajectory))} steps)")
    print(f"  Trajectory B:  {b_meta.get('policy_class', '?')}"
          f"   (round {b_meta.get('round', '?')}, "
          f"{b_meta.get('n_steps', len(rec_pref_metric.trajectory))} steps)")
    print()
    print(f"                                         {'Trajectory A':>14s}   {'Trajectory B':>14s}")
    print(f"  designed reward (rewards A ≻ B)        {a_r:>+14.3f}   {b_r:>+14.3f}")
    print(f"  stakeholder metric (prefers B ≻ A)     {a_m:>+14.3f}   {b_m:>+14.3f}")
    print()
    print(f"  → preference disagrees   "
          f"(reward says A ≻ B; stakeholder says B ≻ A)")
    print()
    if rec_pref_reward.features and rec_pref_metric.features and rec_pref_reward.feature_names:
        print(f"  Trajectory features (Δ = B − A; see feature key above):")
        print(f"    {'feature':28s}  {'A':>10s}   {'B':>10s}   {'Δ (B−A)':>10s}")
        print(f"    {'─' * 64}")
        for (short, _desc), va, vb in zip(rec_pref_reward.feature_names,
                                          rec_pref_reward.features,
                                          rec_pref_metric.features):
            d = vb - va
            print(f"    {short:28s}  {va:>+10.2f}   {vb:>+10.2f}   {d:>+10.2f}")
    else:
        print("  (no features on disk for one of these trajectories)")


def _print_feature_key(feature_names) -> None:
    """Print a brief key explaining each feature short_name."""
    if not feature_names:
        return
    print()
    print("─" * 78)
    print("Feature key (short_name — what it measures):")
    print("─" * 78)
    for short, desc in feature_names:
        print(f"  {short:24s}   {desc}")


def _strip_policy_prefix(name: str) -> str:
    """Drop the `DataCollectionPolicy` prefix for readability."""
    return name.removeprefix("DataCollectionPolicy") if name else name


def _print_pair_preference_labels(rank: int, total: int,
                                  rec_pref_reward: TrajRecord,
                                  rec_pref_metric: TrajRecord,
                                  metric_name: str, metric_doc: str,
                                  score: float | None = None) -> None:
    """Compact preference-label view: explicit A≻B labels, side-by-side features.

    Metric short-names are intentionally anonymized as "candidate
    performance metric" — only the stakeholder docstring is shown — so
    the user reads the perspective in plain language.
    """
    a_r = rec_pref_reward.designed_reward
    b_r = rec_pref_metric.designed_reward
    a_m = rec_pref_reward.metric_values[metric_name]
    b_m = rec_pref_metric.metric_values[metric_name]
    a_meta = rec_pref_reward.metadata
    b_meta = rec_pref_metric.metadata

    print()
    print("═" * 78)
    print(f"PAIR {rank}/{total}")
    print(f"Stakeholder: {metric_doc}")
    print("─" * 78)
    a_name = _strip_policy_prefix(a_meta.get('policy_class', '?'))
    b_name = _strip_policy_prefix(b_meta.get('policy_class', '?'))
    print(f"  Trajectory A:  {a_name}   (round {a_meta.get('round', '?')}, "
          f"{a_meta.get('n_steps', len(rec_pref_reward.trajectory))} steps)")
    print(f"  Trajectory B:  {b_name}   (round {b_meta.get('round', '?')}, "
          f"{b_meta.get('n_steps', len(rec_pref_metric.trajectory))} steps)")
    print()
    print(f"                                  {'Trajectory A':>14s}   {'Trajectory B':>14s}")
    print(f"  designed reward                 {a_r:>+14.3f}   {b_r:>+14.3f}")
    print(f"  {'candidate performance metric':30s}  {a_m:>+14.3f}   {b_m:>+14.3f}")
    print()
    print(f"  preference by designed reward         →   A ≻ B")
    print(f"  preference by candidate metric        →   B ≻ A   ✗ disagrees")
    print()
    if rec_pref_reward.features and rec_pref_metric.features:
        print(f"  Trajectory features:")
        print(f"    {'feature':24s}   {'A':>10s}   {'B':>10s}   {'Δ (B−A)':>10s}")
        print(f"    {'─' * 60}")
        for (short, _desc), va, vb in zip(rec_pref_reward.feature_names,
                                          rec_pref_reward.features,
                                          rec_pref_metric.features):
            d = vb - va
            print(f"    {short:24s}   {va:>+10.2f}   {vb:>+10.2f}   {d:>+10.2f}")
    else:
        print("  (no features on disk for these trajectories)")


HISTORY_DIR = ".repair_history"


def _next_snapshot_id() -> int:
    if not os.path.isdir(HISTORY_DIR):
        return 1
    existing = glob.glob(os.path.join(HISTORY_DIR, "pref_round_*.json"))
    nums = [int(re.search(r"pref_round_(\d+)\.json", p).group(1)) for p in existing]
    return (max(nums) + 1) if nums else 1


def _save_snapshot(records: list[TrajRecord],
                   pairs: list[tuple]) -> str:
    """Save the top-N preference-labels pairs to a JSON snapshot.

    Each pair is (score, metric_name, metric_doc, i_pref_reward, j_pref_metric)
    in the convention used by --top-overall: i is the reward-preferring side
    (printed as "A"), j is the metric-preferring side (printed as "B").
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    snap_id = _next_snapshot_id()
    snap = {
        "snapshot_id": snap_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pairs": [],
    }
    for rank, (score, name, doc, i, j) in enumerate(pairs, 1):
        rec_a = records[i]
        rec_b = records[j]
        snap["pairs"].append({
            "rank": rank,
            "score": float(score),
            "metric_name": name,
            "metric_doc": doc,
            "traj_a_path": rec_a.path,
            "traj_b_path": rec_b.path,
            "stakeholder_prefers": "B",
            "snapshot_reward_a": float(rec_a.designed_reward),
            "snapshot_reward_b": float(rec_b.designed_reward),
            "snapshot_metric_a": float(rec_a.metric_values[name]),
            "snapshot_metric_b": float(rec_b.metric_values[name]),
        })
    out_path = os.path.join(HISTORY_DIR, f"pref_round_{snap_id:03d}.json")
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=2)
    return out_path


def _check_snapshots(records: list[TrajRecord]) -> int:
    """Re-verify every past snapshot under the current designed reward.

    Returns 0 if all past disagreements are resolved, 1 if any still
    disagree (the unresolved pairs are pretty-printed).
    """
    if not os.path.isdir(HISTORY_DIR):
        print("No past disagreement snapshots have been saved for this workspace yet.")
        return 0
    snap_paths = sorted(glob.glob(os.path.join(HISTORY_DIR, "pref_round_*.json")))
    if not snap_paths:
        print("No past disagreement snapshots have been saved for this workspace yet.")
        return 0

    rec_by_path = {r.path: r for r in records}
    unresolved: list[dict] = []
    total = 0
    missing = 0
    for snap_path in snap_paths:
        with open(snap_path) as f:
            snap = json.load(f)
        for pair in snap["pairs"]:
            total += 1
            rec_a = rec_by_path.get(pair["traj_a_path"])
            rec_b = rec_by_path.get(pair["traj_b_path"])
            if rec_a is None or rec_b is None:
                missing += 1
                continue
            r_a = rec_a.designed_reward
            r_b = rec_b.designed_reward
            # Pair was "reward prefers A". "Resolved" = reward no longer prefers A
            # (either prefers B or ties).
            if r_a > r_b:
                unresolved.append({
                    "snapshot_id": snap["snapshot_id"],
                    "pair": pair,
                    "rec_a": rec_a,
                    "rec_b": rec_b,
                })

    checked = total - missing
    resolved = checked - len(unresolved)
    print(f"Checked {checked} pair(s) from {len(snap_paths)} past snapshot(s).")
    if missing:
        print(f"  ({missing} pair(s) skipped — underlying trajectories no longer on disk.)")
    print(f"  Resolved by current reward:   {resolved}")
    print(f"  Still disagreeing:            {len(unresolved)}")

    if not unresolved:
        print("\nAll past disagreements are now resolved by the current reward.")
        return 0

    print(f"\nThe following {len(unresolved)} pair(s) still disagree under the "
          f"current reward function:")
    total_u = len(unresolved)
    for idx, u in enumerate(unresolved, 1):
        _print_pair_preference_labels(idx, total_u,
                                       u["rec_a"], u["rec_b"],
                                       u["pair"]["metric_name"],
                                       u["pair"]["metric_doc"])
    return 1


def main():
    p = argparse.ArgumentParser()
    add_workspace_arg(p)
    p.add_argument("--top-k", type=int, default=5,
                   help="Pairs per metric. Ignored when --top-overall is set.")
    p.add_argument("--top-overall", type=int, default=None,
                   help="If set, print the top N disagreements across all "
                        "metrics. Overrides --top-k.")
    p.add_argument("--format", choices=["default", "preference-labels"],
                   default="default",
                   help="default: full per-trajectory feature dumps. "
                        "preference-labels: compact A≻B / B≻A view with a "
                        "side-by-side feature delta table.")
    p.add_argument("--margin", type=float, default=0.0,
                   help="Minimum normalized rank-gap geometric mean (0..1).")
    p.add_argument("--rounds", type=str, default=None,
                   help="Comma-separated round numbers; default = all.")
    p.add_argument("--check-snapshots", action="store_true",
                   help="Instead of finding new disagreements, re-verify every "
                        "past preference-labels snapshot under the current reward "
                        "function and print any pairs that still disagree.")
    p.add_argument("--no-save-snapshot", action="store_true",
                   help="Suppress the automatic snapshot save in --format "
                        "preference-labels mode.")
    args = p.parse_args()

    ws = activate(args.workspace)
    print(f"[find_informative_pairs] workspace={ws}")

    rounds = None
    if args.rounds:
        rounds = [int(x) for x in args.rounds.split(",")]

    records = load_records(rounds)
    print(f"[find_informative_pairs] loaded {len(records)} trajectories "
          f"from rounds {sorted({r.metadata.get('round') for r in records})}")

    metrics = _load_performance_metrics()
    metric_docs = {name: doc for (name, _fn, doc) in metrics}

    if args.check_snapshots:
        return _check_snapshots(records)

    finder = FindDisagreeingPairs(records, metric_docs)

    if args.top_overall is not None:
        all_pairs: list[tuple[float, str, str, int, int]] = []  # (score, metric, doc, i, j)
        per_metric_top = max(args.top_overall * 3, 30)
        for (name, _fn, doc) in metrics:
            for score, i, j in finder.disagreeing_for(name, top_k=per_metric_top,
                                                      margin=args.margin):
                all_pairs.append((score, name, doc, i, j))
        all_pairs.sort(key=lambda x: x[0], reverse=True)

        if not all_pairs:
            print("\nNo disagreements found across any metric.")
            return 0

        # Dedup by (stakeholder, policy_class_A, policy_class_B): keep the
        # highest-scoring rollout pair per triple, so the top-N covers
        # genuinely different stakeholders / policy-pair comparisons
        # instead of 5 copies of the loudest one.
        seen_triples: set[tuple[str, str, str]] = set()
        deduped: list[tuple[float, str, str, int, int]] = []
        for entry in all_pairs:
            _score, name, _doc, i, j = entry
            triple = (
                name,
                records[i].metadata.get("policy_class", "?"),
                records[j].metadata.get("policy_class", "?"),
            )
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            deduped.append(entry)
        all_pairs = deduped

        kept = all_pairs[:args.top_overall]
        total = len(kept)
        if args.format == "preference-labels":
            print(f"\nTop {total} preference-label disagreements across all stakeholders:")
            feature_names = records[0].feature_names if records else []
            _print_feature_key(feature_names)
            for rank, (_score, name, doc, i, j) in enumerate(kept, 1):
                _print_pair_preference_labels(rank, total, records[i], records[j],
                                              name, doc, score=_score)
            if not args.no_save_snapshot and kept:
                snap_path = _save_snapshot(records, kept)
                print(f"\n[snapshot saved: {snap_path}]")
        else:
            print(f"\nTop {total} disagreements across all metrics.")
            feature_names = records[0].feature_names if records else []
            _print_feature_key(feature_names)
            for rank, (score, name, doc, i, j) in enumerate(kept, 1):
                _print_pair(records[i], records[j], name, doc,
                            rank=rank, total=total, score=score)
        return 0

    any_printed = False
    feature_names = records[0].feature_names if records else []
    _print_feature_key(feature_names)
    for (name, _fn, doc) in metrics:
        pairs = finder.disagreeing_for(name, top_k=args.top_k, margin=args.margin)
        print()
        print("═" * 78)
        print(f"== {name}")
        print(f"   {doc}")
        print("═" * 78)
        if not pairs:
            print("  (no disagreements found at the requested margin)")
            continue
        any_printed = True
        for rank, (score, i, j) in enumerate(pairs, 1):
            _print_pair(records[i], records[j], name, doc,
                        rank=rank, total=len(pairs), score=score)

    if not any_printed:
        print("\nNo disagreements found across any metric. Either the reward and "
              "metrics agree everywhere, or rollouts are not diverse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
