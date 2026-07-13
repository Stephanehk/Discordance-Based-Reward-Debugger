"""Find trajectory pairs where designed_reward_fn disagrees with a performance metric.

Selection algorithm (--top-overall mode):
  1. Build the *disagreeing-pair set*: all unordered trajectory pairs (i, j)
     for which the designed reward and at least one performance metric rank
     the two trajectories in opposite orders (rank-product check, with
     optional --margin threshold per metric).
  2. Greedy informative selection on that set. Each pair contributes a
     feature delta x = φ(B) − φ(A), where A is the reward-preferred side.
     Pick pairs one at a time, each time choosing
         argmax_x  x^T V_t^{-1} x
     where  V_1 = κλ · I_d  and  V_{t+1} = V_t + x_t x_t^T.
     This is the elliptic-potential / G-optimal active-query criterion
     (cf. arxiv 2111.04850): each new pick prefers a direction in
     feature-delta space that the prior picks have not yet covered.

Usage:
    python find_informative_pairs.py --workspace workspaces/<name>
    python find_informative_pairs.py --workspace workspaces/<name> --top-overall 5
    python find_informative_pairs.py --workspace workspaces/<name> --rounds 1,2
    python find_informative_pairs.py --workspace workspaces/<name> \
           --top-overall 5 --kappa-lambda 0.5
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

import numpy as np

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


@dataclass
class PairCandidate:
    """A disagreeing trajectory pair. A is the reward-preferred side."""
    a_idx: int                  # index into records (reward prefers a_idx)
    b_idx: int
    delta: np.ndarray           # φ(B) − φ(A)
    disagreements: list[dict]   # [{perspective, metric_a, metric_b}, ...]


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


def _build_disagreeing_candidates(
    records: list[TrajRecord],
    metrics: list[tuple[str, callable, str]],
    margin: float,
) -> list[PairCandidate]:
    """All pairs where the reward and ≥1 metric disagree on the ranking.

    Pair orientation: A is the side the reward prefers (higher reward rank).
    Tied-reward pairs are skipped. Pairs missing features on either side
    are skipped (no φ → no Mahalanobis score).

    `margin` is the geometric-mean rank-gap threshold per metric: a metric
    is counted as disagreeing only if |Δrank_reward| · |Δrank_metric| ≥ margin².
    """
    n = len(records)
    if n < 2:
        return []
    metric_names = [m[0] for m in metrics]
    metric_docs_by_name = {m[0]: m[2] for m in metrics}
    reward_ranks = _ranks([r.designed_reward for r in records])
    metric_ranks_by_name = {
        name: _ranks([r.metric_values[name] for r in records])
        for name in metric_names
    }
    inv_n = 1.0 / max(n - 1, 1)
    margin_sq = margin * margin

    out: list[PairCandidate] = []
    for i in range(n):
        for j in range(i + 1, n):
            dr = (reward_ranks[i] - reward_ranks[j]) * inv_n
            if dr == 0:
                continue
            if dr > 0:
                a_idx, b_idx, dr_ab = i, j, dr
            else:
                a_idx, b_idx, dr_ab = j, i, -dr
            # dr_ab > 0; reward prefers A over B.

            disagreements: list[dict] = []
            for name in metric_names:
                m_rank = metric_ranks_by_name[name]
                dm_ab = (m_rank[a_idx] - m_rank[b_idx]) * inv_n
                if dm_ab >= 0:
                    continue  # metric agrees with reward, or ties
                if dr_ab * (-dm_ab) < margin_sq:
                    continue  # disagreement too small to count
                disagreements.append({
                    "perspective": metric_docs_by_name[name],
                    "metric_a": float(records[a_idx].metric_values[name]),
                    "metric_b": float(records[b_idx].metric_values[name]),
                })
            if not disagreements:
                continue

            rec_a = records[a_idx]
            rec_b = records[b_idx]
            if not rec_a.features or not rec_b.features:
                continue
            delta = (np.asarray(rec_b.features, dtype=float)
                     - np.asarray(rec_a.features, dtype=float))
            out.append(PairCandidate(
                a_idx=a_idx, b_idx=b_idx,
                delta=delta, disagreements=disagreements,
            ))
    return out


def _greedy_mahalanobis_select(
    deltas: np.ndarray, n: int, kappa_lambda: float,
) -> list[tuple[float, int]]:
    """Greedy selection of up to n rows of `deltas` maximizing x^T V_t^{-1} x.

    V_1 = kappa_lambda * I_d. At each step the row with the highest score is
    chosen; V_{t+1} = V_t + x_t x_t^T. Returns [(score_at_pick, row_index), …]
    in selection order.
    """
    m = deltas.shape[0]
    if m == 0:
        return []
    d = deltas.shape[1]
    V = float(kappa_lambda) * np.eye(d)
    available = np.ones(m, dtype=bool)
    selected: list[tuple[float, int]] = []
    pick_n = min(n, m)
    for _ in range(pick_n):
        V_inv = np.linalg.inv(V)
        proj = deltas @ V_inv
        scores = np.einsum('ij,ij->i', proj, deltas)
        scores_masked = np.where(available, scores, -np.inf)
        idx = int(np.argmax(scores_masked))
        score = float(scores_masked[idx])
        if not np.isfinite(score):
            break
        selected.append((score, idx))
        available[idx] = False
        x = deltas[idx]
        V = V + np.outer(x, x)
    return selected


class FindDisagreeingPairs:
    """Legacy per-metric rank-product disagreement finder.

    Used only by the default (non --top-overall) output mode below. The
    primary selection algorithm is _build_disagreeing_candidates +
    _greedy_mahalanobis_select.
    """
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


def _print_pair(rec_a: TrajRecord, rec_b: TrajRecord,
                disagreements: list[dict],
                rank: int | None = None, total: int | None = None,
                score: float | None = None):
    """Full per-trajectory feature dump format.

    A is the reward-preferred trajectory; `disagreements` is the list of
    other perspectives whose ranking prefers B over A. The feature key is
    printed once at the top of the run (see `_print_feature_key`).
    """
    a_r = rec_a.designed_reward
    b_r = rec_b.designed_reward
    a_meta = rec_a.metadata
    b_meta = rec_b.metadata

    print()
    print("═" * 78)
    header = "PAIR"
    if rank is not None and total is not None:
        header = f"PAIR {rank}/{total}"
    print(header)
    print("═" * 78)
    print()
    print(f"  Trajectory A")
    print(f"  Trajectory B")
    print()
    print(f"                                         {'Trajectory A':>14s}   {'Trajectory B':>14s}")
    print(f"  designed reward (prefers A ≻ B)        {a_r:>+14.3f}   {b_r:>+14.3f}")
    print()
    print(f"  Other perspectives that disagree (prefer B ≻ A):")
    for d in disagreements:
        print(f"    • {d['perspective']}")
        print(f"        A={d['metric_a']:+10.3f}   B={d['metric_b']:+10.3f}")
    print()
    if rec_a.features and rec_b.features and rec_a.feature_names:
        print(f"  Trajectory features (Δ = B − A; see feature key above):")
        print(f"    {'feature':28s}  {'A':>10s}   {'B':>10s}   {'Δ (B−A)':>10s}")
        print(f"    {'─' * 64}")
        for (short, _desc), va, vb in zip(rec_a.feature_names,
                                          rec_a.features,
                                          rec_b.features):
            d_v = vb - va
            print(f"    {short:28s}  {va:>+10.2f}   {vb:>+10.2f}   {d_v:>+10.2f}")
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


def _fmt_value(v) -> str:
    """Compact display of a feature value: integers plain, others 2 decimals."""
    v = float(v)
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


_FEATURE_UNITS_CACHE = None


def _feature_units_map() -> dict:
    """Map short_name -> unit string, read from the workspace's optional
    `Trajectory2Features.feature_units`. Returns {} if none is declared."""
    global _FEATURE_UNITS_CACHE
    if _FEATURE_UNITS_CACHE is not None:
        return _FEATURE_UNITS_CACHE
    units: dict = {}
    try:
        from gather_evidence.trajectory_features import Trajectory2Features
        names = getattr(Trajectory2Features, "feature_names", [])
        unit_list = getattr(Trajectory2Features, "feature_units", None)
        if unit_list:
            for (short, _desc), u in zip(names, unit_list):
                units[short] = u
    except Exception:
        pass
    _FEATURE_UNITS_CACHE = units
    return units


def _print_pair_preference_labels(rank: int, total: int,
                                  rec_a: TrajRecord, rec_b: TrajRecord,
                                  disagreements: list[dict],
                                  score: float | None = None,
                                  order: list | None = None) -> None:
    """One sentence per characteristic contrasting the reward-preferred
    trajectory A ("Designed reward function's preferred trajectory") with B
    ("the alternative trajectory"). Perspective metrics are used only for
    under-the-hood selection and are never shown to the user.

    `order` is an optional list of characteristic short_names, ranked
    most→least informative about why the reward's preference may be wrong;
    listed characteristics are shown first in that order, the rest after.
    """
    if not (rec_a.features and rec_b.features):
        print("(no characteristics recorded for these trajectories)")
        return
    units = _feature_units_map()
    rows = list(zip(rec_a.feature_names, rec_a.features, rec_b.features))
    if order:
        rank_of = {short: i for i, short in enumerate(order)}
        # Stable sort: listed characteristics by their rank, the rest keep
        # their original order after them.
        rows.sort(key=lambda t: rank_of.get(t[0][0], len(order)))
    # First line is the lead-in header; each following line is a bulleted item
    # (the web UI renders line 1 as a heading and the rest as the purple boxes).
    print("Designed reward function's preferred trajectory has a")
    for (short, desc), va, vb in rows:
        label = desc.rstrip(". ")
        # Lowercase the leading capital so it reads naturally, but leave
        # acronyms (e.g. "CO2") alone.
        if label[:2] != label[:2].upper():
            label = label[0].lower() + label[1:]
        unit = units.get(short, "")
        suffix = f", {unit}" if unit else ""
        vals = f"({_fmt_value(va)} vs. {_fmt_value(vb)}{suffix})"
        word = "higher" if va > vb else "lower" if va < vb else "same"
        print(f"**{word}** {label} {vals}")


def _print_trajectory_features_block(rec: TrajRecord, indent: str = "    ") -> None:
    """Print one trajectory's features as a small (feature, value) table."""
    if not rec.features:
        print(f"{indent}(no features on disk for this trajectory)")
        return
    print(f"{indent}{'feature':24s}   {'value':>10s}")
    print(f"{indent}{'─' * 40}")
    for (short, _desc), v in zip(rec.feature_names, rec.features):
        print(f"{indent}{short:24s}   {v:>+10.2f}")


def _print_lottery_preference_labels(rank: int, total: int,
                                     rec_c: TrajRecord,
                                     rec_a: TrajRecord,
                                     rec_b: TrajRecord,
                                     p: float,
                                     reward_prefers_c: bool,
                                     r_c: float,
                                     r_lottery: float,
                                     disagreements: list[dict],
                                     justification: str | None = None) -> None:
    """Lottery comparison: certain trajectory C vs (p·A, (1-p)·B).

    The user-facing description (a plain-English summary of how the certain
    trajectory C and the lottery's branches A and B differ) is authored by
    the skill and posted via `--content`, so nothing is printed here.
    Perspective metrics are used only for under-the-hood selection.
    """
    return


HISTORY_DIR = ".repair_history"


def _next_snapshot_id() -> int:
    if not os.path.isdir(HISTORY_DIR):
        return 1
    existing = glob.glob(os.path.join(HISTORY_DIR, "pref_round_*.json"))
    nums = [int(re.search(r"pref_round_(\d+)\.json", p).group(1)) for p in existing]
    return (max(nums) + 1) if nums else 1


def _save_snapshot(records: list[TrajRecord],
                   selected: list[tuple[float, PairCandidate]],
                   kappa_lambda: float) -> str:
    """Save the selected pairs to a JSON snapshot.

    Convention: pair["traj_a_path"] is the reward-preferred trajectory and
    pair["disagreements"] is the list of other perspectives that prefer B
    over A on this pair. /repair-verify rechecks resolution by reward
    direction alone.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    snap_id = _next_snapshot_id()
    snap = {
        "snapshot_id": snap_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selection_method": "mahalanobis_greedy_vt_inv",
        "kappa_lambda": float(kappa_lambda),
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


def _pair_disagreements_from_snapshot(pair: dict) -> list[dict]:
    """Schema-tolerant extraction of disagreements from a snapshot pair.

    New schema stores `disagreements`; legacy snapshots stored a single
    metric_doc + per-metric scores. Either way return a list[dict] with
    `perspective`, `metric_a`, `metric_b` keys.
    """
    if "disagreements" in pair:
        return pair["disagreements"]
    return [{
        "perspective": pair.get("metric_doc") or pair.get("metric_name") or "",
        "metric_a": float(pair.get("snapshot_metric_a", 0.0)),
        "metric_b": float(pair.get("snapshot_metric_b", 0.0)),
    }]


def _check_snapshots(records: list[TrajRecord]) -> int:
    """Re-verify every past snapshot under the current designed reward.

    Returns 0 if all past disagreements are resolved, 1 if any still
    disagree (the unresolved pairs and lotteries are pretty-printed).
    """
    if not os.path.isdir(HISTORY_DIR):
        print("No past disagreement snapshots have been saved for this workspace yet.")
        return 0
    snap_paths = sorted(glob.glob(os.path.join(HISTORY_DIR, "pref_round_*.json")))
    if not snap_paths:
        print("No past disagreement snapshots have been saved for this workspace yet.")
        return 0

    rec_by_path = {r.path: r for r in records}
    unresolved_pairs: list[dict] = []
    unresolved_lotteries: list[dict] = []
    pair_total = 0
    pair_missing = 0
    lottery_total = 0
    lottery_missing = 0
    for snap_path in snap_paths:
        with open(snap_path) as f:
            snap = json.load(f)
        for pair in snap.get("pairs", []):
            pair_total += 1
            rec_a = rec_by_path.get(pair["traj_a_path"])
            rec_b = rec_by_path.get(pair["traj_b_path"])
            if rec_a is None or rec_b is None:
                pair_missing += 1
                continue
            r_a = rec_a.designed_reward
            r_b = rec_b.designed_reward
            # Pair was "reward prefers A". "Resolved" = reward no longer prefers A
            # (either prefers B or ties).
            if r_a > r_b:
                unresolved_pairs.append({
                    "snapshot_id": snap["snapshot_id"],
                    "rec_a": rec_a,
                    "rec_b": rec_b,
                    "disagreements": _pair_disagreements_from_snapshot(pair),
                })
        for lot in snap.get("lotteries", []):
            lottery_total += 1
            rec_c = rec_by_path.get(lot["traj_c_path"])
            rec_a = rec_by_path.get(lot["traj_a_path"])
            rec_b = rec_by_path.get(lot["traj_b_path"])
            if rec_c is None or rec_a is None or rec_b is None:
                lottery_missing += 1
                continue
            p = float(lot["p"])
            r_c = rec_c.designed_reward
            r_lottery = p * rec_a.designed_reward + (1.0 - p) * rec_b.designed_reward
            reward_prefers_c_now = r_c >= r_lottery
            snapshot_prefers_c = bool(lot["snapshot_reward_prefers_c"])
            if reward_prefers_c_now == snapshot_prefers_c:
                # Reward kept its original preference → unresolved.
                unresolved_lotteries.append({
                    "snapshot_id": snap["snapshot_id"],
                    "rec_c": rec_c,
                    "rec_a": rec_a,
                    "rec_b": rec_b,
                    "p": p,
                    "reward_prefers_c": reward_prefers_c_now,
                    "r_c": r_c,
                    "r_lottery": r_lottery,
                    "disagreements": lot.get("disagreements", []),
                    "justification": lot.get("justification"),
                })

    checked_pairs = pair_total - pair_missing
    resolved_pairs = checked_pairs - len(unresolved_pairs)
    checked_lotteries = lottery_total - lottery_missing
    resolved_lotteries = checked_lotteries - len(unresolved_lotteries)

    print(f"Checked {checked_pairs} pair(s) and {checked_lotteries} lottery comparison(s) "
          f"from {len(snap_paths)} past snapshot(s).")
    if pair_missing:
        print(f"  ({pair_missing} pair(s) skipped — underlying trajectories no longer on disk.)")
    if lottery_missing:
        print(f"  ({lottery_missing} lottery(ies) skipped — underlying trajectories no longer on disk.)")
    print(f"  Pairs      resolved: {resolved_pairs:>3d}   still disagreeing: {len(unresolved_pairs)}")
    print(f"  Lotteries  resolved: {resolved_lotteries:>3d}   still disagreeing: {len(unresolved_lotteries)}")

    if not unresolved_pairs and not unresolved_lotteries:
        print("\nAll past disagreements are now resolved by the current reward.")
        return 0

    if unresolved_pairs:
        print(f"\nThe following {len(unresolved_pairs)} pair(s) still disagree under the "
              f"current reward function:")
        total_u = len(unresolved_pairs)
        for idx, u in enumerate(unresolved_pairs, 1):
            _print_pair_preference_labels(idx, total_u,
                                           u["rec_a"], u["rec_b"],
                                           u["disagreements"])
    if unresolved_lotteries:
        print(f"\nThe following {len(unresolved_lotteries)} lottery comparison(s) still "
              f"disagree under the current reward function:")
        total_u = len(unresolved_lotteries)
        for idx, u in enumerate(unresolved_lotteries, 1):
            _print_lottery_preference_labels(
                idx, total_u,
                u["rec_c"], u["rec_a"], u["rec_b"], u["p"],
                u["reward_prefers_c"], u["r_c"], u["r_lottery"],
                u["disagreements"],
                justification=u.get("justification"),
            )
    return 1


def main():
    p = argparse.ArgumentParser()
    add_workspace_arg(p)
    p.add_argument("--top-k", type=int, default=5,
                   help="Pairs per metric (legacy per-metric mode). "
                        "Ignored when --top-overall is set.")
    p.add_argument("--top-overall", type=int, default=None,
                   help="Number of pairs to surface via Mahalanobis-greedy "
                        "selection across all perspectives. Overrides --top-k.")
    p.add_argument("--format", choices=["default", "preference-labels"],
                   default="default",
                   help="default: full per-trajectory feature dumps. "
                        "preference-labels: compact A≻B view with a "
                        "side-by-side feature delta table.")
    p.add_argument("--margin", type=float, default=0.0,
                   help="Minimum |Δrank_reward|·|Δrank_metric| (geometric-mean "
                        "rank-gap squared) for a metric to count as "
                        "disagreeing on a pair. 0..1.")
    p.add_argument("--kappa-lambda", type=float, default=1.0,
                   help="Ridge scalar κλ for V_1 = κλ·I in the "
                        "Mahalanobis-greedy selection.")
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

    if args.top_overall is not None:
        candidates = _build_disagreeing_candidates(records, metrics, args.margin)
        if not candidates:
            print("\nNo disagreeing pairs found (no pair has another perspective "
                  "whose ranking opposes the reward at the requested margin).")
            return 0
        deltas = np.stack([c.delta for c in candidates], axis=0)
        picks = _greedy_mahalanobis_select(deltas, args.top_overall,
                                           args.kappa_lambda)
        if not picks:
            print("\nNo pairs selected.")
            return 0

        selected: list[tuple[float, PairCandidate]] = [
            (score, candidates[idx]) for (score, idx) in picks
        ]
        total = len(selected)
        feature_names = records[0].feature_names if records else []
        _print_feature_key(feature_names)

        if args.format == "preference-labels":
            print(f"\nTop {total} preference-label disagreements "
                  f"(Mahalanobis-greedy, κλ={args.kappa_lambda}):")
            for rank, (score, cand) in enumerate(selected, 1):
                _print_pair_preference_labels(
                    rank, total,
                    records[cand.a_idx], records[cand.b_idx],
                    cand.disagreements, score=score,
                )
            if not args.no_save_snapshot:
                snap_path = _save_snapshot(records, selected, args.kappa_lambda)
                print(f"\n[snapshot saved: {snap_path}]")
        else:
            print(f"\nTop {total} disagreements "
                  f"(Mahalanobis-greedy, κλ={args.kappa_lambda}).")
            for rank, (score, cand) in enumerate(selected, 1):
                _print_pair(
                    records[cand.a_idx], records[cand.b_idx],
                    cand.disagreements,
                    rank=rank, total=total, score=score,
                )
        return 0

    # ─── Legacy per-metric mode (kept for back-compat / debugging) ──────
    finder = FindDisagreeingPairs(records, metric_docs)
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
            rec_a = records[i]
            rec_b = records[j]
            disag = [{
                "perspective": doc,
                "metric_a": float(rec_a.metric_values[name]),
                "metric_b": float(rec_b.metric_values[name]),
            }]
            _print_pair(rec_a, rec_b, disag,
                        rank=rank, total=len(pairs), score=score)

    if not any_printed:
        print("\nNo disagreements found across any metric. Either the reward and "
              "metrics agree everywhere, or rollouts are not diverse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
