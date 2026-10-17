"""Claude-driven trajectory-pair selection (replaces the algorithmic
selector in find_informative_pairs.py for the /repair-preferences step).

The old selector (find_informative_pairs.py, kept for reference) chose
pairs with a Mahalanobis-greedy criterion over feature deltas. This
script instead exposes the raw material — every trajectory summarised by
its features, its designed-reward total, and every alternative-perspective
metric — and lets Claude pick the pairs that will be most useful to show
the human. Claude reads the table, writes a small selection file naming
the pairs it wants, and this script renders them in the exact same
preference-label format and writes the same snapshot schema, so
/repair-verify is unaffected.

Two modes:

  --mode dump
      Load every trajectory, compute its designed-reward total and each
      alternative-perspective metric, assign each a stable id (T001, T002,
      ...), and write a human-legible table to
      .repair_history/_trajectory_table.txt (plus a machine pool
      .repair_history/_pair_pool.json mapping ids to paths). Claude reads
      the table to decide which pairs to surface.

  --mode render --selection <file.json>
      Read a selection file of the form
          {
            "pairs":     [{"a": "T012", "b": "T034"}, ...],            # ≤ 5
            "lotteries": [{"c": "T020", "a": "T005", "b": "T088",      # ≤ 2
                            "p": 0.5, "justification": "..."}, ...]
          }
      For each pair, orient A = the higher-designed-reward side (the
      reward-preferred trajectory, matching the snapshot/verify contract),
      list the other perspectives that prefer B, and print the
      preference-label block. For each lottery, compute return(C) vs
      p·return(A)+(1-p)·return(B) under the designed reward and each
      alternative-perspective metric, then render the lottery as a
      question to the user (C vs the lottery), listing the perspectives
      that disagree with the reward's preference. Snapshots both kinds
      under .repair_history/ so /repair-verify can recheck them later.

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
# Render mode writes a ready-to-post bundle here so the skill never has to
# hand-build the per-card `--viz` JSON, feature-key JSON, or content files.
UI_PAYLOAD_DIR = os.path.join(HISTORY_DIR, "_ui_payload")


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
    lines.append("each alternative-perspective metric. Pick the top 5 most informative")
    lines.append("pairs (by id) — pairs where the designed reward ranks the two")
    lines.append("trajectories oppositely to another perspective the human cares about,")
    lines.append("or that otherwise best expose how the reward trades off the outcomes")
    lines.append("that matter. Also pick the top 2 most informative lottery comparisons")
    lines.append("{c, a, b, p}, where a 'certain' trajectory C is compared against a")
    lines.append("p-vs-(1-p) lottery over A (good branch) and B (bad branch); lotteries")
    lines.append("surface curvature and risk-attitude disagreements that simple pairs")
    lines.append("miss. Lottery p must be in [0, 1] and Claude must justify its choice.")
    lines.append("")
    lines.append("Pick p so the choice between C and the lottery is genuinely")
    lines.append("difficult — don't default to a fixed number. Compute the designed-")
    lines.append("reward indifference point p* = (return(C) - return(B)) /")
    lines.append("(return(A) - return(B)), then move slightly off it (usually onto the")
    lines.append("C-preferring side) so the reward gives a small-but-real preference")
    lines.append("that a human might intuitively reverse. The exact (1-p) depends on")
    lines.append("the reward magnitudes in the picked trajectories — sometimes very")
    lines.append("small, sometimes not. Justify the chosen p in the selection file.")
    lines.append("If picking two lotteries, prefer two different shapes (different")
    lines.append("C/A/B or a different trade-off) over two values of p on the same")
    lines.append("shape — sweeping p on one shape is a single question rendered twice.")
    lines.append("")
    lines.append("Favor diversity across the comparisons — don't pad with near-")
    lines.append("duplicates.")
    lines.append("")
    lines.append("Feature key (short_name — what it measures):")
    for short, desc in feature_names:
        lines.append(f"  {short:<18} {desc}")
    lines.append("")
    lines.append("Other perspectives (higher = better by that perspective's lights):")
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
    print(f'               {{"pairs":     [{{"a": "T012", "b": "T034"}}, ...]   (≤ 5)')
    print(f'                "lotteries": [{{"c": "T020", "a": "T005", "b": "T088",')
    print(f'                                "p": 0.5, "justification": "..."}}, ...]   (≤ 2)}}')
    print(f"               and run --mode render --selection <that file>.")


MAX_PAIRS = 5
MAX_LOTTERIES = 2


def _build_pair(rec_by_id, records, entry, metric_items):
    """Resolve a pair entry to a (PairCandidate-shaped) record.

    Orients A = the reward-preferred side (snapshot/verify contract) and
    collects the other perspectives that prefer B.
    """
    import numpy as np

    a_id, b_id = entry["a"], entry["b"]
    if a_id not in rec_by_id or b_id not in rec_by_id:
        raise SystemExit(f"[select_pairs] unknown trajectory id in selection: "
                         f"{a_id!r} / {b_id!r}")
    ra, rb = rec_by_id[a_id], rec_by_id[b_id]
    if rb.designed_reward > ra.designed_reward:
        ra, rb = rb, ra
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
    delta = (np.asarray(rb.features, dtype=float)
             - np.asarray(ra.features, dtype=float))
    return fip.PairCandidate(
        a_idx=a_idx, b_idx=b_idx, delta=delta, disagreements=disagreements,
    )


def _build_lottery(rec_by_id, entry, metric_items):
    """Resolve a lottery entry; compute reward + per-metric preferences."""
    c_id = entry["c"]
    a_id = entry["a"]
    b_id = entry["b"]
    p = float(entry["p"])
    if not (0.0 <= p <= 1.0):
        raise SystemExit(f"[select_pairs] lottery probability must be in [0, 1]; "
                         f"got p={p}")
    for tid in (c_id, a_id, b_id):
        if tid not in rec_by_id:
            raise SystemExit(f"[select_pairs] unknown trajectory id in lottery: {tid!r}")
    rec_c = rec_by_id[c_id]
    rec_a = rec_by_id[a_id]
    rec_b = rec_by_id[b_id]
    r_c = rec_c.designed_reward
    r_lottery = p * rec_a.designed_reward + (1.0 - p) * rec_b.designed_reward
    reward_prefers_c = r_c >= r_lottery
    disagreements = []
    for name, doc in metric_items:
        m_c = rec_c.metric_values[name]
        m_lot = (p * rec_a.metric_values[name]
                 + (1.0 - p) * rec_b.metric_values[name])
        metric_prefers_c = m_c >= m_lot
        if metric_prefers_c != reward_prefers_c:
            disagreements.append({
                "perspective": doc,
                "metric_c": float(m_c),
                "metric_lottery": float(m_lot),
                "prefers": "C" if metric_prefers_c else "L",
            })
    return {
        "rec_c": rec_c, "rec_a": rec_a, "rec_b": rec_b,
        "p": p,
        "reward_prefers_c": reward_prefers_c,
        "r_c": float(r_c),
        "r_lottery": float(r_lottery),
        "disagreements": disagreements,
        "justification": entry.get("justification"),
    }


def _try_import_workspace_viz():
    """Import the workspace's optional `viz` module if it exposes the contract."""
    try:
        import importlib
        viz_mod = importlib.import_module("viz")
    except Exception:
        return None
    if not hasattr(viz_mod, "visualize_trajectory"):
        return None
    if not hasattr(viz_mod, "VIZ_EXT"):
        return None
    return viz_mod


def _save_traj_viz(viz_mod, rec, out_path: str) -> str | None:
    """Run the workspace's viz hook on one trajectory; return path on success."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        viz_mod.visualize_trajectory(
            rec.trajectory, rec.metadata, out_path, source_path=rec.path,
        )
        return out_path
    except Exception as exc:
        print(f"  (visualization failed for {out_path}: {exc})")
        return None


def _ws_display_path(workspace_dir: str, rel_path: str) -> str:
    """Repo-relative path for display, e.g. 'workspaces/pandemic/viz/...'."""
    ws_name = os.path.basename(os.path.normpath(workspace_dir))
    return os.path.join("workspaces", ws_name, rel_path)


def _viz_url(ws_name: str, local_path: str) -> str:
    """Map a workspace-relative viz path ('viz/round1/pair1/A.mp4') to the
    server URL the web UI expects ('/workspace/<ws>/viz/round1/pair1/A.mp4')."""
    return f"/workspace/{ws_name}/{local_path}"


def _capture_and_echo(render_fn, *args, **kwargs) -> str:
    """Run a stdout-printing render fn, echo its output to the real stdout (so
    the human-readable stream is byte-for-byte unchanged) and also return the
    captured text, so it can be written to a per-card UI payload file."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        render_fn(*args, **kwargs)
    text = buf.getvalue()
    sys.stdout.write(text)
    return text


def _emit_ui_payload(ws_name: str, snap_id: int, feature_names, cards):
    """Write a ready-to-post UI bundle for this render to `UI_PAYLOAD_DIR`.

    The bundle is regenerated on every render, so the per-card visuals it wires
    up are always this round's (the snapshot id and the selected trajectories
    differ each round). The skill posts each card straight from the manifest and
    supplies only its own per-round judgment text (`--why`, plus the overall
    pattern paragraph) — it no longer hand-builds `--viz` JSON, the feature-key
    JSON, or the content files. Returns (payload_dir, manifest_path).
    """
    payload_dir = os.path.abspath(UI_PAYLOAD_DIR)
    # Clear any stale payload from a prior render so the skill never posts a
    # card (or a viz path) that belongs to an earlier round.
    if os.path.isdir(payload_dir):
        for fn in os.listdir(payload_dir):
            try:
                os.remove(os.path.join(payload_dir, fn))
            except OSError:
                pass
    os.makedirs(payload_dir, exist_ok=True)

    feature_key = [{"name": short, "description": desc}
                   for short, desc in feature_names]
    fk_path = os.path.join(payload_dir, "feature_key.json")
    with open(fk_path, "w") as f:
        json.dump(feature_key, f, indent=2)

    manifest_cards = []
    for card in cards:
        stem = f"{card['kind']}{card['rank']}"
        content_path = os.path.join(payload_dir, f"{stem}.content.txt")
        with open(content_path, "w") as f:
            f.write(card["content"])
        viz = card["viz"]
        viz_path = None
        if viz:
            viz_path = os.path.join(payload_dir, f"{stem}.viz.json")
            with open(viz_path, "w") as f:
                json.dump(viz, f)
        manifest_cards.append({
            "kind": card["kind"],
            "label": card["label"],
            "layout": card["layout"],
            "content_file": content_path,
            "viz": viz,
            "viz_file": viz_path,
        })

    manifest_path = os.path.join(payload_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "workspace": ws_name,
            "snapshot_id": snap_id,
            "feature_key_file": fk_path,
            "cards": manifest_cards,
        }, f, indent=2)
    return payload_dir, manifest_path


def _do_render(records, metrics, selection_path: str, workspace_dir: str) -> int:
    with open(selection_path) as f:
        selection = json.load(f)
    pairs_entries = selection.get("pairs", []) or []
    lottery_entries = selection.get("lotteries", []) or []
    if not pairs_entries and not lottery_entries:
        print("[select_pairs] selection file contains no pairs or lotteries.")
        return 1
    if len(pairs_entries) > MAX_PAIRS:
        raise SystemExit(f"[select_pairs] too many pairs: "
                         f"{len(pairs_entries)} > {MAX_PAIRS}")
    if len(lottery_entries) > MAX_LOTTERIES:
        raise SystemExit(f"[select_pairs] too many lotteries: "
                         f"{len(lottery_entries)} > {MAX_LOTTERIES}")

    ids = _assign_ids(records)
    rec_by_id = dict(zip(ids, records))
    metric_items = [(m[0], m[2]) for m in metrics]  # (name, doc)

    feature_names = records[0].feature_names if records else []
    fip._print_feature_key(feature_names)

    selected_pairs: list[fip.PairCandidate] = [
        _build_pair(rec_by_id, records, e, metric_items) for e in pairs_entries
    ]
    selected_lotteries: list[dict] = [
        _build_lottery(rec_by_id, e, metric_items) for e in lottery_entries
    ]

    # Compute the next snapshot id up front so the viz subtree can use it.
    snap_id = _next_snapshot_id()
    viz_mod = _try_import_workspace_viz()
    viz_root = f"viz/round{snap_id}"
    ws_name = os.path.basename(os.path.normpath(workspace_dir))

    # Collected per-card payloads (content text + round-specific viz list) that
    # get written to UI_PAYLOAD_DIR for the skill to post verbatim.
    cards: list[dict] = []

    if selected_pairs:
        print(f"\nTop {len(selected_pairs)} preference-label disagreements "
              f"(Claude-selected pairs):")
        total = len(selected_pairs)
        for rank, cand in enumerate(selected_pairs, 1):
            rec_a = records[cand.a_idx]
            rec_b = records[cand.b_idx]
            # Optional per-pair ranking of characteristics (short_names, most→
            # least informative about why the reward's preference may be wrong).
            order = pairs_entries[rank - 1].get("characteristics")
            content = _capture_and_echo(
                fip._print_pair_preference_labels,
                rank, total, rec_a, rec_b, cand.disagreements, 0.0, order,
            )
            viz_items: list[dict] = []
            if viz_mod is not None:
                ext = viz_mod.VIZ_EXT
                pair_dir = f"{viz_root}/pair{rank}"
                a_local = f"{pair_dir}/A{ext}"
                b_local = f"{pair_dir}/B{ext}"
                a_saved = _save_traj_viz(viz_mod, rec_a, a_local)
                b_saved = _save_traj_viz(viz_mod, rec_b, b_local)
                if a_saved or b_saved:
                    print()
                    print("  Visualizations:")
                    if a_saved:
                        print(f"    Trajectory A: {_ws_display_path(workspace_dir, a_saved)}")
                    if b_saved:
                        print(f"    Trajectory B: {_ws_display_path(workspace_dir, b_saved)}")
                # A is always the reward-preferred side (render orients it so),
                # so it is captioned as the designed reward's preferred trajectory.
                if a_saved:
                    viz_items.append({"src": _viz_url(ws_name, a_saved),
                                      "caption": "Designed reward function's preferred trajectory"})
                if b_saved:
                    viz_items.append({"src": _viz_url(ws_name, b_saved),
                                      "caption": "The alternative trajectory"})
            cards.append({
                "kind": "pair", "rank": rank, "label": f"PAIR {rank}/{total}",
                "layout": None, "content": content, "viz": viz_items,
            })

    if selected_lotteries:
        print(f"\nTop {len(selected_lotteries)} lottery comparison(s) "
              f"(Claude-selected):")
        total = len(selected_lotteries)
        for rank, lot in enumerate(selected_lotteries, 1):
            content = _capture_and_echo(
                fip._print_lottery_preference_labels,
                rank, total, lot["rec_c"], lot["rec_a"], lot["rec_b"], lot["p"],
                lot["reward_prefers_c"], lot["r_c"], lot["r_lottery"],
                lot["disagreements"], lot.get("justification"),
            )
            viz_items = []
            if viz_mod is not None:
                ext = viz_mod.VIZ_EXT
                lot_dir = f"{viz_root}/lottery{rank}"
                c_local = f"{lot_dir}/C{ext}"
                a_local = f"{lot_dir}/A{ext}"
                b_local = f"{lot_dir}/B{ext}"
                c_saved = _save_traj_viz(viz_mod, lot["rec_c"], c_local)
                a_saved = _save_traj_viz(viz_mod, lot["rec_a"], a_local)
                b_saved = _save_traj_viz(viz_mod, lot["rec_b"], b_local)
                if c_saved or a_saved or b_saved:
                    print()
                    print("  Visualizations:")
                    if c_saved:
                        print(f"    Certain (C): {_ws_display_path(workspace_dir, c_saved)}")
                    if a_saved:
                        print(f"    Branch A:    {_ws_display_path(workspace_dir, a_saved)}")
                    if b_saved:
                        print(f"    Branch B:    {_ws_display_path(workspace_dir, b_saved)}")
                p = lot["p"]
                # Which side the designed reward assigns higher return to: the
                # certain trajectory C, or the lottery (the gamble over A and B).
                # Carried on each viz item so the UI can flag the preferred side,
                # mirroring the preference shown on trajectory pairs.
                prefers_c = bool(lot["reward_prefers_c"])
                # Render order matches the lottery layout: A, B (top row with
                # probabilities), then C below.
                if a_saved:
                    viz_items.append({"src": _viz_url(ws_name, a_saved),
                                      "caption": "Trajectory A", "role": "A",
                                      "probability": round(p, 6),
                                      "reward_preferred": not prefers_c})
                if b_saved:
                    viz_items.append({"src": _viz_url(ws_name, b_saved),
                                      "caption": "Trajectory B", "role": "B",
                                      "probability": round(1.0 - p, 6),
                                      "reward_preferred": not prefers_c})
                if c_saved:
                    viz_items.append({"src": _viz_url(ws_name, c_saved),
                                      "caption": "Certain (C)", "role": "C",
                                      "reward_preferred": prefers_c})
            cards.append({
                "kind": "lottery", "rank": rank,
                "label": f"LOTTERY {rank}/{total}", "layout": "lottery",
                "content": content, "viz": viz_items,
            })

    snap_path = _save_claude_snapshot(
        records, selected_pairs, selected_lotteries, snap_id=snap_id,
    )
    print(f"\n[snapshot saved: {snap_path}]")

    payload_dir, manifest_path = _emit_ui_payload(
        ws_name, snap_id, feature_names, cards,
    )
    print(f"\n[ui_payload] ready-to-post bundle written to {payload_dir}")
    print(f"[ui_payload] manifest: {manifest_path}")
    print(f"[ui_payload] The per-card visuals are this round's — already wired")
    print(f"[ui_payload] into each card's viz list. Post once:")
    print(f"[ui_payload]   python -m web_ui.cli post --type feature_key "
          f"--features-file {os.path.join(payload_dir, 'feature_key.json')}")
    print(f"[ui_payload] then, for each card in manifest['cards'] (in order),")
    print(f"[ui_payload] supplying ONLY your own --why judgment text:")
    print(f"[ui_payload]   python -m web_ui.cli post --type comparison "
          f"--label <label> [--layout <layout>] \\")
    print(f"[ui_payload]       --content-file <content_file> "
          f"--viz \"$(cat <viz_file>)\" --why \"<your per-card interpretation>\"")
    print(f"[ui_payload] (omit --viz when viz_file is null; omit --layout when "
          f"it is null.)")
    return 0


def _next_snapshot_id() -> int:
    """Next available pref_round_*.json id; matches find_informative_pairs'."""
    import glob
    import re
    if not os.path.isdir(HISTORY_DIR):
        return 1
    existing = glob.glob(os.path.join(HISTORY_DIR, "pref_round_*.json"))
    nums = [int(re.search(r"pref_round_(\d+)\.json", p).group(1)) for p in existing]
    return (max(nums) + 1) if nums else 1


def _save_claude_snapshot(records, selected_pairs, selected_lotteries,
                           snap_id: int | None = None) -> str:
    """Write a snapshot in the same schema find_informative_pairs.py uses,
    tagged as Claude-selected. /repair-verify reads it via --check-snapshots.
    """
    from datetime import datetime

    os.makedirs(HISTORY_DIR, exist_ok=True)
    if snap_id is None:
        snap_id = _next_snapshot_id()
    snap = {
        "snapshot_id": snap_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selection_method": "claude_selected",
        "pairs": [],
        "lotteries": [],
    }
    for rank, cand in enumerate(selected_pairs, 1):
        rec_a = records[cand.a_idx]
        rec_b = records[cand.b_idx]
        snap["pairs"].append({
            "rank": rank,
            "score": 0.0,
            "traj_a_path": rec_a.path,
            "traj_b_path": rec_b.path,
            "snapshot_reward_a": float(rec_a.designed_reward),
            "snapshot_reward_b": float(rec_b.designed_reward),
            "disagreements": cand.disagreements,
        })
    for rank, lot in enumerate(selected_lotteries, 1):
        snap["lotteries"].append({
            "rank": rank,
            "traj_c_path": lot["rec_c"].path,
            "traj_a_path": lot["rec_a"].path,
            "traj_b_path": lot["rec_b"].path,
            "p": float(lot["p"]),
            "snapshot_reward_c": lot["r_c"],
            "snapshot_reward_lottery": lot["r_lottery"],
            "snapshot_reward_prefers_c": bool(lot["reward_prefers_c"]),
            "disagreements": lot["disagreements"],
            "justification": lot.get("justification"),
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
    return _do_render(records, metrics, args.selection, ws)


if __name__ == "__main__":
    sys.exit(main())
