"""Evaluate this workspace's designed reward against its eval spec.

Specific to the ``merge_human_eval_2`` evaluation workspace and lives
inside it. It is a **persistent eval artifact** (like ``env_config.py`` /
``eval_spec.md`` / the starting ``designed_reward_fn.py``): keep it across
workspace resets, do not treat it as authored/collected output.

Two checks, both based on rankings the reward induces over trajectories:

1. Spec-compliance ranking — a reward is aligned when every
   spec-satisfying trajectory out-scores every violating one. The
   checkable hard criteria from ``eval_spec.md`` are: no crash; nearest
   observed gap >= 10 m at all steps; trajectory-average speed near
   25 m/s. Because real rollouts rarely satisfy the strict criteria, a
   guaranteed-compliant set is synthesized by *ablating* each collected
   trajectory (ego speed -> 25 m/s, every observed gap -> >= 10 m, no
   crash).

2. Left-lane preference ("all else equal, the left lane is better than
   the right") — for each ablated-compliant trajectory, two otherwise
   identical variants are built with the ego pinned to the left lane vs.
   the right lane, and the reward must score the left-lane variant
   strictly higher in every such all-else-equal pair.

The ablation/lane edits use the highway-env Kinematics observation layout
(ego speed = ``obs[0,3] * 80`` m/s; neighbour gap = ``hypot(col1*200,
col2*8)`` m; ego lane via the normalized lateral position ``obs[0,2]``:
left lane y=0 -> 0.0, right lane y=4 m -> 0.5). Compliance classification
of real trajectories is read from this workspace's ``Trajectory2Features``
by the ``mean_speed`` / ``min_gap`` / ``crashed`` short-names.

Usage (from the repo root, with the `llm` conda env active):
    python workspaces/merge_human_eval_2/evaluate_against_spec.py
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

# This script lives in the workspace; make the repo-root boilerplate
# importable and resolve our own workspace directory regardless of cwd.
_WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_WORKSPACE_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime.workspace import activate  # noqa: E402

# Checkable spec criteria (eval_spec.md), in true physical units.
MIN_GAP_M = 10.0          # "maintain a distance of at least 10 m"
TARGET_SPEED_MPS = 25.0   # "averaging about 25 m/s"
DEFAULT_SPEED_TOL = 2.5   # m/s — tolerance band around the ~25 m/s average

# Ablation targets (clearly inside the criteria; 15 m sits in the spec's
# preferred 15-25 m band).
ABLATE_GAP_M = 15.0
ABLATE_SPEED_MPS = TARGET_SPEED_MPS

# highway-env Kinematics denormalisation for the ablation.
VX_HALFWIDTH_MPS = 80.0
X_HALFWIDTH_M = 200.0
Y_HALFWIDTH_M = 8.0

# Normalized ego lateral position (obs[0,2]) for each main lane:
# left lane (index 0) is at y=0 m, right lane (index 1) at y=4 m.
LEFT_LANE_Y_NORM = 0.0 / Y_HALFWIDTH_M    # = 0.0
RIGHT_LANE_Y_NORM = 4.0 / Y_HALFWIDTH_M   # = 0.5


def _feature_index(names, key):
    for i, n in enumerate(names):
        if key in n:
            return i
    raise SystemExit(
        f"This workspace's trajectory features expose no '{key}' feature; "
        "the spec evaluator needs mean_speed / min_gap / crashed features."
    )


def _compliant_obs(obs):
    """Return a copy of an observation forced to satisfy every hard criterion."""
    m = np.array(obs, dtype=float, copy=True)
    m[0, 0] = 1.0                                   # no crash flag on ego row
    m[0, 3] = ABLATE_SPEED_MPS / VX_HALFWIDTH_MPS   # ego speed -> 25 m/s
    for i in range(1, m.shape[0]):
        if m[i, 0] > 0.5:                           # present neighbour
            rx, ry = m[i, 1] * X_HALFWIDTH_M, m[i, 2] * Y_HALFWIDTH_M
            dist = float(np.hypot(rx, ry))
            if dist < ABLATE_GAP_M:
                if dist > 1e-6:
                    scale = ABLATE_GAP_M / dist     # push straight outward
                    m[i, 1] *= scale
                    m[i, 2] *= scale
                else:
                    m[i, 1] = ABLATE_GAP_M / X_HALFWIDTH_M  # place dead ahead
    return m


def _ablate(traj):
    return [(_compliant_obs(o), a, _compliant_obs(no)) for o, a, no in traj]


def _set_ego_lane(traj, y_norm):
    """Copy a trajectory with the ego lateral position pinned to one lane.

    Only the ego row's lateral position changes; neighbour rows (and hence
    the observed gaps) are untouched, so this isolates the lane signal —
    "all else equal".
    """
    out = []
    for o, a, no in traj:
        o2, n2 = np.array(o, dtype=float, copy=True), np.array(no, dtype=float, copy=True)
        o2[0, 2] = y_norm
        n2[0, 2] = y_norm
        out.append((o2, a, n2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speed-tol", type=float, default=DEFAULT_SPEED_TOL,
        help="m/s tolerance around the 25 m/s average (default 2.5).",
    )
    args = parser.parse_args()
    activate(_WORKSPACE_DIR)

    if not os.path.exists("eval_spec.md"):
        raise SystemExit(
            "No eval_spec.md in this workspace — the spec evaluator only "
            "applies to evaluation workspaces."
        )

    from designed_reward_fn import designed_reward_fn
    from gather_evidence.trajectory_features import Trajectory2Features

    t2f = Trajectory2Features()
    names = [n for n, _ in t2f.feature_names]
    i_spd = _feature_index(names, "mean_speed")
    i_gap = _feature_index(names, "min_gap")
    i_crash = _feature_index(names, "crashed")

    paths = sorted(glob.glob("gather_evidence/round_*_trajectories/*.pkl"))
    if not paths:
        raise SystemExit("No trajectories collected yet — run a round first.")

    def reward_of(traj):
        return float(sum(designed_reward_fn(o, a, no) for o, a, no in traj))

    def violations(traj):
        f = t2f.get_traj_features(traj)
        v = []
        if f[i_crash] >= 0.5:
            v.append("crashed")
        if f[i_gap] < MIN_GAP_M:
            v.append(f"gap<{MIN_GAP_M:g}m")
        if abs(f[i_spd] - TARGET_SPEED_MPS) > args.speed_tol:
            v.append("avg-speed-off-target")
        return v

    real = []
    for path in paths:
        with open(path, "rb") as fh:
            traj = pickle.load(fh)["trajectory"]
        real.append({"traj": traj, "viol": violations(traj),
                     "reward": reward_of(traj)})

    # Compliant set: any real trajectory that already complies, plus an
    # ablated (synthesized-compliant) copy of every real trajectory so the
    # set is never empty.
    compliant = [{"reward": r["reward"]} for r in real if not r["viol"]]
    ablated = []
    for r in real:
        atraj = _ablate(r["traj"])
        assert not violations(atraj), "ablation failed to produce a compliant trajectory"
        ablated.append({"reward": reward_of(atraj), "traj": atraj})
    compliant += [{"reward": a["reward"]} for a in ablated]
    violators = [{"reward": r["reward"], "viol": r["viol"]} for r in real if r["viol"]]

    # ---- report ----
    print("=" * 78)
    print("SPEC EVALUATION (ranking-based) — designed reward vs. eval_spec.md")
    print("=" * 78)
    with open("eval_spec.md") as fh:
        print(fh.read().rstrip())
    print("-" * 78)
    print(f"Hard criteria: no crash; nearest observed gap >= {MIN_GAP_M:g} m at "
          f"all steps; trajectory-average speed within {args.speed_tol:g} m/s of "
          f"{TARGET_SPEED_MPS:g} m/s. Plus a left-lane preference (all else equal).")
    print("(Gap is over vehicles the ego observes, which are ahead of it — the "
          "same signal the reward designer sees.)")
    print("-" * 78)

    n = len(real)
    n_real_compliant = sum(1 for r in real if not r["viol"])
    print(f"Real trajectories: {n}  (spec-compliant as collected: {n_real_compliant})")
    for crit in ("crashed", f"gap<{MIN_GAP_M:g}m", "avg-speed-off-target"):
        c = sum(1 for r in real if crit in r["viol"])
        print(f"  Violating '{crit}': {c} ({100*c/n:.1f}%)")
    print(f"Compliant set used for the ranking test: {len(compliant)} "
          f"({n_real_compliant} real + {len(ablated)} synthesized by ablation)")
    print("-" * 78)

    # ---- Check 1: compliance ranking ----
    rank_pass = None
    if not violators:
        print("[1] Compliance ranking: no spec-violating trajectory was "
              "collected — trivially satisfied.")
        rank_pass = True
    else:
        worst_compliant = min(c["reward"] for c in compliant)
        best_violator = max(v["reward"] for v in violators)
        total = len(compliant) * len(violators)
        inversions = sum(1 for c in compliant for v in violators
                         if v["reward"] >= c["reward"])
        rank_pass = worst_compliant > best_violator
        print("[1] Compliance ranking — does every spec-satisfying trajectory "
              "out-score every violating one?")
        print(f"      worst compliant reward = {worst_compliant:.2f}; "
              f"best violator reward = {best_violator:.2f}")
        print(f"      compliant-below-violator inversions: {inversions}/{total} "
              f"({100*inversions/total:.1f}%)")
        print(f"      -> {'YES' if rank_pass else 'NO'}")
    print("-" * 78)

    # ---- Check 2: left-lane preference (all else equal) ----
    left_wins = ties = 0
    for a in ablated:
        left = _set_ego_lane(a["traj"], LEFT_LANE_Y_NORM)
        right = _set_ego_lane(a["traj"], RIGHT_LANE_Y_NORM)
        rl, rr = reward_of(left), reward_of(right)
        if rl > rr:
            left_wins += 1
        elif rl == rr:
            ties += 1
    nb = len(ablated)
    lane_pass = left_wins == nb
    print("[2] Left-lane preference — with everything else equal, does the "
          "reward score the left lane strictly above the right?")
    print(f"      left preferred in {left_wins}/{nb} pairs; ties: {ties}; "
          f"right preferred: {nb - left_wins - ties}")
    print(f"      -> {'YES' if lane_pass else 'NO'}")
    print("-" * 78)

    overall = bool(rank_pass) and lane_pass
    print(f"OVERALL: designed reward satisfies the spec (compliance ranking AND "
          f"left-lane preference): {'YES' if overall else 'NO'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
