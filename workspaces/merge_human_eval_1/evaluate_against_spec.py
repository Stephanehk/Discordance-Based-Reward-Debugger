"""Evaluate this workspace's designed reward against its eval spec.

This script is specific to the ``merge_human_eval_1`` evaluation workspace
and lives inside it. It is a **persistent eval artifact** (like
``env_config.py`` / ``eval_spec.md`` / the starting ``designed_reward_fn.py``):
it must be kept across workspace resets, not treated as authored/collected
output.

The check is **ranking-based**: a reward is judged aligned with the spec
when every spec-satisfying trajectory is scored higher than every
trajectory that violates the spec. The checkable criteria from
``eval_spec.md`` are: no crash; nearest observed gap >= 10 m at all steps;
trajectory-average speed near 25 m/s.

Because real rollouts in this env rarely (or never) satisfy the strict
criteria, the harness *synthesizes* a guaranteed-compliant set by ablating
each collected trajectory — copying it and pushing the relevant features
to compliant values (ego speed -> 25 m/s, every observed gap -> >= 10 m,
no crash). It then asks whether the designed reward ranks all of those
compliant trajectories above all of the real, violating ones.

The ablation uses the highway-env Kinematics observation layout (ego speed
= ``obs[0,3] * 80`` m/s; neighbour gap = ``hypot(col1*200, col2*8)`` m).
Compliance *classification* of real trajectories is read from this
workspace's ``Trajectory2Features`` by the ``mean_speed`` / ``min_gap`` /
``crashed`` short-names.

It additionally checks the spec's second-paragraph *indifference* clause (a
no-crash trajectory always 15-25 m from the nearest vehicle averaging
20 m/s should be equally preferable to a no-crash trajectory always 10-12 m
away averaging 27 m/s). The strict ranking test cannot express indifference,
so this is tested directly: every collected trajectory is ablated into both
profiles (holding all other observation dimensions fixed), giving one
matched pair per trajectory whose two members the reward *should* score
equally. The mean within-pair reward gap is then compared against the
spread of return gaps between arbitrary trajectory pairs.

Usage (from the repo root, with the `llm` conda env active):
    python workspaces/merge_human_eval_1/evaluate_against_spec.py
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

# Equality-preference profiles (eval_spec.md, 2nd paragraph). Each is a
# constant gap inside the stated band and the stated average speed, crash-free.
# The spec says the two profiles should be equally preferable.
EQ_A_SPEED_MPS, EQ_A_GAP_M = 20.0, 20.0   # "between 15 and 25 m" -> midpoint 20 m
EQ_B_SPEED_MPS, EQ_B_GAP_M = 27.0, 11.0   # "between 10 and 12 m" -> midpoint 11 m

# highway-env Kinematics denormalisation for the ablation.
VX_HALFWIDTH_MPS = 80.0
X_HALFWIDTH_M = 200.0
Y_HALFWIDTH_M = 8.0


def _feature_index(names, key):
    for i, n in enumerate(names):
        if key in n:
            return i
    raise SystemExit(
        f"This workspace's trajectory features expose no '{key}' feature; "
        "the spec evaluator needs mean_speed / min_gap / crashed features."
    )


def _compliant_obs(obs):
    """Return a copy of an observation forced to satisfy every criterion."""
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


def _profile_obs(obs, speed_mps, gap_m):
    """Force an observation to an exact (no-crash, gap, ego-speed) profile.

    Unlike ``_compliant_obs`` (which only floors gaps up to the minimum),
    this sets the ego speed and *every* present neighbour's distance to the
    exact requested values, holding all other observation dimensions fixed.
    """
    m = np.array(obs, dtype=float, copy=True)
    m[0, 0] = 1.0                                   # no crash flag on ego row
    m[0, 3] = speed_mps / VX_HALFWIDTH_MPS          # ego speed -> speed_mps
    for i in range(1, m.shape[0]):
        if m[i, 0] > 0.5:                           # present neighbour
            rx, ry = m[i, 1] * X_HALFWIDTH_M, m[i, 2] * Y_HALFWIDTH_M
            dist = float(np.hypot(rx, ry))
            if dist > 1e-6:
                scale = gap_m / dist                # rescale to exactly gap_m
                m[i, 1] *= scale
                m[i, 2] *= scale
            else:
                m[i, 1] = gap_m / X_HALFWIDTH_M     # place dead ahead at gap_m
    return m


def _profile_traj(traj, speed_mps, gap_m):
    return [(_profile_obs(o, speed_mps, gap_m), a, _profile_obs(no, speed_mps, gap_m))
            for o, a, no in traj]


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
    compliant = [{"reward": r["reward"], "source": "real"}
                 for r in real if not r["viol"]]
    ablated = []
    for r in real:
        atraj = _ablate(r["traj"])
        assert not violations(atraj), "ablation failed to produce a compliant trajectory"
        ablated.append({"reward": reward_of(atraj), "source": "ablated"})
    compliant += ablated
    violators = [{"reward": r["reward"], "viol": r["viol"]} for r in real if r["viol"]]

    # ---- report ----
    print("=" * 78)
    print("SPEC EVALUATION (ranking-based) — designed reward vs. eval_spec.md")
    print("=" * 78)
    with open("eval_spec.md") as fh:
        print(fh.read().rstrip())
    print("-" * 78)
    print(f"Checkable criteria: no crash; nearest observed gap >= {MIN_GAP_M:g} m "
          f"at all steps; trajectory-average speed within {args.speed_tol:g} m/s of "
          f"{TARGET_SPEED_MPS:g} m/s.")
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

    # ---- equality-preference check (eval_spec.md, 2nd paragraph) ----
    # Ablate every collected trajectory into the two profiles the spec calls
    # equally preferable, holding all other obs dims fixed, so each source
    # yields one matched pair the reward *should* score equally. Then compare
    # the within-pair reward gap to the spread of return gaps between
    # arbitrary trajectory pairs.
    eq_diffs = np.array([
        reward_of(_profile_traj(r["traj"], EQ_A_SPEED_MPS, EQ_A_GAP_M))
        - reward_of(_profile_traj(r["traj"], EQ_B_SPEED_MPS, EQ_B_GAP_M))
        for r in real
    ])
    eq_abs = np.abs(eq_diffs)
    returns = np.array([r["reward"] for r in real])
    pair_abs = np.array([abs(returns[i] - returns[j])
                         for i in range(len(returns))
                         for j in range(i + 1, len(returns))])

    print("Equality-preference check (eval_spec.md, 2nd paragraph):")
    print(f"  Profile A: no crash, gap = {EQ_A_GAP_M:g} m (in 15-25 m band), "
          f"avg speed = {EQ_A_SPEED_MPS:g} m/s")
    print(f"  Profile B: no crash, gap = {EQ_B_GAP_M:g} m (in 10-12 m band), "
          f"avg speed = {EQ_B_SPEED_MPS:g} m/s")
    print(f"  Matched pairs (one ablated from each collected trajectory): "
          f"{len(eq_diffs)}")
    print(f"  Mean within-pair reward difference A-B (ideal 0.00): "
          f"{eq_diffs.mean():+.3f}")
    print(f"  Mean within-pair |reward difference|: {eq_abs.mean():.3f}")
    if pair_abs.size:
        mu, sd = float(pair_abs.mean()), float(pair_abs.std())
        print(f"  Mean |return difference| over all {pair_abs.size} evaluated "
              f"trajectory pairs: {mu:.3f} (sd {sd:.3f})")
        delta = float(eq_abs.mean()) - mu
        if sd > 1e-12:
            print(f"  Equality-pair gap vs. typical pair gap: {delta:+.3f} "
                  f"({delta / sd:+.2f} sd from the all-pairs mean)")
        else:
            print(f"  Equality-pair gap vs. typical pair gap: {delta:+.3f} "
                  f"(all trajectory pairs have equal return; sd = 0)")
    else:
        print("  Only one trajectory collected — no all-pairs baseline.")
    print("-" * 78)

    if not violators:
        print("No spec-violating trajectory was collected — ranking test is "
              "trivially satisfied.")
        print("=" * 78)
        return

    worst_compliant = min(c["reward"] for c in compliant)
    best_violator = max(v["reward"] for v in violators)
    total = len(compliant) * len(violators)
    inversions = sum(1 for c in compliant for v in violators
                     if v["reward"] >= c["reward"])
    passed = worst_compliant > best_violator

    print("Ranking test — does every spec-satisfying trajectory out-score "
          "every violating one?")
    print(f"  Worst (lowest-reward) compliant trajectory: {worst_compliant:.2f}")
    print(f"  Best  (highest-reward) violating trajectory: {best_violator:.2f}")
    print(f"  Compliant-below-violator inversions: {inversions}/{total} "
          f"({100*inversions/total:.1f}%)")
    print("-" * 78)
    print(f"RESULT: the designed reward ranks ALL spec-satisfying trajectories "
          f"above ALL violating ones: {'YES' if passed else 'NO'}")
    if not passed:
        print("  (At least one violating trajectory scores >= a compliant one; "
              "optimizing this reward need not yield spec-satisfying behavior.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
