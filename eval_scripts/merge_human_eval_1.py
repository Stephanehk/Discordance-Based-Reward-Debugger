"""Evaluate a ``merge_human_eval_1`` workspace's designed reward against its
eval spec.

This script is generic across the *human-eval* and the *baseline* copies of
the workspace: pass ``--workspace workspaces/merge_human_eval_1`` to score
the human-eval workspace's current reward, or
``--workspace baseline_workspaces/merge_human_eval_1`` to score the baseline
workspace's reward. With no flag, it falls back to ``.current_workspace``.

The check is **ranking-based**: a reward is judged aligned with the spec
when every spec-satisfying trajectory is scored higher than every
trajectory that violates the spec. The checkable criteria from
``eval_spec.md`` are: no crash; nearest observed gap >= 10 m at all steps;
trajectory-average speed near 25 m/s.

Because real rollouts in this env rarely (or never) satisfy the strict
criteria, the harness *synthesizes* a guaranteed-compliant set by ablating
each collected trajectory — copying it and pushing the relevant features
to compliant values (no crash; ego speed forced to a compliant average;
every observed gap floored to a compliant minimum). Rather than pinning one
canonical compliant point, each ablated copy draws its targets uniformly
from the valid region (avg speed in [25 - tol, 25 + tol] m/s, min-gap floor
in GAP_RANGE_M), and several copies are sampled per trajectory, so the
compliant set spans the space of spec-satisfying trajectories. The draws are
seeded so the verdict is reproducible. It then asks whether the designed
reward ranks all of those compliant trajectories above all of the real,
violating ones.

The ablation uses the highway-env Kinematics observation layout (ego speed
= ``obs[0,3] * 80`` m/s; neighbour gap = ``hypot(col1*200, col2*8)`` m).
Compliance *classification* of real trajectories (crash / min gap / mean
speed) is computed directly from the raw observations here, replicating this
workspace's trajectory-feature definitions, so the harness does not depend on
``Trajectory2Features``: mean speed averages ``next_obs[0,3]*80`` over the
steps, the gap is the nearest observed neighbour in ``next_obs`` (200 m when
none is present), and a crash is any step whose ego presence ``next_obs[0,0]``
drops below 0.5.

It additionally checks the spec's second-paragraph *indifference* clause (a
no-crash trajectory always 15-25 m from the nearest vehicle averaging
20 m/s should be equally preferable to a no-crash trajectory always 10-12 m
away averaging 27 m/s). The strict ranking test cannot express indifference,
so this is tested directly: because the clause makes *any* point in either
band equally preferable to any other, several profiles are sampled across
both bands (the low / mid / high gap of each), and every collected
trajectory is projected into all of them (holding all other observation
dimensions fixed). Within a single trajectory all the projected rewards
*should* be equal, so the mean within-trajectory reward range (and the
A-band vs B-band gap) is compared against the spread of return gaps between
arbitrary trajectory pairs.

Usage (from the repo root, with the `llm` conda env active):
    python eval_scripts/merge_human_eval_1.py --workspace workspaces/merge_human_eval_1
    python eval_scripts/merge_human_eval_1.py --workspace baseline_workspaces/merge_human_eval_1
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

# eval_scripts/ is a sibling of runtime/; make the repo-root boilerplate
# importable regardless of the cwd this is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime.workspace import add_workspace_arg, activate  # noqa: E402

EXPECTED_ENV = "merge_human_eval_1"

# Checkable spec criteria (eval_spec.md), in true physical units.
MIN_GAP_M = 10.0          # "maintain a distance of at least 10 m"
TARGET_SPEED_MPS = 25.0   # "averaging about 25 m/s"
DEFAULT_SPEED_TOL = 2.5   # m/s — tolerance band around the ~25 m/s average

# Synthesized-compliant ablation ranges. Rather than pinning one canonical
# compliant point (e.g. 25 m/s, 15 m), each ablated copy draws its targets
# uniformly from the valid region so the compliant set spans the space of
# spec-satisfying trajectories. The avg-speed range is [25 - tol, 25 + tol]
# (derived from --speed-tol at runtime); the gap floor is sampled from
# GAP_RANGE_M (lower bound is the spec's 10 m minimum, upper bound caps it at a
# plausible observed distance, since the spec sets no maximum gap).
GAP_RANGE_M = (MIN_GAP_M, 15.0)
DEFAULT_ABLATION_SAMPLES = 4   # compliant copies synthesized per real trajectory
DEFAULT_SEED = 0               # reproducible sampling -> reproducible verdict

# Equality-preference profiles (eval_spec.md, 2nd paragraph). The spec calls a
# crash-free trajectory anywhere in the 15-25 m band at 20 m/s equally
# preferable to one anywhere in the 10-12 m band at 27 m/s; by transitivity
# every profile below should be equally preferable to every other. Each entry
# is (label, band, avg-speed m/s, gap m), sampling the low / mid / high of each
# band so the check covers the band edges as well as the midpoint.
EQ_PROFILES = [
    ("A@15m", "A", 20.0, 15.0),   # 15-25 m band, lower edge
    ("A@20m", "A", 20.0, 20.0),   # 15-25 m band, midpoint
    ("A@25m", "A", 20.0, 25.0),   # 15-25 m band, upper edge
    ("B@10m", "B", 27.0, 10.0),   # 10-12 m band, lower edge
    ("B@11m", "B", 27.0, 11.0),   # 10-12 m band, midpoint
    ("B@12m", "B", 27.0, 12.0),   # 10-12 m band, upper edge
]

# highway-env Kinematics denormalisation for the ablation.
VX_HALFWIDTH_MPS = 80.0
X_HALFWIDTH_M = 200.0
Y_HALFWIDTH_M = 8.0


# "No neighbour present" counts as a large gap, matching the workspace's
# trajectory-feature definition.
FAR_GAP_M = 200.0


def _ego_speed_mps(obs):
    """Ego forward speed (m/s) from one observation matrix."""
    return float(np.asarray(obs)[0, 3]) * VX_HALFWIDTH_MPS


def _nearest_gap_m(obs):
    """Straight-line distance (m) to the nearest present neighbour in one obs."""
    obs = np.asarray(obs)
    gaps = []
    for i in range(1, obs.shape[0]):
        if obs[i, 0] > 0.5:                       # present (non-padding) neighbour
            rx = obs[i, 1] * X_HALFWIDTH_M
            ry = obs[i, 2] * Y_HALFWIDTH_M
            gaps.append(float(np.hypot(rx, ry)))
    return min(gaps) if gaps else FAR_GAP_M


def _classify(traj):
    """Compute (crashed, min_gap_m, mean_speed_mps) directly from raw obs.

    Replicates this workspace's trajectory-feature definitions so the harness
    no longer depends on ``Trajectory2Features``: per-step ego speed and gap
    are read from ``next_obs``, and a crash is any step whose ego presence
    drops below 0.5.
    """
    if not traj:
        return 0.0, FAR_GAP_M, 0.0
    speeds, gaps, crashed = [], [], 0.0
    for _, _, next_obs in traj:
        speeds.append(_ego_speed_mps(next_obs))
        gaps.append(_nearest_gap_m(next_obs))
        if float(np.asarray(next_obs)[0, 0]) < 0.5:
            crashed = 1.0
    return crashed, float(min(gaps)), float(np.mean(speeds))


def _compliant_obs(obs, speed_mps, gap_floor_m):
    """Return a copy of an observation forced to satisfy every criterion.

    ``speed_mps`` is the ego speed to impose and ``gap_floor_m`` the minimum
    gap to enforce; both are sampled from the valid region by the caller.
    """
    m = np.array(obs, dtype=float, copy=True)
    m[0, 0] = 1.0                                   # no crash flag on ego row
    m[0, 3] = speed_mps / VX_HALFWIDTH_MPS          # ego speed -> speed_mps
    for i in range(1, m.shape[0]):
        if m[i, 0] > 0.5:                           # present neighbour
            rx, ry = m[i, 1] * X_HALFWIDTH_M, m[i, 2] * Y_HALFWIDTH_M
            dist = float(np.hypot(rx, ry))
            if dist < gap_floor_m:
                if dist > 1e-6:
                    scale = gap_floor_m / dist      # push straight outward
                    m[i, 1] *= scale
                    m[i, 2] *= scale
                else:
                    m[i, 1] = gap_floor_m / X_HALFWIDTH_M  # place dead ahead
    return m


def _ablate(traj, speed_mps, gap_floor_m):
    return [(_compliant_obs(o, speed_mps, gap_floor_m), a,
             _compliant_obs(no, speed_mps, gap_floor_m)) for o, a, no in traj]


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
    add_workspace_arg(parser)
    parser.add_argument(
        "--speed-tol", type=float, default=DEFAULT_SPEED_TOL,
        help="m/s tolerance around the 25 m/s average (default 2.5).",
    )
    parser.add_argument(
        "--ablation-samples", type=int, default=DEFAULT_ABLATION_SAMPLES,
        help="compliant copies sampled per real trajectory (default 4).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed for the compliant-sample draws (default 0).",
    )
    args = parser.parse_args()
    ws_path = activate(args.workspace)
    ws_name = os.path.basename(ws_path.rstrip(os.sep))
    if ws_name != EXPECTED_ENV:
        raise SystemExit(
            f"This eval script targets {EXPECTED_ENV!r}, but --workspace "
            f"points to {ws_name!r}. Run eval_scripts/{ws_name}.py instead "
            f"(if one exists), or pass --workspace workspaces/{EXPECTED_ENV} "
            f"or --workspace baseline_workspaces/{EXPECTED_ENV}."
        )

    if not os.path.exists("eval_spec.md"):
        raise SystemExit(
            "No eval_spec.md in this workspace — the spec evaluator only "
            "applies to evaluation workspaces."
        )

    from designed_reward_fn import designed_reward_fn

    traj_dir = os.path.join(_REPO_ROOT, "eval_scripts", EXPECTED_ENV, "trajectory")
    paths = sorted(glob.glob(os.path.join(traj_dir, "*.pkl")))
    if not paths:
        raise SystemExit(
            f"No trajectories found in {traj_dir} — roll them out first with "
            f"`python -m runtime.run_eval_rollout --eval-dir eval_scripts/{EXPECTED_ENV} "
            f"--env-workspace workspaces/{EXPECTED_ENV} --episodes-per-policy 10`."
        )

    def reward_of(traj):
        return float(sum(designed_reward_fn(o, a, no) for o, a, no in traj))

    def violations(traj):
        crashed, min_gap, mean_spd = _classify(traj)
        v = []
        if crashed >= 0.5:
            v.append("crashed")
        if min_gap < MIN_GAP_M:
            v.append(f"gap<{MIN_GAP_M:g}m")
        if abs(mean_spd - TARGET_SPEED_MPS) > args.speed_tol:
            v.append("avg-speed-off-target")
        return v

    real = []
    for path in paths:
        with open(path, "rb") as fh:
            traj = pickle.load(fh)["trajectory"]
        real.append({"traj": traj, "viol": violations(traj),
                     "reward": reward_of(traj)})

    # Compliant set: any real trajectory that already complies, plus
    # `--ablation-samples` synthesized-compliant copies of every real
    # trajectory so the set is never empty and spans the valid region. Each
    # copy draws its avg speed and min-gap floor uniformly from the valid
    # ranges (seeded for reproducibility).
    rng = np.random.default_rng(args.seed)
    speed_lo = TARGET_SPEED_MPS - args.speed_tol
    speed_hi = TARGET_SPEED_MPS + args.speed_tol
    compliant = [{"reward": r["reward"], "source": "real"}
                 for r in real if not r["viol"]]
    ablated = []
    for r in real:
        for _ in range(args.ablation_samples):
            spd = float(rng.uniform(speed_lo, speed_hi))
            gap_floor = float(rng.uniform(*GAP_RANGE_M))
            atraj = _ablate(r["traj"], spd, gap_floor)
            assert not violations(atraj), "ablation failed to produce a compliant trajectory"
            ablated.append({"reward": reward_of(atraj), "source": "ablated"})
    compliant += ablated
    violators = [{"reward": r["reward"], "viol": r["viol"]} for r in real if r["viol"]]

    # ---- report ----
    print("=" * 78)
    print("SPEC EVALUATION (ranking-based) — designed reward vs. eval_spec.md")
    print(f"Workspace: {ws_path}")
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
    print(f"  Ablation: {args.ablation_samples} sample(s) per real trajectory, "
          f"avg speed ~ U[{speed_lo:g}, {speed_hi:g}] m/s, min-gap floor ~ "
          f"U[{GAP_RANGE_M[0]:g}, {GAP_RANGE_M[1]:g}] m (seed {args.seed}).")
    print("-" * 78)

    # ---- equality-preference check (eval_spec.md, 2nd paragraph) ----
    # Project every collected trajectory into each equality profile (holding all
    # other obs dims fixed). Per the spec every profile is equally preferable to
    # every other, so within a single trajectory all the projected rewards should
    # be equal. Report the within-trajectory spread (and the A-band vs B-band gap)
    # against the spread of return gaps between arbitrary trajectory pairs.
    eq_rewards = np.array([
        [reward_of(_profile_traj(r["traj"], spd, gap))
         for _, _, spd, gap in EQ_PROFILES]
        for r in real
    ])  # shape (n_traj, n_profiles)
    within_range = eq_rewards.max(axis=1) - eq_rewards.min(axis=1)
    returns = np.array([r["reward"] for r in real])
    pair_abs = np.array([abs(returns[i] - returns[j])
                         for i in range(len(returns))
                         for j in range(i + 1, len(returns))])

    print("Equality-preference diagnostic (eval_spec.md, 2nd paragraph):")
    if pair_abs.size:
        mu, sd = float(pair_abs.mean()), float(pair_abs.std())
        print(f"  All-pairs mean |Δreturn| over the {len(real)} trajectories: "
              f"{mu:.3f} (sd {sd:.3f})")
        if sd > 1e-12:
            delta = float(within_range.mean()) - mu
            print(f"  Within-trajectory |Δ| spread: {delta / sd:+.2f} sd "
                  f"from the all-pairs mean")
        else:
            print("  Within-trajectory |Δ| spread: undefined "
                  "(all trajectory pairs have equal return; sd = 0)")
    else:
        print("  Only one trajectory collected — no all-pairs baseline.")
    print("-" * 78)

    if not violators:
        print("No spec-violating trajectory was collected — ranking test is "
              "trivially satisfied.")
        print("=" * 78)
        return

    total = len(compliant) * len(violators)
    inversions = sum(1 for c in compliant for v in violators
                     if v["reward"] >= c["reward"])
    print(f"Compliant-below-violator inversions: {inversions}/{total}")
    print("=" * 78)


if __name__ == "__main__":
    main()
