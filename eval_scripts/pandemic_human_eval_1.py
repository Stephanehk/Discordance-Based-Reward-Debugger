"""Evaluate a ``pandemic_human_eval_1`` workspace's designed reward against
its eval spec.

This script is generic across the *human-eval* and the *baseline* copies of
the workspace: pass ``--workspace workspaces/pandemic_human_eval_1`` to score
the human-eval workspace's current reward, or
``--workspace baseline_workspaces/pandemic_human_eval_1`` to score the
baseline workspace's reward. With no flag, it falls back to
``.current_workspace``.

The spec constrains the *regulation-stage decision as a function of the
critical-case proportion* — not the outbreak outcome (which the agent cannot
set directly). A single global "every compliant trajectory out-scores every
violating one" test would therefore misfire here, because two trajectories can
differ in reward purely because one had a worse outbreak regardless of whether
the stage decisions were good. So each spec criterion is tested with its own
**per-trajectory comparison**: starting from each collected trajectory, the
harness synthesizes a counterpart that varies just the relevant signal
(stage / critical) and asks whether the reward responds the way the spec
dictates.

Five checks are run, covering both clauses of criterion 1 and one check per
remaining criterion in ``eval_spec.md``:

  [1] Criterion 1, first clause (stage 0 when critical < 0.01) — ablate each
      trajectory to hold critical < 0.01 and stage = 0 at every step; the
      reward should score the compliant copy strictly above the original
      whenever the original violates the criterion. Report the fraction of
      violations.

  [2] Criterion 1, second clause (higher critical is always worse) — add a
      sampled delta (~ U[0.01, 0.05]) to the total critical proportion at
      every step, leaving the stage and everything else unchanged. A reward
      aligned with this clause should score the higher-critical copy strictly
      below the original. Report the fraction of violations.

  [3] Criterion 2 (spend as little time at stage 4 as possible) — relabel k
      random non-stage-4 steps to stage 4, with k sampled uniformly in
      [1, n_non4]. A criterion-2-aligned reward should score the
      more-stage-4 copy strictly below the original. Report the fraction of
      violations.

  [4] Criterion 3 (avoid rapidly changing stage) — equality framing. For
      each trajectory, build:
        Side A (smooth): stage held constant; critical drifts by 0.001 over
          the trajectory.
        Side B (jittery): critical held constant; stage uses the original's
          stage sequence (must vary; trajectories with no stage change are
          skipped).
      A criterion-3-aligned reward should score the two equally — report
      mean within-pair |difference|, ideal 0.

  [5] Criterion 4 (single-step Δcritical/Δstage trade-off) — synthesize
      matched transition pairs from randomly-sampled "other features" (shared
      between the two sides within a pair):
        Side A: Δcritical = +0.01, stage 0 → 0.
        Side B: Δcritical = +0.005, stage 0 → 1.
      A criterion-4-aligned reward should score the two equally — report
      mean within-pair |difference|, ideal 0.

Obs accessors used here (env_config.OBS_DESCRIPTION): latest row is
``obs[-1, 0]``, length 13. Stage = index [1] (int 0..4), critical proportion
= [3] (alpha) + [8] (delta), deaths = [4] + [9]. Ablations apply *minimal
edits*: only the critical/stage indices are touched, leaving every other
field as in the source observation — so a strain's five-category proportions
will not generally re-sum to 1 after ablation. This matches the user's
"everything else stays the same" intent and avoids picking an arbitrary
redistribution rule.

Usage (from the repo root, with the `llm` conda env active):
    python eval_scripts/pandemic_human_eval_1.py --workspace workspaces/pandemic_human_eval_1
    python eval_scripts/pandemic_human_eval_1.py --workspace baseline_workspaces/pandemic_human_eval_1
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

EXPECTED_ENV = "pandemic_human_eval_1"

# Spec thresholds and ablation constants.
CRIT_THRESHOLD = 0.01            # criterion 1 (first clause) boundary
COMPLIANT_CRIT_TOTAL = 0.005     # target total critical for the [1] ablation (< 0.01)
RISING_CRIT_RANGE = (0.01, 0.05) # uniform delta added in the [2] ablation
LOCKDOWN_STAGE = 4               # criterion 2's "strictest" stage
A_RAMP_TOTAL = 0.001             # criterion 3 side A's total critical drift
CRIT4_DELTA_A = 0.01             # criterion 4 side A's Δcritical
CRIT4_DELTA_B = 0.005            # criterion 4 side B's Δcritical

DEFAULT_ABLATION_SAMPLES = 4     # sampled copies per trajectory for [2] and [3]
DEFAULT_CRIT4_SAMPLES = 200      # synthetic pairs for [5] (criterion 4)
DEFAULT_SEED = 0                 # reproducible sampling -> reproducible verdict


# ---- obs accessors -------------------------------------------------------

def _row(o):
    """Latest length-13 state vector for an observation array."""
    return np.asarray(o)[-1, 0]


def _stage(o):
    return int(round(float(_row(o)[1])))


def _critical(o):
    r = _row(o)
    return float(r[3]) + float(r[8])


# ---- minimal-edit obs mutations ------------------------------------------

def _set_stage(o, s):
    """Copy ``o`` with the regulation stage set to ``s`` on the latest row."""
    m = np.array(o, dtype=float, copy=True)
    m[-1, 0, 1] = float(s)
    return m


def _set_critical_total(o, target_total):
    """Copy ``o`` with the total critical proportion set to ``target_total``.

    Split evenly across the alpha [3] and delta [8] strains. Minimal edits:
    every other index is left untouched, so strain sums need not equal 1.
    """
    m = np.array(o, dtype=float, copy=True)
    m[-1, 0, 3] = target_total / 2.0
    m[-1, 0, 8] = target_total / 2.0
    return m


def _add_critical_total(o, delta_total):
    """Copy ``o`` with ``delta_total`` added to the total critical proportion.

    Split evenly across alpha [3] and delta [8]. Minimal edits, no resum.
    """
    m = np.array(o, dtype=float, copy=True)
    m[-1, 0, 3] += delta_total / 2.0
    m[-1, 0, 8] += delta_total / 2.0
    return m


# ---- whole-trajectory ablations ------------------------------------------

def _ablate_crit1a(traj):
    """Force critical = COMPLIANT_CRIT_TOTAL (< 0.01) and stage = 0 everywhere."""
    out = []
    for o, a, no in traj:
        o2 = _set_critical_total(_set_stage(o, 0), COMPLIANT_CRIT_TOTAL)
        n2 = _set_critical_total(_set_stage(no, 0), COMPLIANT_CRIT_TOTAL)
        out.append((o2, a, n2))
    return out


def _ablate_raise_critical(traj, delta_total):
    """Add ``delta_total`` to the critical proportion at every step (stage unchanged)."""
    return [(_add_critical_total(o, delta_total), a, _add_critical_total(no, delta_total))
            for o, a, no in traj]


def _ablate_more_stage4(traj, k, rng):
    """Relabel ``k`` random non-stage-4 steps to stage 4 (modifies obs + next_obs)."""
    non4 = [i for i, (_, _, no) in enumerate(traj) if _stage(no) != LOCKDOWN_STAGE]
    if not non4:
        return None
    k = min(k, len(non4))
    picked = set(rng.choice(non4, size=k, replace=False).tolist())
    out = []
    for i, (o, a, no) in enumerate(traj):
        if i in picked:
            out.append((_set_stage(o, LOCKDOWN_STAGE), a, _set_stage(no, LOCKDOWN_STAGE)))
        else:
            out.append((o, a, no))
    return out


def _satisfies_crit1a(traj):
    """At every step with critical < 0.01, the stage must be 0."""
    for _, _, no in traj:
        if _critical(no) < CRIT_THRESHOLD and _stage(no) != 0:
            return False
    return True


# ---- criterion 3 (equality framing) --------------------------------------

def _build_crit3_sides(traj):
    """Return (A, B) trajectories for the smooth/jittery comparison, or (None, None).

    Side A: stage held constant (at the original's first stage); critical
    follows a linear ramp from the original's first-step critical to that
    value + A_RAMP_TOTAL over the trajectory.

    Side B: critical held constant (at the original's first-step critical);
    stage uses the original's stage sequence — must contain change, else
    (None, None) is returned (the comparison would be ill-defined).
    """
    T = len(traj)
    if T < 2:
        return None, None
    first_o, _, _ = traj[0]
    base_stage = _stage(first_o)
    base_crit = _critical(first_o)
    stages_no = [_stage(no) for _, _, no in traj]
    if all(s == stages_no[0] for s in stages_no) and base_stage == stages_no[0]:
        return None, None
    A, B = [], []
    for i, (o, a, no) in enumerate(traj):
        # Side A: stage flat, critical linear ramp over the trajectory.
        c_o = base_crit + A_RAMP_TOTAL * (i / T)
        c_n = base_crit + A_RAMP_TOTAL * ((i + 1) / T)
        oA = _set_critical_total(_set_stage(o, base_stage), c_o)
        nA = _set_critical_total(_set_stage(no, base_stage), c_n)
        A.append((oA, a, nA))
        # Side B: original stage sequence kept; critical flat.
        oB = _set_critical_total(o, base_crit)
        nB = _set_critical_total(no, base_crit)
        B.append((oB, a, nB))
    return A, B


# ---- criterion 4 (equality framing, synthetic transitions) ---------------

def _build_crit4_pair(rng):
    """Return matched (A_transition, B_transition); other obs features shared.

    Each transition is a single ``(obs, act, next_obs)`` triple. Both obs and
    next_obs within a transition (and across the two sides of the pair) share
    the same randomly-sampled length-13 base vector, so the *only* differences
    between A and B are the controlled stage and critical changes. The action
    is set to the stage that next_obs lands at.
    """
    base = rng.uniform(0.0, 1.0, size=(1, 1, 13))
    c0 = float(rng.uniform(0.0, 0.05))  # leaves room below 1 for c0 + 0.01

    def make(s_obs, s_no, c_obs, c_no):
        o = _set_critical_total(_set_stage(base, s_obs), c_obs)
        n = _set_critical_total(_set_stage(base, s_no), c_no)
        return o, int(s_no), n

    # Side A: stage 0 -> 0, critical c0 -> c0 + 0.01
    A = make(0, 0, c0, c0 + CRIT4_DELTA_A)
    # Side B: stage 0 -> 1, critical c0 -> c0 + 0.005
    B = make(0, 1, c0, c0 + CRIT4_DELTA_B)
    return A, B


# ---- driver --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arg(parser)
    parser.add_argument(
        "--ablation-samples", type=int, default=DEFAULT_ABLATION_SAMPLES,
        help="Sampled ablation copies per trajectory for checks [2] and [3] "
             "(default 4).",
    )
    parser.add_argument(
        "--crit4-samples", type=int, default=DEFAULT_CRIT4_SAMPLES,
        help="Synthetic transition pairs for the criterion-4 equality check "
             "(default 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed (default 0). Independent streams are spawned for the "
             "trajectory-ablation sampling and the criterion-4 synthesis, so "
             "changing --ablation-samples does not shift the criterion-4 draws.",
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
            f"`python -m runtime.run_eval_rollout_parallel --eval-dir eval_scripts/{EXPECTED_ENV} "
            f"--env-workspace workspaces/{EXPECTED_ENV} --episodes-per-policy 10`."
        )

    def reward_of(traj):
        return float(sum(designed_reward_fn(o, a, no) for o, a, no in traj))

    trajs = []
    for path in paths:
        with open(path, "rb") as fh:
            traj = pickle.load(fh)["trajectory"]
        if traj:
            trajs.append({
                "name": os.path.basename(path),
                "traj": traj,
                "reward": reward_of(traj),
            })

    # Independent RNG streams for trajectory ablations vs criterion 4 synthesis,
    # so altering --ablation-samples doesn't shift the criterion-4 results.
    ss = np.random.SeedSequence(args.seed)
    rng_abl, rng4 = (np.random.default_rng(s) for s in ss.spawn(2))

    # ---- report header ---------------------------------------------------
    print("=" * 78)
    print("SPEC EVALUATION — designed reward vs. eval_spec.md (per-criterion)")
    print(f"Workspace: {ws_path}")
    print("=" * 78)
    with open("eval_spec.md") as fh:
        print(fh.read().rstrip())
    print("-" * 78)
    print(f"Trajectories analyzed: {len(trajs)}")
    print(f"Seed: {args.seed}; trajectory ablations: {args.ablation_samples}/traj; "
          f"criterion-4 synthetic pairs: {args.crit4_samples}.")
    print("-" * 78)

    # All-pairs |Δreturn| scale reference over the collected trajectories,
    # used by the equality-framed checks [4] and [5].
    returns = np.array([t["reward"] for t in trajs])
    pair_abs = np.array([abs(returns[i] - returns[j])
                         for i in range(len(returns))
                         for j in range(i + 1, len(returns))])
    all_pairs_mu = float(pair_abs.mean()) if pair_abs.size else 0.0
    all_pairs_sd = float(pair_abs.std()) if pair_abs.size else 0.0

    # ---- [1] criterion 1, first clause: stage 0 when critical < 0.01 ----
    print("[1] Criterion 1 (first clause) — stage 0 when critical < 0.01")
    print(f"    Ablate every trajectory to: critical = {COMPLIANT_CRIT_TOTAL:g} total at "
          f"every step, stage = 0 at every step.")
    print("    A reward aligned with this clause should score the compliant copy strictly")
    print("    above any original that violates the criterion.")
    nonc1 = []
    for t in trajs:
        if _satisfies_crit1a(t["traj"]):
            continue
        a = _ablate_crit1a(t["traj"])
        nonc1.append({"r_real": t["reward"], "r_abl": reward_of(a)})
    inv1 = sum(1 for r in nonc1 if r["r_abl"] <= r["r_real"])
    n1 = len(nonc1)
    print(f"    Compliant-below-violator inversions: {inv1}/{n1}")
    print("-" * 78)

    # ---- [2] criterion 1, second clause: higher critical is always worse -
    print("[2] Criterion 1 (second clause) — higher critical is always worse")
    print(f"    Ablate every trajectory by adding δ ~ U[{RISING_CRIT_RANGE[0]:g}, "
          f"{RISING_CRIT_RANGE[1]:g}] to the total critical proportion at every step.")
    print("    Stage and every other obs field are unchanged. A reward aligned with this")
    print("    clause should score the higher-critical copy strictly below the original.")
    sampled2 = []
    for t in trajs:
        for _ in range(args.ablation_samples):
            delta = float(rng_abl.uniform(*RISING_CRIT_RANGE))
            a = _ablate_raise_critical(t["traj"], delta)
            sampled2.append({"r_real": t["reward"], "r_abl": reward_of(a)})
    inv2 = sum(1 for r in sampled2 if r["r_abl"] >= r["r_real"])
    n2 = len(sampled2)
    print(f"    Compliant-below-violator inversions: {inv2}/{n2}")
    print("-" * 78)

    # ---- [3] criterion 2: more time in stage 4 ⇒ reward drops -----------
    print("[3] Criterion 2 — more time in stage 4 ⇒ reward should drop")
    print("    Ablate each trajectory by relabeling k random non-stage-4 steps to stage 4,")
    print("    with k ~ Uniform{1..n_non4} (anywhere from one step to all non-stage-4 steps).")
    print("    A criterion-2-aligned reward should score the more-stage-4 copy strictly below.")
    sampled3 = []
    for t in trajs:
        non4 = [i for i, (_, _, no) in enumerate(t["traj"])
                if _stage(no) != LOCKDOWN_STAGE]
        if not non4:
            continue
        for _ in range(args.ablation_samples):
            k = int(rng_abl.integers(1, len(non4) + 1))
            a = _ablate_more_stage4(t["traj"], k, rng_abl)
            sampled3.append({"r_real": t["reward"], "r_abl": reward_of(a)})
    inv3 = sum(1 for r in sampled3 if r["r_abl"] >= r["r_real"])
    n3 = len(sampled3)
    print(f"    Compliant-below-violator inversions: {inv3}/{n3}")
    print("-" * 78)

    # ---- [4] criterion 3: smooth vs jittery (equality framing) ----------
    print("[4] Criterion 3 — smooth stage vs jittery stage (equality framing)")
    print(f"    Side A (smooth): stage held constant; critical drifts by {A_RAMP_TOTAL:g} "
          f"over the trajectory.")
    print("    Side B (jittery): critical held constant; stage uses the original's stage")
    print("    sequence (must contain change; trajectories with no stage change are skipped).")
    print("    A criterion-3-aligned reward should score them equally.")
    diffs4 = []
    for t in trajs:
        A, B = _build_crit3_sides(t["traj"])
        if A is None:
            continue
        diffs4.append(reward_of(A) - reward_of(B))
    if diffs4 and pair_abs.size:
        within4 = float(np.abs(np.array(diffs4)).mean())
        print(f"    All-pairs mean |Δreturn| over the {len(trajs)} trajectories: "
              f"{all_pairs_mu:.3f} (sd {all_pairs_sd:.3f})")
        if all_pairs_sd > 1e-12:
            delta = within4 - all_pairs_mu
            print(f"    Within-pair |Δ| spread: {delta / all_pairs_sd:+.2f} sd "
                  f"from the all-pairs mean")
        else:
            print("    Within-pair |Δ| spread: undefined "
                  "(all trajectory pairs have equal return; sd = 0)")
    else:
        print("    No usable trajectories — cannot evaluate.")
    print("-" * 78)

    # ---- [5] criterion 4: single-step Δcritical/Δstage trade-off ---------
    print("[5] Criterion 4 — single-step Δcritical/Δstage trade-off (equality framing)")
    print(f"    Side A: Δcritical = +{CRIT4_DELTA_A:g}, stage 0 → 0 (no change).")
    print(f"    Side B: Δcritical = +{CRIT4_DELTA_B:g}, stage 0 → 1 (increase).")
    print("    Other observation features are randomly sampled and shared between A and B")
    print("    within each pair, so the only differences are the controlled stage / Δcritical.")
    print("    A criterion-4-aligned reward should score them equally.")
    diffs5 = []
    for _ in range(args.crit4_samples):
        (oA, aA, nA), (oB, aB, nB) = _build_crit4_pair(rng4)
        rA = float(designed_reward_fn(oA, aA, nA))
        rB = float(designed_reward_fn(oB, aB, nB))
        diffs5.append(rA - rB)
    if pair_abs.size:
        within5 = float(np.abs(np.array(diffs5)).mean())
        print(f"    All-pairs mean |Δreturn| over the {len(trajs)} trajectories: "
              f"{all_pairs_mu:.3f} (sd {all_pairs_sd:.3f})")
        if all_pairs_sd > 1e-12:
            delta = within5 - all_pairs_mu
            print(f"    Within-pair |Δ| spread: {delta / all_pairs_sd:+.2f} sd "
                  f"from the all-pairs mean")
        else:
            print("    Within-pair |Δ| spread: undefined "
                  "(all trajectory pairs have equal return; sd = 0)")
    else:
        print("    Only one trajectory collected — no all-pairs baseline.")
    print("-" * 78)

    # ---- totals --------------------------------------------------------
    print(f"Total compliant-below-violator inversions across [1]+[2]+[3]: "
          f"{inv1 + inv2 + inv3}/{n1 + n2 + n3}")
    print("=" * 78)


if __name__ == "__main__":
    main()
