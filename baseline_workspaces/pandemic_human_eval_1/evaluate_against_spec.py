"""Evaluate this workspace's designed reward against its eval spec.

This script is specific to the ``pandemic_human_eval_1`` evaluation workspace
and lives inside it. It is a **persistent eval artifact** (like
``env_config.py`` / ``eval_spec.md`` / the starting ``designed_reward_fn.py``):
it must be kept across workspace resets, not treated as authored/collected
output.

The check is **ranking-based**, adapted to this environment. The spec
constrains the *regulation-stage decision as a function of the critical-case
proportion* — not the outbreak outcome, which the agent cannot set directly.
A global "every compliant trajectory out-scores every violating one" test
(as in the merge workspace) would therefore misfire here: even a correct
reward scores a severe-outbreak trajectory below a mild one. So the test is
run **pairwise**: for every collected trajectory we synthesize a
spec-compliant copy by ablating only the regulation stage at each step to the
value the spec prescribes for that step's observed critical proportion
(leaving the critical-case trace untouched), and check that the designed
reward scores the spec-corrected copy at least as high as the original. A
spec-aligned reward should never prefer the original stage decisions over the
spec-corrected ones.

Checkable structural criteria operationalized here (from eval_spec.md):
  - stage 0 while critical < 0.01, and the stage rises monotonically with the
    critical proportion (criterion 1);
  - the strictest lockdown (stage 4) is reached only in the top critical band,
    i.e. used as little as possible (criterion 2);
  - stage 2 is never used (criterion 4).
The target-stage mapping below encodes these. The marginal trade-offs
(criteria 3 and 5) are not separately scored by this harness.

Observation layout (env_config.OBS_DESCRIPTION): latest row obs[-1, 0],
stage = index [1], critical proportion = index [3] (alpha) + [8] (delta).

Usage (from the repo root, with the `llm` conda env active):
    python baseline_workspaces/pandemic_human_eval_1/evaluate_against_spec.py
"""
from __future__ import annotations

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

# Spec-prescribed regulation stage as a function of the critical-case
# proportion: stage 0 below 0.01 (criterion 1), rising monotonically with
# critical, skipping stage 2 entirely (criterion 4), and reaching the
# strictest lockdown only in the top band (criterion 2).
_STAGE_BANDS = [          # (critical strict-upper-bound, stage)
    (0.01, 0),
    (0.02, 1),
    (0.04, 3),            # note: stage 2 is skipped
    (float("inf"), 4),
]


def _row(o):
    return np.asarray(o)[-1, 0]


def _critical(o):
    r = _row(o)
    return float(r[3]) + float(r[8])


def target_stage(critical: float) -> int:
    for upper, stage in _STAGE_BANDS:
        if critical < upper:
            return stage
    return _STAGE_BANDS[-1][1]


def _ablate_obs(o):
    """Copy an observation with the regulation stage forced to spec."""
    m = np.array(o, dtype=float, copy=True)
    m[-1, 0, 1] = float(target_stage(_critical(o)))
    return m


def _ablate(traj):
    return [(_ablate_obs(o), a, _ablate_obs(no)) for o, a, no in traj]


def _violating_steps(traj):
    """How many steps run a regulation stage other than the spec-prescribed one."""
    n = 0
    for _, _, no in traj:
        stage = int(round(float(_row(no)[1])))
        if stage != target_stage(_critical(no)):
            n += 1
    return n


def main():
    activate(_WORKSPACE_DIR)

    if not os.path.exists("eval_spec.md"):
        raise SystemExit(
            "No eval_spec.md in this workspace — the spec evaluator only "
            "applies to evaluation workspaces."
        )

    from designed_reward_fn import designed_reward_fn

    paths = sorted(glob.glob("gather_evidence/round_*_trajectories/*.pkl"))
    if not paths:
        raise SystemExit("No trajectories collected yet — run a round first.")

    def reward_of(traj):
        return float(sum(designed_reward_fn(o, a, no) for o, a, no in traj))

    rows = []
    for path in paths:
        with open(path, "rb") as fh:
            traj = pickle.load(fh)["trajectory"]
        if not traj:
            continue
        rows.append({
            "name": os.path.basename(path),
            "steps": len(traj),
            "bad": _violating_steps(traj),
            "r_real": reward_of(traj),
            "r_fixed": reward_of(_ablate(traj)),
        })

    # ---- report ----
    print("=" * 78)
    print("SPEC EVALUATION (pairwise ranking) — designed reward vs. eval_spec.md")
    print("=" * 78)
    with open("eval_spec.md") as fh:
        print(fh.read().rstrip())
    print("-" * 78)
    print("Test: for every collected trajectory, force the regulation stage at each")
    print("step to the spec-prescribed stage for that step's critical proportion")
    print("(leaving the outbreak trace unchanged), and check that the designed")
    print("reward scores the spec-corrected trajectory at least as high as the")
    print("original.")
    print("Target stage by critical proportion: "
          "<0.01 -> 0, <0.02 -> 1, <0.04 -> 3, else 4  (stage 2 never used).")
    print("-" * 78)

    need_fix = [r for r in rows if r["bad"] > 0]
    inversions = [r for r in need_fix if r["r_fixed"] < r["r_real"] - 1e-9]

    print(f"Trajectories: {len(rows)}  "
          f"(already spec-matching at every step: {len(rows) - len(need_fix)}; "
          f"needing stage correction: {len(need_fix)})")
    for r in rows:
        flag = ""
        if r["bad"] > 0:
            flag = ("  <-- INVERSION" if r["r_fixed"] < r["r_real"] - 1e-9
                    else "  ok")
        print(f"  {r['name']:<52} bad-steps {r['bad']:>3}/{r['steps']:<4} "
              f"real {r['r_real']:>10.2f}  spec-corrected {r['r_fixed']:>10.2f}{flag}")
    print("-" * 78)

    if not need_fix:
        print("No collected trajectory deviated from the spec-prescribed stages, so")
        print("the ranking test is trivially satisfied (collect more varied evidence")
        print("for a stronger check).")
        print("=" * 78)
        return

    passed = not inversions
    print(f"Spec-corrected-below-original inversions: {len(inversions)}/{len(need_fix)}")
    print("RESULT: the designed reward never prefers the original stage decisions "
          f"over the spec-corrected ones: {'YES' if passed else 'NO'}")
    if not passed:
        print("  (At least one trajectory scores higher with its original, "
              "spec-violating stage decisions than after correcting them; "
              "optimizing this reward need not satisfy the spec.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
