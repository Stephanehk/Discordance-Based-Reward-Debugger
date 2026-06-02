# Submitted designed reward — workspaces/wipe

Submitted at: 2026-05-27T20:01:48Z (UTC)

```python
"""Designed reward for the robosuite Wipe task.

Pulled verbatim from robosuite's own default shaped reward for the
``Wipe`` environment (``robosuite.environments.manipulation.wipe.Wipe.reward``)
with the constants from ``DEFAULT_WIPE_CONFIG``. This is the initial
reward to repair — every weight and term below is editable.

The per-step reward components are appended to the observation by
``env_config._WipeRewardEnv``; see ``OBS_DESCRIPTION`` for the trailing
10-slot layout. ``designed_reward_fn`` reads those slots and combines
them into the env's default formula:

  - If the arm collided with the table or hit a joint limit this step,
    the per-step reward is clamped to ``ARM_LIMIT_COLLISION_PENALTY``
    and no other terms apply.

  - Otherwise:
      + UNIT_WIPED_REWARD for every dirt marker wiped on this step.
      + DISTANCE_MULTIPLIER * (1 - tanh(DISTANCE_TH_MULTIPLIER * d))
        reaching reward (only while dirt remains; d = mean distance
        from the wiper to the remaining markers).
      + WIPE_CONTACT_REWARD when the wiper is touching the table.
      - EXCESS_FORCE_PENALTY_MUL * total_force_ee when the end-effector
        force exceeds PRESSURE_THRESHOLD_MAX (otherwise, if it exceeds
        PRESSURE_THRESHOLD with > 1 sim contact, a small force-bonus
        applies; with > 50 contacts there is an additional bonus).
      - EE_ACCEL_PENALTY * mean |end-effector acceleration|.
      + TASK_COMPLETE_REWARD if all markers have been wiped.

The unnormalised sum is then scaled by the env's reward-normalisation
factor (``HORIZON / (num_markers * UNIT_WIPED_REWARD + HORIZON *
(WIPE_CONTACT_REWARD + TASK_COMPLETE_REWARD))``), matching the env's
own normalisation so per-step magnitudes stay comparable to what the
underlying robosuite ``step`` returns.
"""

import numpy as np


# Weights — DEFAULT_WIPE_CONFIG values from robosuite's Wipe env.
ARM_LIMIT_COLLISION_PENALTY = -10.0  # per-step penalty when the arm collides
                                     # with the table or saturates a joint limit;
                                     # nullifies all other reward terms this step.
WIPE_CONTACT_REWARD          =  0.01 # per-step bonus when the wiper is touching
                                     # the table (encourages staying on-surface).
UNIT_WIPED_REWARD            = 600.0 # per dirt marker cleaned this step.
                                     # Raised from the env's default of 50 in two
                                     # passes. The 50→75 bump fixed the Goodhart
                                     # where 0–3-wipe pulsed trajectories beat 12–
                                     # 16-wipe ones. After a fresh round surfaced
                                     # zero-motion 500-step trajectories scoring
                                     # +10 reward while real low-yield cleaning
                                     # earned only +5–7, this second bump (75→600)
                                     # is the smallest weight that flips both that
                                     # zero-motion-beats-4-wipes preference *and*
                                     # the associated lottery where a 5% chance of
                                     # real cleaning beats a sure 4-marker pass.
                                     # The lottery flip in particular is what
                                     # requires the multiplier this high — the
                                     # 95%-idle branch is reward-inflated and only
                                     # dominates the lottery up to U ≈ 488.
EE_ACCEL_PENALTY             =  0.0  # multiplier on mean |EE acceleration|;
                                     # zero in the default config.
EXCESS_FORCE_PENALTY_MUL     =  0.05 # per-N penalty when force exceeds the
                                     # safe upper bound (PRESSURE_THRESHOLD_MAX).
DISTANCE_MULTIPLIER          =  5.0  # max reaching reward when the wiper sits
                                     # directly on the dirt centroid.
DISTANCE_TH_MULTIPLIER       =  5.0  # sharpness of the tanh inside the
                                     # reaching reward (higher = tighter falloff).

# Force thresholds (Newtons) — robosuite Wipe defaults.
PRESSURE_THRESHOLD     =  0.5
PRESSURE_THRESHOLD_MAX = 60.0

# Task-complete bonus formula from the env: encourages finishing over
# loitering at maximum contact reward for the whole horizon.
TASK_COMPLETE_REWARD = UNIT_WIPED_REWARD * (WIPE_CONTACT_REWARD + 0.5)

# If True, scale the per-step reward by the env's normalisation factor so
# the totals are comparable to what robosuite's own ``step`` returns.
APPLY_REWARD_NORMALIZATION = True
REWARD_SCALE = 1.0


# ----- Layout of the appended reward-component channels in obs ---------------
# Must match ``env_config._WipeRewardEnv._reward_components``.
IDX_TOTAL_WIPED        = -10
IDX_NEW_WIPED          = -9
IDX_MEAN_DISTANCE      = -8
IDX_HAS_CONTACT        = -7
IDX_NCON               = -6
IDX_TOTAL_FORCE_EE     = -5
IDX_MEAN_EE_ACC        = -4
IDX_COLLISION_OR_QLIM  = -3
IDX_NUM_MARKERS        = -2
IDX_HORIZON            = -1
# -----------------------------------------------------------------------------


def designed_reward_fn(obs, act, next_obs) -> float:
    """Return the per-step designed reward for one (obs, act, next_obs) step."""
    s = np.asarray(next_obs, dtype=np.float64)

    # If the arm collided with the table or hit a joint limit, the env
    # returns ``arm_limit_collision_penalty`` for the step and discards
    # every other component. Mirror that branch.
    if s[IDX_COLLISION_OR_QLIM] > 0.5:
        unnormalised = ARM_LIMIT_COLLISION_PENALTY
    else:
        new_wiped       = float(s[IDX_NEW_WIPED])
        total_wiped     = float(s[IDX_TOTAL_WIPED])
        mean_distance   = float(s[IDX_MEAN_DISTANCE])
        has_contact     = s[IDX_HAS_CONTACT] > 0.5
        ncon            = float(s[IDX_NCON])
        total_force_ee  = float(s[IDX_TOTAL_FORCE_EE])
        mean_ee_acc     = float(s[IDX_MEAN_EE_ACC])
        num_markers     = float(s[IDX_NUM_MARKERS])

        unnormalised = 0.0

        # +UNIT_WIPED_REWARD per dirt marker cleaned during this step.
        unnormalised += UNIT_WIPED_REWARD * new_wiped

        # Reaching reward — only while dirt remains.
        if total_wiped < num_markers:
            unnormalised += DISTANCE_MULTIPLIER * (
                1.0 - np.tanh(DISTANCE_TH_MULTIPLIER * mean_distance)
            )

        # Wiper-on-table contact bonus.
        if ncon != 0 and has_contact:
            unnormalised += WIPE_CONTACT_REWARD

        # Force-related terms (excessive-force penalty OR press-down bonus).
        if total_force_ee > PRESSURE_THRESHOLD_MAX:
            unnormalised -= EXCESS_FORCE_PENALTY_MUL * total_force_ee
        elif total_force_ee > PRESSURE_THRESHOLD and ncon > 1:
            unnormalised += WIPE_CONTACT_REWARD + 0.01 * total_force_ee
            if ncon > 50:
                unnormalised += 10.0 * WIPE_CONTACT_REWARD

        # Smoothness penalty (zero in the default config).
        unnormalised -= EE_ACCEL_PENALTY * mean_ee_acc

        # Task-complete bonus if the table is now clean.
        if total_wiped >= num_markers:
            unnormalised += TASK_COMPLETE_REWARD

    if APPLY_REWARD_NORMALIZATION:
        horizon = float(s[IDX_HORIZON])
        num_markers = float(s[IDX_NUM_MARKERS])
        denom = (
            num_markers * UNIT_WIPED_REWARD
            + horizon * (WIPE_CONTACT_REWARD + TASK_COMPLETE_REWARD)
        )
        norm = horizon / denom if denom > 0 else 1.0
        return float(REWARD_SCALE * norm * unnormalised)

    return float(REWARD_SCALE * unnormalised)
```
