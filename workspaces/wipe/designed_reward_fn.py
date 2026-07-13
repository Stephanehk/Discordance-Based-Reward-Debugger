"""Repaired reward for the robosuite Wipe task (Panda arm, OSC_POSE control).

The agent controls a Panda arm holding a wiping pad and must clean small
"dirt" markers off a table over a 500-step episode.

------------------------------------------------------------------------
OBSERVATION (1-D float32 vector, length = proprio_dim + 10)
------------------------------------------------------------------------
Trailing 10 channels carry per-step reward components (extracted from sim
internals at step time). Index them off the END of the vector:
  next_obs[-10] = total_wiped         - cumulative # markers wiped so far
  next_obs[ -9] = new_wiped           - # markers wiped on this step
  next_obs[ -8] = mean_distance       - mean dist from wiper to remaining dirt
  next_obs[ -7] = has_gripper_contact - 1.0 iff wiper is touching the table
  next_obs[ -6] = ncon                - # of active MuJoCo contacts
  next_obs[ -5] = total_force_ee      - max end-effector force magnitude (N)
  next_obs[ -4] = mean_ee_acc         - mean |end-effector acceleration|
  next_obs[ -3] = collision_or_qlim   - 1.0 if arm collided / hit joint limit
  next_obs[ -2] = num_markers         - constant: total dirt markers spawned
  next_obs[ -1] = horizon             - constant: episode horizon (500)

------------------------------------------------------------------------
WHAT CHANGED vs. the robosuite default shaped reward, and why
------------------------------------------------------------------------
The default shaped Wipe reward paid two large *per-step proxy* terms that
accrue every step regardless of whether any dirt is removed:

  * a distance-shaping term  DISTANCE_MULTIPLIER * (1 - tanh(k * mean_dist))
    — up to +5/step just for hovering the pad near the remaining dirt, and
  * a press-into-table bonus  (contact + 0.01*force, +extra when ncon>50)
    — paid for keeping the pad pressed on the table.

Because both accrue without cleaning, a policy that gently rests / drifts
on the table can bank more reward than one that actually removes dirt.
That is exactly what the shown comparisons exposed: resting near the dirt
(0 cleaned) outscoring a real 22-marker wipe; a 25-marker gentle press
outscoring a 47-marker sweep; a 38-marker gentle run outscoring a
53-marker sweep; and four lotteries where a certain low/zero-cleaning
trajectory was preferred over a gamble that cleans most of the table.

This repaired reward makes **actual dirt removal the dominant driver** by
removing the two farmable proxy terms. It keeps:
  * the per-marker cleaning reward (the true outcome),
  * the collision / joint-limit gate,
  * the excess-force penalty (so genuinely over-forceful wiping is still
    discouraged — this is what keeps a gentle equal-cleaning wipe ranked
    above a high-force one), and
  * the task-completion bonus.

The end-effector acceleration penalty is retained with its original
weight (0 by default) so the structure is unchanged there.
------------------------------------------------------------------------
"""

import numpy as np

# Reward weights, pulled straight from robosuite's default Wipe config.
try:
    from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG as _CFG
except Exception:  # pragma: no cover
    _CFG = {
        "arm_limit_collision_penalty": -10.0,
        "wipe_contact_reward": 0.01,
        "unit_wiped_reward": 50.0,
        "ee_accel_penalty": 0.0,
        "excess_force_penalty_mul": 0.05,
        "distance_multiplier": 5.0,
        "distance_th_multiplier": 5.0,
        "pressure_threshold": 0.5,
        "pressure_threshold_max": 60.0,
        "num_markers": 100,
    }

ARM_LIMIT_COLLISION_PENALTY = float(_CFG["arm_limit_collision_penalty"])
WIPE_CONTACT_REWARD = float(_CFG["wipe_contact_reward"])
UNIT_WIPED_REWARD = float(_CFG["unit_wiped_reward"])
EE_ACCEL_PENALTY = float(_CFG["ee_accel_penalty"])
EXCESS_FORCE_PENALTY_MUL = float(_CFG["excess_force_penalty_mul"])
PRESSURE_THRESHOLD_MAX = float(_CFG["pressure_threshold_max"])

# task_complete_reward exactly as computed in Wipe.__init__.
TASK_COMPLETE_REWARD = UNIT_WIPED_REWARD * (WIPE_CONTACT_REWARD + 0.5)

# env_config builds the env with reward_scale=1.0.
REWARD_SCALE = 1.0


def designed_reward_fn(obs, act, next_obs) -> float:
    """Repaired Wipe reward: cleaning-dominant, no farmable proxy terms.

    Per non-colliding step the reward is
        UNIT_WIPED_REWARD * new_wiped                 # true outcome (dominant)
      + WIPE_CONTACT_REWARD  if in contact            # tiny, negligible
      - EXCESS_FORCE_PENALTY_MUL * force  if force>60 # discourage over-forcing
      - EE_ACCEL_PENALTY * mean_ee_acc                # weight 0 by default
      + TASK_COMPLETE_REWARD  once all markers wiped
    A collision / joint-limit hit replaces the whole step with the fixed
    penalty, as in the original. The distance-shaping and press-into-table
    bonuses of the default reward are intentionally omitted.
    """
    s = np.asarray(next_obs, dtype=np.float64)

    total_wiped = s[-10]
    new_wiped = s[-9]
    has_gripper_contact = s[-7] > 0.5
    ncon = s[-6]
    total_force_ee = s[-5]
    mean_ee_acc = s[-4]
    collision_or_qlim = s[-3] > 0.5
    num_markers = s[-2]
    horizon = s[-1]

    reward = 0

    # Collisions / joint-limit hits zero out all other reward this step.
    if collision_or_qlim:
        reward = ARM_LIMIT_COLLISION_PENALTY
    else:
        # Reward each dirt marker actually removed this step (the true outcome).
        reward += UNIT_WIPED_REWARD * new_wiped

        # Tiny flat contact reward (kept from the default; negligible).
        if ncon != 0 and has_gripper_contact:
            reward += WIPE_CONTACT_REWARD

        # Penalty for excessive end-effector force. Over-forceful wiping is
        # still discouraged, which is what keeps a gentle wipe ranked above a
        # high-force one at equal cleaning. (No press-into-table bonus below
        # the threshold — that proxy is removed.)
        if total_force_ee > PRESSURE_THRESHOLD_MAX:
            reward -= EXCESS_FORCE_PENALTY_MUL * total_force_ee

        # Penalize large accelerations (weight 0 by default).
        reward -= EE_ACCEL_PENALTY * mean_ee_acc

        # Final bonus if the whole table is clean.
        if total_wiped >= num_markers:
            reward += TASK_COMPLETE_REWARD

    # Normalize per-step rewards by the theoretical best episode return, exactly
    # as the env does. (Distance/press terms are gone from the numerator above
    # but the normalization constant is unchanged, matching the env's formula.)
    reward_normalization_factor = horizon / (
        num_markers * UNIT_WIPED_REWARD
        + horizon * (WIPE_CONTACT_REWARD + TASK_COMPLETE_REWARD)
    )
    if REWARD_SCALE:
        reward *= REWARD_SCALE * reward_normalization_factor

    return float(reward)
