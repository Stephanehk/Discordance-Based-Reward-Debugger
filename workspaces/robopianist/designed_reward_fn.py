"""Designed reward for RoboPianist-debug-TwinkleTwinkleRousseau-v0.

Two Shadow dexterous hands play "Twinkle Twinkle Little Star" on an 88-key
piano, pressing the correct keys at the correct times while following per-step
fingering labels and a sustain pedal.

Each transition gives you `(obs, act, next_obs)`. Both `obs` and `next_obs`
are the flattened observation vector; `act` is the 45-dim action. The physical
quantities the reward is most likely to read are appended to the END of the
observation vector (slice them off the tail). The trailing block, in order, is:

  goal_current (89)     : the per-step target [88 key flags (0/1) + 1 sustain flag]
  norm_state (88)       : each key's depth normalized to [0, 1] (how far pressed)
  activation (88)       : per-key "pressed" flag (0/1)
  sustain_activation (1): sustain-pedal engaged flag (0/1)
  fingertip_pos (30)    : 10 fingertip world positions, meters
                          (rh thumb..pinky, lh thumb..pinky; xyz each)
  finger_target_pos (30): per finger, its assigned target key's contact point
                          in meters (0 if that finger is idle this step)
  finger_active (10)    : per finger, 1 if it is assigned a key this step
  hand_power (2)        : summed actuator mechanical power [right, left]
  forearm_no_collision (1): 1 if the two forearms are NOT touching

The action vector `act` is:
  [0:22]  right-hand actuators (20 finger/wrist joints + 2 forearm slides)
  [22:44] left-hand actuators (same layout)
  [44]    sustain-pedal command, in [0, 1]

Fill in the body. The signature must be unchanged.
"""

import numpy as np

# Size of the appended physical-readout block (tail of the observation vector).
# Slice the quantities you need with `np.asarray(next_obs)[-_AUG_DIM:]`, then
# index within that block using the offsets documented above. Must stay in sync
# with env_config.py's _AUG_SIZES.
_AUG_DIM = 339


"""Designed reward for `robopianist` (the env's default dense reward).

Two Shadow hands play a MIDI song on a piano. This reproduces robopianist's
built-in dense reward — the unweighted sum of five terms — from a single
`(obs, act, next_obs)` transition:

  key_press  : press the keys that should be down now, and no others,
  sustain    : engage the sustain pedal when the song calls for it,
  energy     : a small penalty on total actuator power,
  fingering  : keep each active fingertip near its assigned key,
  forearm    : don't let the two forearms collide.

All five are computed from the physical-readout block that env_config appends to
the end of the observation (see env_config.OBS_DESCRIPTION). Every term uses
dm_control's `tolerance` shaping (a Gaussian falloff outside a tight band), which
is reimplemented here in numpy. The reward is a function of the resulting state
(next_obs); `obs`/`act` are unused, matching the env, which scores each control
step from its post-step state.
"""

import numpy as np

_AUG_DIM = 339               # size of the appended physical-readout block
_ENERGY_COEF = 5e-3
_KEY_T = 0.05                # key/sustain "close enough" band
_FINGER_T = 0.01             # fingertip "close enough" band (meters)


def _tolerance(x, lo, hi, margin, value_at_margin=0.1):
    """dm_control rewards.tolerance with a Gaussian sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    in_bounds = (x >= lo) & (x <= hi)
    d = np.where(x < lo, lo - x, x - hi) / margin
    scale = np.sqrt(-2.0 * np.log(value_at_margin))
    s = np.exp(-0.5 * (d * scale) ** 2)
    return np.where(in_bounds, 1.0, s)


def designed_reward_fn(obs, act, next_obs):
    a = np.asarray(next_obs, dtype=np.float64)[-_AUG_DIM:]

    # Unpack the appended physical-readout block.
    goal_current = a[0:89]
    norm_state = a[89:177]
    activation = a[177:265]
    sustain_activation = a[265]
    fingertip_pos = a[266:296].reshape(2, 5, 3)
    finger_target_pos = a[296:326].reshape(2, 5, 3)
    finger_active = a[326:336].reshape(2, 5)
    hand_power = a[336:338]
    forearm_no_collision = a[338]

    goal_keys = goal_current[:88]
    sustain_target = goal_current[88]

    # --- key press: 0.5 for pressing the right keys, 0.5 for no false presses.
    key_press = 0.0
    on = np.flatnonzero(goal_keys)
    if on.size > 0:
        rews = _tolerance(goal_keys[on] - norm_state[on], 0.0, _KEY_T, _KEY_T * 10)
        key_press += 0.5 * float(rews.mean())
    off = np.flatnonzero(1.0 - goal_keys)
    key_press += 0.5 * (1.0 - float((activation[off] > 0.5).any()))

    # --- sustain pedal.
    sustain = float(_tolerance(sustain_target - sustain_activation, 0.0, _KEY_T, _KEY_T * 10))

    # --- energy penalty (both hands).
    energy = -_ENERGY_COEF * float(hand_power[0] + hand_power[1])

    # --- fingering: fingertip-to-target distance for each active finger.
    dists = []
    for h in range(2):
        for f in range(5):
            if finger_active[h, f] > 0.5:
                dists.append(float(np.linalg.norm(finger_target_pos[h, f] - fingertip_pos[h, f])))
    if dists:
        fingering = float(np.mean(_tolerance(np.array(dists), 0.0, _FINGER_T, _FINGER_T * 10)))
    else:
        fingering = 0.0

    # --- forearm anti-collision (0.5 when clear).
    forearm = 0.5 * float(forearm_no_collision)

    return float(key_press + sustain + energy + fingering + forearm)
