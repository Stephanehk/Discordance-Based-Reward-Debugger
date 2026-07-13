"""Environment configuration for Google Research's `robopianist`
(`RoboPianist-debug-TwinkleTwinkleRousseau-v0`, the default dense-reward task).

Two Shadow dexterous hands must play a MIDI song ("Twinkle Twinkle Little Star")
on a full 88-key piano, pressing the right keys at the right time while following
human fingering labels and a sustain pedal.

robopianist is a dm_control (dm_env) `composer.Environment`, NOT a gymnasium env.
`make_env` wraps it to expose the gym-style `reset/step/render/action_space`
interface the rest of the pipeline expects, flattening the dict observation into
a single vector.

Observation
-----------
The wrapper flattens the default dm_control observation (329 dims) and then
appends a block of physical quantities the designed reward reads but which the
default observation omits (per-key activation, normalized key depth, fingertip
and target-key positions, per-hand actuator power, the forearm-collision flag,
and the frozen per-step goal). Everything `(obs, act, next_obs)` analysis needs
is therefore present in the vector. See OBS_DESCRIPTION for the exact layout.
"""

from __future__ import annotations

import os

# MuJoCo offscreen rendering on headless nodes.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
from robopianist import suite
from robopianist.suite.tasks import piano_with_shadow_hands as _pwsh

CAPTURE_FRAMES = True

_ENV_NAME = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
_RENDER_CAMERA = "piano/topdown"
_RENDER_HW = (480, 640)

# ---------------------------------------------------------------------------
# Augmented-observation layout (appended after the flattened base observation).
# designed_reward_fn.py slices this block from the END of the vector, so these
# sizes are the contract between the two files — keep them in sync.
# ---------------------------------------------------------------------------
_N_KEYS = 88
_AUG_SIZES = [
    ("goal_current", _N_KEYS + 1),   # 88 key targets (0/1) + 1 sustain target
    ("norm_state", _N_KEYS),         # piano.state / qpos_range[:, 1]
    ("activation", _N_KEYS),         # piano.activation as float (0/1)
    ("sustain_activation", 1),       # piano.sustain_activation[0] as float
    ("fingertip_pos", 30),           # rh f0..f4, lh f0..f4 (xyz each)
    ("finger_target_pos", 30),       # per finger: adjusted target key xyz (0 if inactive)
    ("finger_active", 10),           # 1.0 if that finger targets a key this step
    ("hand_power", 2),               # [sum rh actuator power, sum lh actuator power]
    ("forearm_no_collision", 1),     # 1.0 if the forearms are NOT colliding
]
AUG_DIM = sum(n for _, n in _AUG_SIZES)   # 339

OBS_DESCRIPTION = """\
Box(float64) = a flattened robopianist observation followed by a physical-readout
block. The leading base block (329 dims, dm_control default) is:
  goal (178)                 : two rows of (88 key targets + 1 sustain target);
                               a 1 marks a key/pedal that should be active now
                               and one step ahead
  fingering (10)             : which of the 10 fingers are used this step (0/1)
  piano/state (88)           : raw key joint positions (how far each key is down)
  piano/sustain_state (1)    : sustain-pedal position
  rh_shadow_hand/joints_pos (26), lh_shadow_hand/joints_pos (26): hand joint angles
The trailing block (339 dims) exposes the physical quantities the designed
reward reads (all in SI / normalized units, world-frame positions in meters):
  goal_current (89)          : the reward's frozen per-step target [88 keys + sustain]
  norm_state (88)            : key depth normalized to [0, 1] (state / max travel)
  activation (88)            : per-key "pressed" flag (0/1)
  sustain_activation (1)     : sustain-pedal engaged flag (0/1)
  fingertip_pos (30)         : 10 fingertip positions (rh thumb..pinky, lh thumb..pinky)
  finger_target_pos (30)     : per finger, the target key's contact point (0 if idle)
  finger_active (10)         : per finger, 1 if it is assigned a key this step
  hand_power (2)             : summed actuator mechanical power [right, left]
  forearm_no_collision (1)   : 1 if the two forearms are not in contact
"""

ACTION_DESCRIPTION = """\
Box(shape=(45,), dtype=float64). Target joint positions for the two Shadow hands
plus a sustain-pedal command:
  [0:22]  right-hand actuators (20 finger/wrist joints + 2 forearm x/y slides)
  [22:44] left-hand actuators (same layout)
  [44]    sustain pedal, in [0, 1]
Each hand actuator's bounds are its native joint range (roughly [-1.05, 3.14] rad
across the different joints).
"""

# The debug song is 451 control steps long; the episode ends when it finishes.
MAX_EPISODE_STEPS = 451


def _adjusted_key_pos(physics, piano, key):
    """Target contact point for a key geom (matches the reward's offsets)."""
    key_geom = piano.keys[key].geom[0]
    pos = physics.bind(key_geom).xpos.copy()
    size = physics.bind(key_geom).size
    pos[-1] += 0.5 * size[2]
    pos[0] += 0.35 * size[0]
    return pos


class _RoboPianistGymEnv:
    """Minimal gym-style adapter over the robopianist dm_env task."""

    def __init__(self, render_mode: str | None = None, seed: int | None = None):
        self._dm = suite.load(_ENV_NAME, seed=seed)
        self._task = self._dm.task
        self.render_mode = render_mode

        # Fixed key order for deterministic flattening of the dict observation.
        self._obs_keys = list(self._dm.observation_spec().keys())

        aspec = self._dm.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.asarray(aspec.minimum, dtype=np.float64),
            high=np.asarray(aspec.maximum, dtype=np.float64),
            dtype=np.float64,
        )
        self._act_dtype = aspec.dtype

        base_dim = sum(int(np.prod(self._dm.observation_spec()[k].shape))
                       for k in self._obs_keys)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + AUG_DIM,), dtype=np.float64
        )

    # -- observation assembly ------------------------------------------------
    def _flatten_base(self, obs) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float64).reshape(-1) for k in self._obs_keys]
        )

    def _augment(self) -> np.ndarray:
        task, physics = self._task, self._dm.physics
        piano = task.piano

        # The reward reads the goal / fingering frozen in `after_step`; fall back
        # to the current window on the very first observation (no reward there).
        goal_current = np.asarray(
            getattr(task, "_goal_current", task._goal_state[0]), dtype=np.float64
        )
        rh_keys = getattr(task, "_rh_keys_current", getattr(task, "_rh_keys", []))
        lh_keys = getattr(task, "_lh_keys_current", getattr(task, "_lh_keys", []))

        norm_state = np.asarray(piano.state, dtype=np.float64) / piano._qpos_range[:, 1]
        activation = np.asarray(piano.activation, dtype=np.float64)
        sustain_activation = np.array([float(piano.sustain_activation[0])])

        fingertip_pos = np.zeros((2, 5, 3))
        for h, hand in enumerate([task.right_hand, task.left_hand]):
            for f in range(5):
                fingertip_pos[h, f] = physics.bind(hand.fingertip_sites[f]).xpos

        finger_target = np.zeros((2, 5, 3))
        finger_active = np.zeros((2, 5))
        for h, keys in enumerate([rh_keys, lh_keys]):
            for key, finger in keys:
                finger_target[h, finger] = _adjusted_key_pos(physics, piano, key)
                finger_active[h, finger] = 1.0

        hand_power = np.array([
            float(np.sum(task.right_hand.observables.actuators_power(physics))),
            float(np.sum(task.left_hand.observables.actuators_power(physics))),
        ])

        collided = _pwsh.collision_utils.has_collision(
            physics,
            [g.full_identifier for g in task.right_hand.root_body.geom],
            [g.full_identifier for g in task.left_hand.root_body.geom],
        )
        forearm_no_collision = np.array([0.0 if collided else 1.0])

        return np.concatenate([
            goal_current,
            norm_state,
            activation,
            sustain_activation,
            fingertip_pos.reshape(-1),
            finger_target.reshape(-1),
            finger_active.reshape(-1),
            hand_power,
            forearm_no_collision,
        ])

    def _obs(self, dm_obs) -> np.ndarray:
        return np.concatenate([self._flatten_base(dm_obs), self._augment()])

    # -- gym API -------------------------------------------------------------
    def reset(self, **kwargs):
        ts = self._dm.reset()
        return self._obs(ts.observation), {}

    def step(self, action):
        action = np.asarray(action, dtype=self._act_dtype).reshape(-1)
        ts = self._dm.step(action)
        obs = self._obs(ts.observation)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = bool(ts.last())
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        frame = self._dm.physics.render(
            height=_RENDER_HW[0], width=_RENDER_HW[1], camera_id=_RENDER_CAMERA
        )
        return np.asarray(frame, dtype=np.uint8)

    def close(self):
        pass


def make_env(render_mode: str | None = None, seed: int | None = None):
    return _RoboPianistGymEnv(render_mode=render_mode, seed=seed)
