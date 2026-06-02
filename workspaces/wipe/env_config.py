"""Environment config for the robosuite Wipe task (Panda arm).

The agent controls a Panda arm holding a wiping pad and must clean a
set of small "dirt" markers off a table. Set-up mirrors the SAC
training script from Mohan et al.'s combined-HPO/reward-shaping repo
(`source/dehb_robosuite_wipe/sac_hydra.py`):

  - Robot:          Panda
  - Controller:     OSC_POSE (operational-space 6-DoF pose delta)
  - Horizon:        500 control steps at 20 Hz
  - reward_scale:   1.0
  - reward_shaping: True
  - ignore_done:    True (episode ends only on the gymnasium TimeLimit)
  - hard_reset:     False (so subprocess workers can reset quickly)

The robosuite Wipe env's reward formula reads several quantities that
are not part of its default observation dict (force/torque sensors,
new-marker-wiped delta this step, collision flags, end-effector
acceleration, sim contact count, mean distance to remaining dirt).
``_WipeRewardEnv`` extracts those quantities after every ``step`` and
appends them to the flattened observation as the last ``N_RWD_COMPONENTS``
float channels, so ``designed_reward_fn(obs, act, next_obs)`` can
reconstruct the env's default shaped reward without poking at sim
internals. See ``designed_reward_fn.py`` for the formula and weights.
"""

from __future__ import annotations

import os

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import robosuite as suite
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG

# robosuite >=1.5 expects a *composite* controller config (one entry per
# robot body part). The default "BASIC" composite already pairs the arm
# part with the OSC_POSE controller used by the Wipe reference in
# combined_hpo_and_reward_shaping. On older robosuite (<1.5) fall back
# to the flat ``load_controller_config(default_controller="OSC_POSE")``.
try:
    from robosuite.controllers import load_composite_controller_config

    def _wipe_controller(robot: str):
        return load_composite_controller_config(controller="BASIC", robot=robot)
except ImportError:  # robosuite <1.5
    from robosuite.controllers import load_controller_config

    def _wipe_controller(robot: str):
        return load_controller_config(default_controller="OSC_POSE")


# Opt into per-step frame capture so /repair-preferences can show the actual
# rolled-out video (rather than an after-the-fact replay).
CAPTURE_FRAMES = True

HORIZON = 500
CONTROL_FREQ = 20

# Number of float channels appended to the flattened observation that
# carry the reward components. Keep in sync with the slots populated by
# ``_WipeRewardEnv._reward_components`` and consumed by designed_reward_fn.
N_RWD_COMPONENTS = 10


class _WipeRewardEnv(gym.Env):
    """Gymnasium-compatible wrapper around a robosuite Wipe env.

    Flattens the dict observation (robot proprio + condensed object
    state) into a 1-D float array and appends ``N_RWD_COMPONENTS`` float
    channels carrying the per-step quantities needed by
    ``designed_reward_fn`` to reconstruct the env's default reward.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode
        use_camera_obs = render_mode == "rgb_array"

        self._env = suite.make(
            env_name="Wipe",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=use_camera_obs,
            use_camera_obs=use_camera_obs,
            horizon=HORIZON,
            control_freq=CONTROL_FREQ,
            controller_configs=_wipe_controller("Panda"),
            reward_scale=1.0,
            ignore_done=True,
            reward_shaping=True,
            camera_names="frontview",
            hard_reset=False,
        )

        # Decide which observation keys to flatten — same default as
        # robosuite.wrappers.GymWrapper but excluding image observations
        # (they go through render(), not the policy obs).
        self._keys: list[str] = []
        if self._env.use_object_obs:
            self._keys.append("object-state")
        for idx in range(len(self._env.robots)):
            self._keys.append(f"robot{idx}_proprio-state")

        # Initialise spaces by inspecting one reset.
        obs_dict = self._env.reset()
        self._last_image = obs_dict.pop("frontview_image", None) if use_camera_obs else None
        flat = self._flatten_obs(obs_dict)
        self._proprio_dim = int(flat.size)
        self._obs_dim = self._proprio_dim + N_RWD_COMPONENTS
        high = np.full(self._obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        low, high = self._env.action_spec
        self.action_space = spaces.Box(low=np.asarray(low, dtype=np.float32),
                                       high=np.asarray(high, dtype=np.float32),
                                       dtype=np.float32)

        self._prev_wiped_count = 0

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        parts = [np.asarray(obs_dict[k]).flatten() for k in self._keys if k in obs_dict]
        return np.concatenate(parts).astype(np.float32)

    def _reward_components(self) -> np.ndarray:
        """Return the per-step reward-component vector for the current sim state.

        Mirrors the quantities the robosuite Wipe ``reward()`` formula
        consumes. Indices here are the canonical layout the reward fn reads.
        """
        env = self._env
        robot = env.robots[0]

        total_wiped = float(len(env.wiped_markers))
        new_wiped = total_wiped - self._prev_wiped_count
        self._prev_wiped_count = total_wiped

        try:
            total_force_ee = float(max(
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ))
        except Exception:
            total_force_ee = 0.0

        # Mean distance from wiper to remaining dirt (0 if all wiped).
        if total_wiped < env.num_markers:
            try:
                _, _, mean_pos = env._get_wipe_information()
                mean_distance = float(np.linalg.norm(mean_pos))
            except Exception:
                mean_distance = 0.0
        else:
            mean_distance = 0.0

        has_gripper_contact = float(bool(getattr(env, "_has_gripper_contact", False)))
        ncon = float(env.sim.data.ncon)

        try:
            mean_ee_acc = float(max(
                np.mean(np.abs(robot.recent_ee_acc[arm].current))
                for arm in robot.arms
            ))
        except Exception:
            mean_ee_acc = 0.0

        # Either the arm collided with the table or it hit a joint limit;
        # both clamp the env's reward to arm_limit_collision_penalty.
        try:
            arm_collision = bool(env.check_contact(robot.robot_model))
        except Exception:
            arm_collision = False
        try:
            q_limit = bool(robot.check_q_limits())
        except Exception:
            q_limit = False
        collision_or_qlim = float(arm_collision or q_limit)

        num_markers = float(env.num_markers)
        horizon = float(HORIZON)

        return np.asarray([
            total_wiped,           # 0  cumulative markers wiped so far
            new_wiped,             # 1  markers wiped on this single step
            mean_distance,         # 2  mean distance from wiper to remaining dirt
            has_gripper_contact,   # 3  1.0 if wiper geom is in table contact
            ncon,                  # 4  number of active sim contacts
            total_force_ee,        # 5  max end-effector force magnitude (N)
            mean_ee_acc,           # 6  mean |end-effector acceleration|
            collision_or_qlim,     # 7  1.0 if arm collided / hit joint limit
            num_markers,           # 8  constant: total dirt markers spawned
            horizon,               # 9  constant: episode horizon (steps)
        ], dtype=np.float32)

    def _build_obs(self, obs_dict: dict) -> np.ndarray:
        flat = self._flatten_obs(obs_dict)
        return np.concatenate([flat, self._reward_components()]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except Exception:
                pass
        obs_dict = self._env.reset()
        if self.render_mode == "rgb_array":
            self._last_image = obs_dict.pop("frontview_image", None)
        self._prev_wiped_count = 0
        return self._build_obs(obs_dict), {}

    def step(self, action):
        obs_dict, reward, done, info = self._env.step(np.asarray(action, dtype=np.float32))
        if self.render_mode == "rgb_array":
            self._last_image = obs_dict.pop("frontview_image", None)
        obs = self._build_obs(obs_dict)
        # Pass the env's own reward through info so it stays accessible
        # to anyone who wants to compare it against the designed reward.
        info = dict(info or {})
        info["env_reward"] = float(reward)
        return obs, float(reward), bool(done), False, info

    def render(self):
        if self.render_mode == "rgb_array" and self._last_image is not None:
            return np.flipud(np.asarray(self._last_image))
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


def make_env(render_mode: str | None = None):
    """Build a fresh Wipe env wrapped for the repair pipeline.

    ``render_mode="rgb_array"`` enables offscreen camera rendering so
    rollouts can capture per-step frames (consumed by viz.py through the
    ``CAPTURE_FRAMES`` sidecar pipeline).
    """
    env = _WipeRewardEnv(render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=HORIZON)


OBS_DESCRIPTION = """\
Observation: 1-D float32 vector of length (proprio_dim + 10).

The leading ``proprio_dim`` entries are robosuite's default flattened
Wipe observation, concatenated in this order:

  1. object-state — condensed dirt-field summary (proportion of markers
     already wiped, mean dirt radius, mean dirt centroid position
     relative to the gripper) plus per-arm gripper-to-dirt ego-centric
     vectors. Indices and exact layout follow the robosuite Wipe
     ``setup_observables`` defaults (``use_condensed_obj_obs=True``).
  2. robot0_proprio-state — Panda proprioception: joint positions (cos
     and sin), joint velocities, end-effector position, end-effector
     orientation (quat), gripper-state placeholder, and a binary
     contact flag from the wiper.

The trailing 10 float channels carry the per-step reward components,
extracted directly from sim internals at ``env.step`` time so
``designed_reward_fn(obs, act, next_obs)`` can reconstruct the env's
default shaped reward without poking at MuJoCo:

  next_obs[-10] = total_wiped         - cumulative # markers wiped so far
  next_obs[ -9] = new_wiped           - # markers wiped on this single step
  next_obs[ -8] = mean_distance       - mean dist from wiper to remaining dirt
                                        (0 once all markers are wiped)
  next_obs[ -7] = has_gripper_contact - 1.0 iff wiper geom is in table contact
  next_obs[ -6] = ncon                - # of active MuJoCo contacts
  next_obs[ -5] = total_force_ee      - max end-effector force magnitude (N)
  next_obs[ -4] = mean_ee_acc         - mean |end-effector acceleration|
  next_obs[ -3] = collision_or_qlim   - 1.0 if arm collided with the table
                                        OR hit a joint limit this step
  next_obs[ -2] = num_markers         - constant: total dirt markers spawned
  next_obs[ -1] = horizon             - constant: episode horizon (steps)
"""

ACTION_DESCRIPTION = """\
Action: Box(low=-1.0, high=1.0, shape=(6,), dtype=float32).

OSC_POSE end-effector control (operational-space pose delta) for the
Panda's right arm. The WipingGripper has no gripper-open/close DoF, so
there is no separate gripper slot — the action is just the 6-D pose
delta consumed by the OSC controller:

  [0..2] - desired end-effector position delta (dx, dy, dz). The OSC
           controller scales this from [-1, 1] into physical position
           limits (default ±0.05 m per axis per control tick).
  [3..5] - desired end-effector orientation delta as an axis-angle
           rotation (rx, ry, rz). The OSC controller scales this from
           [-1, 1] into physical rotation limits (default ±0.5 rad per
           axis per control tick).
"""

MAX_EPISODE_STEPS = HORIZON
