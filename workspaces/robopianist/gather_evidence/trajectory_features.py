"""Trajectory features for RoboPianist (Twinkle Twinkle, two Shadow hands).

Each transition is (obs, act, next_obs). Every physical quantity we need is in
the 339-dim physical-readout block appended to the END of the observation
vector (identical layout to what designed_reward_fn.py reads). We slice that
block off `next_obs` — the state the reward scores each step from.

Reward-term -> feature coverage
-------------------------------
The designed reward is the unweighted sum of five terms. For each term we
expose whether/how strongly it fired, and — where the term rewards a *proxy*
for the real musical goal — we expose the proxy and the true outcome as
SEPARATE features so a trajectory that inflates the proxy without actually
playing the song is visibly distinct.

  key_press term
    - key_f1                 TRUE  outcome: musical accuracy of the notes played
    - key_recall             TRUE  outcome: were the required keys actually struck
    - press_shaping_score    PROXY: reward's Gaussian depth credit (partial press
                                    earns partial credit even without a real strike)
    - false_press_count      TRUE  (continuous): how many wrong keys were pressed
    - any_false_press_rate   PROXY: the reward's *binary* wrong-key penalty — it
                                    can't tell 1 wrong key from 30
    - idle_rate              exposes the "press nothing" exploit (no false presses,
                                    low energy) that the term rewards by omission
  sustain term
    - sustain_accuracy       did the pedal match the song's sustain target
  energy term
    - mean_hand_power        the actuator-power proxy the term penalizes
  fingering term
    - mean_fingering_dist    PROXY: fingertip-near-key distance (rewarded even if
                                    the key is never actually pressed; the true
                                    press outcome is key_recall / key_f1 above)
  forearm term
    - forearm_collision_rate did the two forearms collide
"""

import numpy as np

_AUG_DIM = 339          # size of the appended physical-readout block
_KEY_T = 0.05           # reward's key "close enough" band
_FINGER_T = 0.01        # reward's fingertip "close enough" band (meters)


def _tolerance(x, lo, hi, margin, value_at_margin=0.1):
    """dm_control rewards.tolerance with a Gaussian sigmoid (matches the reward)."""
    x = np.asarray(x, dtype=np.float64)
    in_bounds = (x >= lo) & (x <= hi)
    d = np.where(x < lo, lo - x, x - hi) / margin
    scale = np.sqrt(-2.0 * np.log(value_at_margin))
    s = np.exp(-0.5 * (d * scale) ** 2)
    return np.where(in_bounds, 1.0, s)


def _unpack(next_obs):
    """Slice the physical-readout block from the tail of an observation vector."""
    a = np.asarray(next_obs, dtype=np.float64)[-_AUG_DIM:]
    return {
        "goal_keys": a[0:88],
        "sustain_target": a[88],
        "norm_state": a[89:177],
        "activation": a[177:265],
        "sustain_activation": a[265],
        "fingertip_pos": a[266:296].reshape(2, 5, 3),
        "finger_target_pos": a[296:326].reshape(2, 5, 3),
        "finger_active": a[326:336].reshape(2, 5),
        "hand_power": a[336:338],
        "forearm_no_collision": a[338],
    }


class Trajectory2Features:
    feature_names: list[tuple[str, str]] = [
        ("key_f1", "F1 of keys actually sounding vs. the song's required keys (musical accuracy)"),
        ("key_recall", "mean fraction of required keys actually struck each step (0-1)"),
        ("press_shaping_score", "reward's shaped depth credit for required keys (proxy; rewards partial presses)"),
        ("false_press_count", "mean number of wrong keys pressed per step"),
        ("any_false_press_rate", "fraction of steps with at least one wrong key (the reward's binary penalty)"),
        ("idle_rate", "fraction of steps with zero keys pressed (silence)"),
        ("sustain_accuracy", "fraction of steps the sustain pedal matched the song's target"),
        ("mean_hand_power", "mean total actuator power across both hands (energy proxy)"),
        ("mean_fingering_dist", "mean active-fingertip distance to its assigned key, m (fingering proxy)"),
        ("forearm_collision_rate", "fraction of steps the two forearms were colliding"),
    ]

    def get_traj_features(self, trajectory) -> list[float]:
        n = len(trajectory)
        if n == 0:
            return [0.0] * len(self.feature_names)

        tp = fp = fn = 0                 # aggregate key-press confusion counts
        recall_per_step = []
        shaping_per_step = []
        false_count_per_step = []
        any_false_per_step = []
        idle_per_step = []
        sustain_match_per_step = []
        power_per_step = []
        fingering_dists = []
        collision_per_step = []

        for _obs, _act, next_obs in trajectory:
            d = _unpack(next_obs)
            goal = d["goal_keys"] > 0.5
            act = d["activation"] > 0.5

            # key-press confusion (aggregated across the whole song for F1)
            tp += int(np.sum(goal & act))
            fp += int(np.sum(~goal & act))
            fn += int(np.sum(goal & ~act))

            on = np.flatnonzero(goal)
            if on.size > 0:
                recall_per_step.append(float(np.mean(act[on])))          # TRUE: struck?
                shaped = _tolerance(d["goal_keys"][on] - d["norm_state"][on],
                                    0.0, _KEY_T, _KEY_T * 10)
                shaping_per_step.append(float(np.mean(shaped)))          # PROXY: depth credit

            off = ~goal
            false_count_per_step.append(float(np.sum(act & off)))
            any_false_per_step.append(1.0 if np.any(act & off) else 0.0)
            idle_per_step.append(1.0 if not np.any(act) else 0.0)

            sustain_match_per_step.append(
                1.0 if (d["sustain_activation"] > 0.5) == (d["sustain_target"] > 0.5) else 0.0
            )
            power_per_step.append(float(d["hand_power"][0] + d["hand_power"][1]))

            fa = d["finger_active"] > 0.5
            if np.any(fa):
                diff = d["finger_target_pos"][fa] - d["fingertip_pos"][fa]
                fingering_dists.extend(np.linalg.norm(diff, axis=-1).tolist())

            collision_per_step.append(0.0 if d["forearm_no_collision"] > 0.5 else 1.0)

        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0

        return [
            float(f1),
            mean(recall_per_step),
            mean(shaping_per_step),
            mean(false_count_per_step),
            mean(any_false_per_step),
            mean(idle_per_step),
            mean(sustain_match_per_step),
            mean(power_per_step),
            mean(fingering_dists),
            mean(collision_per_step),
        ]

    def pretty_print_features(self, features) -> None:
        width = max(len(n) for n, _ in self.feature_names)
        for (short, desc), val in zip(self.feature_names, features):
            print(f"{short:<{width}}  {val:8.2f}   {desc}")
