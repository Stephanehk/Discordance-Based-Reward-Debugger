"""Trajectory visualization for the pandemic workspace.

Produces two stacked panels:
  1. Population proportions (alpha + delta combined) for infected, critical,
     dead, and recovered, each as its own coloured line with a legend.
  2. The regulation stage in force each day (step plot, 0..4).
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VIZ_EXT = ".png"


SERIES = [
    ("infected",  (5, 10),  "tab:orange"),   # alpha_infected + delta_infected
    ("critical",  (3,  8),  "tab:red"),      # alpha_critical + delta_critical
    ("dead",      (4,  9),  "black"),        # alpha_dead    + delta_dead
    ("recovered", (7, 12),  "tab:green"),    # alpha_recovered + delta_recovered
]


def _latest_row(obs):
    arr = np.asarray(obs)
    if arr.ndim == 3:
        return arr[-1, 0]
    return arr.reshape(-1)


def visualize_trajectory(trajectory, metadata, out_path, source_path=None, **_):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = np.stack([_latest_row(t[2]) for t in trajectory], axis=0)  # (T, 13)
    actions = np.array([int(t[1]) for t in trajectory], dtype=int)
    T = rows.shape[0]
    days = np.arange(1, T + 1)

    fig, (ax_pop, ax_stage) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for label, (i_a, i_d), color in SERIES:
        series = rows[:, i_a] + rows[:, i_d]
        ax_pop.plot(days, series, color=color, linewidth=1.8, label=label)
    ax_pop.set_ylabel("share of population")
    ax_pop.set_ylim(0.0, 0.4)
    ax_pop.grid(alpha=0.3)
    ax_pop.legend(loc="upper left", frameon=True, fontsize=9)

    ax_stage.step(days, actions, where="post", color="tab:blue", linewidth=1.6)
    ax_stage.set_ylim(-0.5, 4.5)
    ax_stage.set_yticks([0, 4])
    ax_stage.set_ylabel("regulation stage")
    ax_stage.set_xlabel("simulated day")
    ax_stage.grid(alpha=0.3)

    title = os.path.splitext(os.path.basename(out_path))[0]
    fig.suptitle(f"Trajectory {title}  ({T} days)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
