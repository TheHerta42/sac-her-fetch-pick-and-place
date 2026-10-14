"""Plot the SAC+HER results for the README.

Two figures, both written to results/:
  1. training_curves.png  — eval success rate / reward / episode length per run,
                            read from the TensorBoard event files in tensorboard/.
  2. dr_comparison.png    — success rate and completion length per model under
                            standard / trained-DR / severe-DR physics, read from
                            results/results.json produced by eval_dr.py.

Run from the repository root:
    python tools/plot_results.py

The tensorboard/*/ event files are large (~74 MB total) and are not tracked in
git; the committed PNGs in results/ are the output of this script. To regenerate
the training curves you need those event files, which ship with the full training
logs.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Labels mix Chinese model names with English axis text; pick a font that has both.
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TB = os.path.join(ROOT, "tensorboard")
OUT = os.path.join(ROOT, "results")

# label -> subdirectory of tensorboard/ holding that run's event file
TB_RUNS = {
    "v3 (6M, no DR)": "v3_no_dr",
    "dr_v1 (6.4M-9M, DR)": "dr_finetune",
}

COLORS = {"v3 (6M, no DR)": "#4C72B0", "dr_v1 (6.4M-9M, DR)": "#DD8452"}


def run_event_file(subdir):
    """Locate the event file for a run, tolerating the hostname/PID suffix."""
    matches = glob.glob(os.path.join(TB, subdir, "events.out.tfevents.*"))
    return matches[0] if matches else None


def read_scalars(path, tag):
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None, None
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])



def downsample(x, y, n=2000):
    """Thin long curves so the PNG stays a sane size."""
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]


def plot_training_curves():
    panels = [
        ("eval/success_rate", "Success rate", (0, 1.05)),
        ("eval/mean_reward", "Mean eval reward", None),
        ("eval/mean_ep_length", "Mean episode length", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (tag, title, ylim) in zip(axes, panels):
        plotted = False
        for label, subdir in TB_RUNS.items():
            path = run_event_file(subdir)
            if path is None:
                print(f"  [skip] no event file under tensorboard/{subdir} for {label}")
                continue
            steps, vals = read_scalars(path, tag)
            if steps is None:
                print(f"  [skip] no tag {tag} in {label}")
                continue
            ax.plot(steps, vals, label=label, color=COLORS.get(label), linewidth=1.8)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if tag == "eval/success_rate":
            ax.set_ylabel("success rate")
        if plotted:
            ax.legend(fontsize=9)
    fig.suptitle("SAC + HER on FetchPickAndPlace — evaluation during training",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, "training_curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_dr_comparison():
    jpath = os.path.join(OUT, "results.json")
    if not os.path.exists(jpath):
        print(f"[skip] {jpath} not found — run eval_dr.py first")
        return
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"]
    physics = list(res.keys())
    models = list(res[physics[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(models))
    width = 0.8 / len(physics)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, phys in enumerate(physics):
        succ = [res[phys][m]["success_rate"] for m in models]
        length = [res[phys][m]["mean_length"] for m in models]
        off = (i - (len(physics) - 1) / 2) * width
        axes[0].bar(x + off, succ, width, label=phys, color=palette[i % len(palette)])
        axes[1].bar(x + off, length, width, label=phys, color=palette[i % len(palette)])
        for xi, v in zip(x + off, succ):
            axes[0].text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)

    axes[0].set_title("Success rate by physics condition")
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 108)
    axes[1].set_title("Mean completion length (lower is better)")
    axes[1].set_ylabel("steps")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.suptitle(
        f"Domain-randomization robustness ({data['episodes']} episodes per bar, "
        f"target_range={data['target_range']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(OUT, "dr_comparison.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_training_curves()
    plot_dr_comparison()
