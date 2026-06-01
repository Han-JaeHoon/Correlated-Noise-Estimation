"""
Visualize bag-of-shots classifier sweep results.

Plots:
  1. accuracy_vs_N.png     — main curve: all 4 models + random baseline
  2. per_class_heatmap.png — per-class accuracy heatmap at best N per model
  3. ambiguity_groups.png  — ambiguity-group accuracy bar chart

Usage:
    python scripts/analyze_bag_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "set_classifier_n5000"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["empirical_bayes", "logreg", "mlp", "deepsets"]
N_LIST  = [1, 3, 10, 30, 100, 300]
RANDOM_BASELINE = 1 / 24

MODEL_LABELS = {
    "empirical_bayes": "Empirical Bayes",
    "logreg":          "LogReg (linear)",
    "mlp":             "MLP (nonlinear)",
    "deepsets":        "Deep Sets",
}
MODEL_COLORS = {
    "empirical_bayes": "#4e79a7",
    "logreg":          "#f28e2b",
    "mlp":             "#e15759",
    "deepsets":        "#76b7b2",
}
MODEL_MARKERS = {
    "empirical_bayes": "o",
    "logreg":          "s",
    "mlp":             "^",
    "deepsets":        "D",
}

AMBIGUITY_GROUPS = {
    "G1: {6,7}":       [6, 7],
    "G2: {14,15,17}":  [14, 15, 17],
    "G3: {18,19,20}":  [18, 19, 20],
    "G4: {22,23}":     [22, 23],
}

# ── helpers ─────────────────────────────────────────────────────────────────

def load_acc(model, N):
    if model == "empirical_bayes":
        p = OUT_DIR / "empirical_bayes" / "metrics.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        r = d["results"].get(str(N))
        return r["accuracy"] if r else None
    p = OUT_DIR / f"{model}_N{N}" / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("overall_acc")


def load_per_class(model, N):
    if model == "empirical_bayes":
        return None
    p = OUT_DIR / f"{model}_N{N}" / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pc = d.get("per_class_acc", {})
    return {int(k): v for k, v in pc.items()}


# ── Plot 1: accuracy vs N ────────────────────────────────────────────────────

def plot_accuracy_vs_N():
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        accs = []
        ns   = []
        for N in N_LIST:
            a = load_acc(model, N)
            if a is not None:
                ns.append(N)
                accs.append(a)
        ax.plot(
            ns, accs,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=2, markersize=7,
        )

    ax.axhline(RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.5,
               label=f"Random (1/24 ≈ {RANDOM_BASELINE:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Bag size N (shots per bag)", fontsize=12)
    ax.set_ylabel("Test accuracy (24-class)", fontsize=12)
    ax.set_title("Bag-of-shots classifier: accuracy vs bag size N\n"
                 f"(noreset, d=3, p_high=0.1, p_bg=0.01)", fontsize=11)
    ax.set_xticks(N_LIST)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    out = PLOT_DIR / "accuracy_vs_N.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Plot 2: per-class accuracy heatmap ──────────────────────────────────────

def plot_per_class_heatmap():
    # best N per model (by overall acc)
    best = {}
    for model in ["logreg", "mlp", "deepsets"]:
        best_acc, best_N = -1, None
        for N in N_LIST:
            a = load_acc(model, N)
            if a is not None and a > best_acc:
                best_acc, best_N = a, N
        best[model] = best_N

    models_with_best = [m for m in ["logreg", "mlp", "deepsets"] if best[m] is not None]
    n_models = len(models_with_best)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    ambig_members = {c for g in AMBIGUITY_GROUPS.values() for c in g}

    for ax, model in zip(axes, models_with_best):
        N = best[model]
        pc = load_per_class(model, N)
        if pc is None:
            continue

        acc_vec = np.array([pc.get(c, 0.0) for c in range(24)])
        colors  = ["#e15759" if c in ambig_members else "#4e79a7" for c in range(24)]

        bars = ax.barh(range(24), acc_vec, color=colors, height=0.7)
        ax.axvline(RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.2)
        ax.set_xlim(0, 1.0)
        ax.set_yticks(range(24))
        ax.set_yticklabels([f"CNOT {c:02d}" for c in range(24)], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy", fontsize=10)
        ax.set_title(f"{MODEL_LABELS[model]}\n(N={N}, acc={load_acc(model, N):.3f})",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    # legend for colour coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e15759", label="R1 ambiguity group"),
        Patch(facecolor="#4e79a7", label="Other CNOTs"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle("Per-class test accuracy at best N (red = R1 ambiguity group)", fontsize=11)
    out = PLOT_DIR / "per_class_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Plot 3: ambiguity group breakdown ───────────────────────────────────────

def plot_ambiguity_groups():
    # best N per model
    best = {}
    for model in ["logreg", "mlp", "deepsets"]:
        best_acc, best_N = -1, None
        for N in N_LIST:
            a = load_acc(model, N)
            if a is not None and a > best_acc:
                best_acc, best_N = a, N
        best[model] = best_N

    group_names = list(AMBIGUITY_GROUPS.keys())
    models_plot = ["logreg", "mlp", "deepsets"]
    x = np.arange(len(group_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, model in enumerate(models_plot):
        N = best.get(model)
        pc = load_per_class(model, N) if N else None
        if pc is None:
            continue
        group_accs = []
        for members in AMBIGUITY_GROUPS.values():
            group_accs.append(np.mean([pc.get(c, 0.0) for c in members]))

        ax.bar(x + i * width, group_accs,
               width=width,
               color=MODEL_COLORS[model],
               label=f"{MODEL_LABELS[model]} (N={N})",
               alpha=0.85)

    ax.axhline(RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.4,
               label=f"Random baseline")
    ax.set_xticks(x + width)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Mean accuracy within group", fontsize=11)
    ax.set_title("Per-ambiguity-group accuracy (R1 structurally hard pairs)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out = PLOT_DIR / "ambiguity_groups.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p1 = plot_accuracy_vs_N()
    p2 = plot_per_class_heatmap()
    p3 = plot_ambiguity_groups()
    print("\nAll plots saved to", PLOT_DIR)
