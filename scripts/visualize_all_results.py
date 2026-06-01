"""
Unified visualization across all experiments (N=1..1000).

Outputs:
  plots/full_accuracy_vs_N.png       — all models + all N
  plots/confusion_matrix_*.png       — already in confusion/
  plots/ambiguity_zoom.png           — already in confusion/
  plots/large_N_comparison.png       — logreg vs deepsets at N=300..1000
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PLOT_DIR = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N5000_DIR   = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "set_classifier_n5000"
LARGEN_DIR  = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "set_classifier_largeN"

MODELS  = ["empirical_bayes", "logreg", "mlp", "deepsets"]
N_SMALL = [1, 3, 10, 30, 100, 300]
N_LARGE = [500, 1000]

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

RANDOM_BL = 1 / 24


def load_n5000(model, N):
    if model == "empirical_bayes":
        p = N5000_DIR / "empirical_bayes" / "metrics.json"
        if not p.exists(): return None
        d = json.loads(p.read_text())
        r = d["results"].get(str(N))
        return r["accuracy"] if r else None
    p = N5000_DIR / f"{model}_N{N}" / "metrics.json"
    return json.loads(p.read_text()).get("overall_acc") if p.exists() else None


def load_largeN(model, N):
    p = LARGEN_DIR / f"{model}_N{N}" / "metrics.json"
    return json.loads(p.read_text()).get("overall_acc") if p.exists() else None


# ── Plot 1: Full accuracy vs N (N=1..1000) ──────────────────────────────────

def plot_full_accuracy_vs_N():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in MODELS:
        ns, accs = [], []
        for N in N_SMALL:
            a = load_n5000(model, N)
            if a is not None:
                ns.append(N); accs.append(a)
        for N in N_LARGE:
            a = load_largeN(model, N)
            if a is not None:
                ns.append(N); accs.append(a)

        ls = "--" if model == "empirical_bayes" else "-"
        ax.plot(ns, accs,
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=2, markersize=7,
                linestyle=ls)

        # annotate last point
        if ns:
            ax.annotate(f"{accs[-1]:.3f}",
                        xy=(ns[-1], accs[-1]),
                        xytext=(6, 2), textcoords="offset points",
                        fontsize=8, color=MODEL_COLORS[model])

    ax.axhline(RANDOM_BL, color="gray", linestyle=":", linewidth=1.4,
               label=f"Random (1/24 ≈ {RANDOM_BL:.3f})")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("Bag size N (shots per bag)", fontsize=12)
    ax.set_ylabel("Test accuracy (24-class)", fontsize=12)
    ax.set_title("Bag-of-shots CNOT classifier: accuracy vs N  (N=1…1000)\n"
                 "noreset, d=3, p_high=0.1, p_bg=0.01", fontsize=11)
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    out = PLOT_DIR / "full_accuracy_vs_N.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Plot 2: logreg vs deepsets at large N ───────────────────────────────────

def plot_large_N_comparison():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in ["logreg", "deepsets"]:
        ns, accs = [], []
        for N in [100, 300] + N_LARGE:
            a = load_n5000(model, N) if N <= 300 else load_largeN(model, N)
            if a is not None:
                ns.append(N); accs.append(a)
        ax.plot(ns, accs,
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=2.5, markersize=8)
        for x, y in zip(ns, accs):
            ax.annotate(f"{y:.3f}", xy=(x, y),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, color=MODEL_COLORS[model])

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4, label="Perfect (1.0)")
    ax.axhline(RANDOM_BL, color="gray", linestyle=":", linewidth=1.2, label="Random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Bag size N", fontsize=12)
    ax.set_ylabel("Test accuracy (24-class)", fontsize=12)
    ax.set_title("LogReg vs Deep Sets at large N (N=100…1000)", fontsize=11)
    ax.set_xticks([100, 300, 500, 1000])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = PLOT_DIR / "large_N_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Plot 3: summary table heatmap ───────────────────────────────────────────

def plot_summary_table():
    N_all    = [1, 3, 10, 30, 100, 300, 500, 1000]
    models   = ["logreg", "mlp", "deepsets", "empirical_bayes"]
    data     = np.full((len(models), len(N_all)), np.nan)

    for i, m in enumerate(models):
        for j, N in enumerate(N_all):
            a = load_n5000(m, N) if N <= 300 else load_largeN(m, N)
            if a is not None:
                data[i, j] = a

    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Test accuracy")

    for i in range(len(models)):
        for j in range(len(N_all)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if v < 0.7 else "white")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="gray")

    ax.set_xticks(range(len(N_all)))
    ax.set_xticklabels([str(N) for N in N_all], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=10)
    ax.set_xlabel("Bag size N", fontsize=11)
    ax.set_title("Test accuracy summary — all models × all N", fontsize=11)

    out = PLOT_DIR / "summary_table.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    plot_full_accuracy_vs_N()
    plot_large_N_comparison()
    plot_summary_table()
    print("All plots saved to", PLOT_DIR)
