"""
Aggregate Task #6 classifier sweep results into headline plots:

  1. accuracy_vs_T.png — for each (scenario, model), curve of test
     overall accuracy and group accuracy across T values, with horizontal
     reference lines for (a) random 1/24, (b) R1 marginal-Bayes ceiling
     0.75 per-class, 1.0 group, (c) R3b marginal-Bayes plateau ~0.10.
  2. per_class_bars.png — per-class test accuracy bar chart for the
     largest-T Transformer cell in each scenario, with R1 ambiguity
     groups shaded.
  3. confusion_grid.png — 3 × 2 grid of confusion matrices at the
     largest T.

Outputs land in `data/analysis/9_classifier/`.

Requires that the sweep has produced metrics.json + confusion.npy under
`data/analysis/9_classifier/{scenario}/{model}_T{T}/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import DATA_DIR


R1_AMBIG_GROUPS = [
    [6, 7],
    [14, 15, 17],
    [18, 19, 20],
    [22, 23],
]


def load_cell(root: Path, scenario: str, model: str, T: int):
    cell = root / scenario / f"{model}_T{T}"
    m_path = cell / "metrics.json"
    if not m_path.exists():
        return None
    with open(m_path) as f:
        m = json.load(f)
    conf_path = cell / "confusion.npy"
    conf = np.load(conf_path) if conf_path.exists() else None
    return m, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=DATA_DIR / "analysis" / "9_classifier")
    ap.add_argument("--models", nargs="+", default=["rnn", "gru", "transformer"])
    ap.add_argument("--scenarios", nargs="+", default=["r1", "r3b"])
    ap.add_argument("--T", nargs="+", type=int, default=[100, 300, 1000])
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    root = args.out_root
    root.mkdir(parents=True, exist_ok=True)

    # Collect all available cells
    cells = {}
    for sc in args.scenarios:
        for mo in args.models:
            for T in args.T:
                out = load_cell(root, sc, mo, T)
                if out is not None:
                    cells[(sc, mo, T)] = out

    if not cells:
        print("[error] no cell metrics found — run the sweep first")
        return

    # ----- accuracy_vs_T.png -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    model_colors = {"rnn": "#d62728", "gru": "#1f77b4", "transformer": "#2ca02c"}
    line_styles = {"r1": "-", "r3b": "--"}
    for ax_idx, sc in enumerate(args.scenarios):
        ax = axes[ax_idx]
        for mo in args.models:
            Ts = []
            overall = []
            group = []
            for T in sorted(args.T):
                if (sc, mo, T) in cells:
                    m = cells[(sc, mo, T)][0]
                    Ts.append(T)
                    overall.append(m["test_overall_acc"])
                    group.append(m["test_group_acc"])
            if Ts:
                ax.plot(Ts, overall, marker="o", color=model_colors[mo],
                        ls="-", label=f"{mo} overall")
                ax.plot(Ts, group, marker="s", color=model_colors[mo],
                        ls="--", alpha=0.6, label=f"{mo} group")
        # reference lines
        ax.axhline(1.0 / 24, color="gray", lw=0.5, ls=":", label="random 1/24")
        if sc == "r1":
            ax.axhline(0.75, color="black", lw=0.7, ls=":", label="R1 per-class ceiling 0.75")
            ax.axhline(1.0, color="black", lw=0.7, ls="-", alpha=0.3, label="group ceiling 1.0")
        if sc == "r3b":
            ax.axhline(0.10, color="black", lw=0.7, ls=":", label="R3b L=1 marginal-Bayes plateau ~0.10")
            ax.axhline(1.0, color="black", lw=0.7, ls="-", alpha=0.3, label="sequence-level ceiling (§15)")
        ax.set_xlabel("T (sequence length)")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Scenario: {sc.upper()}")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    plt.suptitle("Task #6: dominant-CNOT identification — model × scenario × T")
    plt.tight_layout()
    out = root / "accuracy_vs_T.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[saved] {out}")

    # ----- per_class bar charts at largest T (one row per scenario) -----
    T_max = max(args.T)
    fig, axes = plt.subplots(2, len(args.models), figsize=(5 * len(args.models), 7),
                              sharey=True)
    if len(args.models) == 1:
        axes = axes.reshape(2, 1)
    for sci, sc in enumerate(args.scenarios):
        for mi, mo in enumerate(args.models):
            ax = axes[sci, mi]
            if (sc, mo, T_max) not in cells:
                ax.axis("off")
                continue
            m, _ = cells[(sc, mo, T_max)]
            per_class = np.array(m["test_per_class_acc"])
            ks = np.arange(24)
            ax.bar(ks, per_class, color=model_colors[mo])
            # shade ambiguity groups
            for gi, g in enumerate(R1_AMBIG_GROUPS):
                ax.axvspan(min(g) - 0.5, max(g) + 0.5, alpha=0.15,
                           color=["#fde", "#dfe", "#def", "#fed"][gi % 4],
                           zorder=0)
            ax.set_xticks(ks)
            ax.set_xticklabels(ks, fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{sc.upper()} / {mo} @ T={T_max}\n"
                         f"overall={m['test_overall_acc']:.3f}  group={m['test_group_acc']:.3f}",
                         fontsize=9)
            ax.set_xlabel("CNOT index k")
            if mi == 0:
                ax.set_ylabel("per-class accuracy")
            ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = root / "per_class_bars.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[saved] {out}")

    # ----- confusion grid (3 cols of models × 2 rows of scenarios) -----
    fig, axes = plt.subplots(2, len(args.models), figsize=(5 * len(args.models), 9),
                              sharex=True, sharey=True)
    if len(args.models) == 1:
        axes = axes.reshape(2, 1)
    for sci, sc in enumerate(args.scenarios):
        for mi, mo in enumerate(args.models):
            ax = axes[sci, mi]
            if (sc, mo, T_max) not in cells:
                ax.axis("off")
                continue
            _, conf = cells[(sc, mo, T_max)]
            if conf is None:
                ax.axis("off")
                continue
            row_norm = conf.astype(np.float64) / conf.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(row_norm, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{sc.upper()} / {mo} @ T={T_max}", fontsize=9)
            if sci == 1:
                ax.set_xlabel("predicted k")
            if mi == 0:
                ax.set_ylabel("true k")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.suptitle(f"Test confusion at T={T_max} (row-normalized)")
    out = root / "confusion_grid.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[saved] {out}")

    # Print short summary
    print()
    print("===== SUMMARY (test set) =====")
    print(f"{'scenario':>9} {'model':>12} {'T':>5}  {'overall':>8} {'group':>8}")
    for (sc, mo, T), (m, _) in sorted(cells.items()):
        print(f"{sc:>9} {mo:>12} {T:>5}  {m['test_overall_acc']:>8.4f} {m['test_group_acc']:>8.4f}")


if __name__ == "__main__":
    main()
