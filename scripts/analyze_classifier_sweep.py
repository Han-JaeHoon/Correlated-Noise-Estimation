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
    ap.add_argument("--scenarios", nargs="+", default=["r1", "r2", "r3b"])
    ap.add_argument("--T", nargs="+", type=int, default=[128, 512])
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

    model_colors = {"rnn": "#d62728", "gru": "#1f77b4", "transformer": "#2ca02c"}
    T_colors = {128: "#1f77b4", 512: "#ff7f0e", 100: "#1f77b4", 300: "#ff7f0e", 1000: "#2ca02c"}
    sc_colors = {"r1": "#1f77b4", "r2": "#ff7f0e", "r3b": "#2ca02c"}

    # ------------------------------------------------------------------ #
    # 1. TRAINING CURVES — loss and accuracy per epoch                     #
    # Layout: scenarios (rows) × models (cols), two sub-rows each         #
    # (top = loss, bottom = accuracy), T values as line colors             #
    # ------------------------------------------------------------------ #
    T_styles = {t: s for t, s in zip(sorted(set(args.T)), ["-", "--", ":", "-."])}

    # (a) loss curves
    n_sc = len(args.scenarios)
    n_mo = len(args.models)
    fig, axes = plt.subplots(
        n_sc, n_mo, figsize=(5.5 * n_mo, 3.5 * n_sc),
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    for si, sc in enumerate(args.scenarios):
        for mi, mo in enumerate(args.models):
            ax = axes[si, mi]
            any_plotted = False
            for T in sorted(args.T):
                res = load_cell(root, sc, mo, T)
                if res is None:
                    continue
                m, _ = res
                ep_data = m.get("epochs", [])
                if not ep_data:
                    continue
                eps = [e["epoch"] for e in ep_data]
                tr_loss = [e["train_loss"] for e in ep_data]
                vl_loss = [e["val_loss"] for e in ep_data]
                col = T_colors.get(T, "black")
                ax.plot(eps, tr_loss, color=col, ls="-", lw=1.5,
                        label=f"T={T} train")
                ax.plot(eps, vl_loss, color=col, ls="--", lw=1.0,
                        alpha=0.7, label=f"T={T} val")
                any_plotted = True
            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.set_title(f"{sc.upper()} / {mo}", fontsize=10)
            ax.set_xlabel("epoch")
            ax.set_ylabel("cross-entropy loss")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper right", ncol=1)
    fig.suptitle("Training & validation loss per epoch", fontsize=12)
    out = root / "training_curves_loss.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # (b) accuracy curves
    fig, axes = plt.subplots(
        n_sc, n_mo, figsize=(5.5 * n_mo, 3.5 * n_sc),
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    for si, sc in enumerate(args.scenarios):
        for mi, mo in enumerate(args.models):
            ax = axes[si, mi]
            any_plotted = False
            for T in sorted(args.T):
                res = load_cell(root, sc, mo, T)
                if res is None:
                    continue
                m, _ = res
                ep_data = m.get("epochs", [])
                if not ep_data:
                    continue
                eps = [e["epoch"] for e in ep_data]
                tr_acc = [e["train_acc"] for e in ep_data]
                vl_acc = [e["val_acc"] for e in ep_data]
                col = T_colors.get(T, "black")
                ax.plot(eps, tr_acc, color=col, ls="-", lw=1.5,
                        label=f"T={T} train")
                ax.plot(eps, vl_acc, color=col, ls="--", lw=1.0,
                        alpha=0.7, label=f"T={T} val")
                any_plotted = True
            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.axhline(1.0 / 24, color="gray", lw=0.7, ls=":", label="random")
            if sc == "r1":
                ax.axhline(0.75, color="black", lw=0.7, ls=":",
                           label="R1 ceiling 0.75")
            if sc in ("r2", "r3b"):
                ax.axhline(0.10, color="black", lw=0.7, ls=":",
                           label="marginal-Bayes ~0.10")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{sc.upper()} / {mo}", fontsize=10)
            ax.set_xlabel("epoch")
            ax.set_ylabel("accuracy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower right", ncol=1)
    fig.suptitle("Training & validation accuracy per epoch", fontsize=12)
    out = root / "training_curves_acc.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # ------------------------------------------------------------------ #
    # 2. accuracy_vs_T.png  (test accuracy × T, one panel per scenario)   #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, n_sc, figsize=(5.5 * n_sc, 5.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, sc in zip(axes, args.scenarios):
        for mo in args.models:
            Ts, overall, group = [], [], []
            for T in sorted(args.T):
                if (sc, mo, T) in cells:
                    m = cells[(sc, mo, T)][0]
                    Ts.append(T)
                    overall.append(m["test_overall_acc"])
                    group.append(m["test_group_acc"])
            if Ts:
                ax.plot(Ts, overall, marker="o", color=model_colors[mo],
                        ls="-", lw=1.5, label=f"{mo} overall")
                ax.plot(Ts, group, marker="s", color=model_colors[mo],
                        ls="--", lw=1.0, alpha=0.65, label=f"{mo} group")
        ax.axhline(1.0 / 24, color="gray", lw=0.7, ls=":", label="random 1/24")
        if sc == "r1":
            ax.axhline(0.75, color="black", lw=0.8, ls=":",
                       label="R1 per-class ceiling 0.75")
        if sc in ("r2", "r3b"):
            ax.axhline(0.10, color="black", lw=0.8, ls=":",
                       label="marginal-Bayes plateau ~0.10")
        ax.set_xlabel("T (sequence length)")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Scenario: {sc.upper()}")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    plt.suptitle("Task #6: dominant-CNOT identification — test accuracy vs T")
    plt.tight_layout()
    out = root / "accuracy_vs_T.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[saved] {out}")

    # ------------------------------------------------------------------ #
    # 3. per_class bar charts at largest T                                 #
    # ------------------------------------------------------------------ #
    T_max = max(args.T)
    fig, axes = plt.subplots(n_sc, n_mo,
                              figsize=(5 * n_mo, 3.5 * n_sc),
                              sharey=True,
                              gridspec_kw={"hspace": 0.55, "wspace": 0.15})
    if n_sc == 1:
        axes = axes.reshape(1, n_mo)
    for sci, sc in enumerate(args.scenarios):
        for mi, mo in enumerate(args.models):
            ax = axes[sci, mi]
            if (sc, mo, T_max) not in cells:
                ax.axis("off")
                continue
            m, _ = cells[(sc, mo, T_max)]
            per_class = np.array(m["test_per_class_acc"])
            ks = np.arange(24)
            ax.bar(ks, per_class, color=model_colors[mo], width=0.7)
            for gi, g in enumerate(R1_AMBIG_GROUPS):
                ax.axvspan(min(g) - 0.5, max(g) + 0.5, alpha=0.15,
                           color=["#fde", "#dfe", "#def", "#fed"][gi % 4],
                           zorder=0)
            ax.axhline(1.0 / 24, color="gray", lw=0.7, ls=":")
            ax.set_xticks(ks)
            ax.set_xticklabels(ks, fontsize=6)
            ax.set_ylim(0, 1.05)
            ax.set_title(
                f"{sc.upper()} / {mo} @ T={T_max}\n"
                f"overall={m['test_overall_acc']:.3f}  "
                f"group={m['test_group_acc']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("CNOT index k", fontsize=8)
            if mi == 0:
                ax.set_ylabel("per-class accuracy")
            ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"Per-class test accuracy @ T={T_max}", fontsize=12)
    out = root / "per_class_bars.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # ------------------------------------------------------------------ #
    # 4. confusion matrix grid                                             #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(n_sc, n_mo,
                              figsize=(4.5 * n_mo, 4.5 * n_sc),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.40, "wspace": 0.15})
    if n_sc == 1:
        axes = axes.reshape(1, n_mo)
    last_im = None
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
            last_im = ax.imshow(row_norm, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{sc.upper()} / {mo} @ T={T_max}", fontsize=9)
            if sci == n_sc - 1:
                ax.set_xlabel("predicted k", fontsize=8)
            if mi == 0:
                ax.set_ylabel("true k", fontsize=8)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6,
                     label="fraction of true-class samples")
    plt.suptitle(f"Test confusion matrix @ T={T_max} (row-normalized)", fontsize=12)
    out = root / "confusion_grid.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
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
