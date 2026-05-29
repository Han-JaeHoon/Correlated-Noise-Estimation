"""
Round-by-round event pattern analysis.

The fault is injected at round 0 only.  This script asks how the (3, 8)
syndrome bit pattern looks as a function of round t in {0, 1, 2}:

  (1) Per-category mean event rate per (round, stabilizer) cell.  Reveals
      which stabilizers light up first, and whether signal persists or
      decays into rounds 1-2.

  (2) Per-round all-zero rate.  Counts how often round t carries no
      detection event for category-X faults — a proxy for "silent rounds".

  (3) Reset vs no-reset bit-rate comparison per round.  Highlights where
      ancilla-state leakage between rounds matters most.

Outputs (data/analysis/10_fault_enumeration_patterns/):
    round_event_rate_heatmap.png
    round_zero_fraction.csv
    reset_vs_noreset_round_event_rate.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fault_enum_analysis._common import (  # noqa: E402
    N_ROUNDS,
    N_STAB,
    STAB_LABELS,
    ensure_out_dir,
    load_table,
    select,
)


def mean_event_rate(sub: dict) -> np.ndarray:
    """(3, 8) float, averaged across all records in sub."""

    return sub["syndrome"].astype(np.float32).mean(axis=0)


def plot_per_cat_per_mode_heatmap(t: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    vmax = 0.0

    panels = []
    for i, cat in enumerate(["A", "C"]):
        for j, mode in enumerate(["reset", "noreset"]):
            sub = select(t, cat=cat, mode=mode)
            m = mean_event_rate(sub)
            panels.append((axes[i, j], m, cat, mode))
            vmax = max(vmax, float(m.max()))

    for ax, m, cat, mode in panels:
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"Category {cat} — mode={mode}")
        ax.set_xticks(range(N_STAB))
        ax.set_xticklabels(STAB_LABELS, rotation=0)
        ax.set_yticks(range(N_ROUNDS))
        ax.set_yticklabels([f"round {r}" for r in range(N_ROUNDS)])
        for r in range(N_ROUNDS):
            for s in range(N_STAB):
                txt = f"{m[r, s]:.2f}"
                ax.text(s, r, txt, ha="center", va="center", fontsize=8,
                        color="white" if m[r, s] < vmax * 0.55 else "black")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="mean event rate")
    fig.suptitle("Per-(round, stabilizer) mean event rate over all single-fault cases")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def round_zero_fraction_csv(t: dict, out_path: Path) -> None:
    """For each (cat, mode, round) record the fraction of cases where that
    round's 8-bit slice is all zero — i.e. the fault is silent in that round."""

    rows = []
    for cat in ["A", "C"]:
        for mode in ["reset", "noreset"]:
            sub = select(t, cat=cat, mode=mode)
            S = sub["syndrome"]  # (n, 3, 8)
            n = S.shape[0]
            for r in range(N_ROUNDS):
                slice_r = S[:, r, :]
                zero_frac = float((slice_r.sum(axis=1) == 0).sum()) / n
                rows.append((cat, mode, r, n, round(zero_frac, 4)))

    with open(out_path, "w") as f:
        f.write("category,mode,round,n_cases,zero_fraction\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"Wrote {out_path}")


def plot_reset_vs_noreset_round_rate(t: dict, out_path: Path) -> None:
    """Per-round mean event rate, summed over stabilizers, for cat=C only.
    reset vs noreset overlay."""

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for mode, color in [("reset", "tab:blue"), ("noreset", "tab:red")]:
        sub = select(t, cat="C", mode=mode)
        m = mean_event_rate(sub)  # (3, 8)
        per_round_total = m.sum(axis=1)  # 8 stabs summed
        ax.plot(range(N_ROUNDS), per_round_total, "-o", color=color, label=f"mode={mode}")

    ax.set_xlabel("round t")
    ax.set_ylabel("Σ_stab mean event rate (Cat C)")
    ax.set_title("Round propagation of a round-0 single CNOT fault")
    ax.set_xticks(range(N_ROUNDS))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = ensure_out_dir()
    t = load_table()

    plot_per_cat_per_mode_heatmap(t, out_dir / "round_event_rate_heatmap.png")
    print(f"Wrote {out_dir / 'round_event_rate_heatmap.png'}")

    round_zero_fraction_csv(t, out_dir / "round_zero_fraction.csv")

    plot_reset_vs_noreset_round_rate(
        t, out_dir / "reset_vs_noreset_round_event_rate.png"
    )
    print(f"Wrote {out_dir / 'reset_vs_noreset_round_event_rate.png'}")


if __name__ == "__main__":
    main()
