"""
Compute the Bayes-optimal ceiling for dominant-CNOT identification under R1.

Sweeps a grid of (p_bg, p_high, T) and saves accuracy + confusion matrices.

Output: data/analysis/6_sequence_ceiling/
    lookup.npy                              # (24, 15) single-round contributions
    ceiling_grid.csv                        # one row per grid point
    confusion_pbg{p}_phigh{p}_T{T}.csv      # per-grid-point 24x24 confusion
    ceiling_curves.png                      # accuracy vs T for each (p_bg, p_high)
    ceiling_heatmap_T{T}.png                # 2D map p_bg x p_high at fixed T

Example:
    python scripts/compute_r1_ceiling.py \
        --p-bg 1e-3 1e-2 --ratio 10 30 --T 10 50 100 --n-test 300
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.ceiling import (
    bayes_classifier_confusion,
    compute_all_distributions,
    compute_single_round_lookup,
    overall_accuracy,
    per_class_accuracy,
)
from src.config import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p-bg",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2],
        help="background fault probability values",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=[10.0, 30.0],
        help="p_high / p_bg ratio values",
    )
    parser.add_argument(
        "--T",
        type=int,
        nargs="+",
        default=[10, 50, 100],
        help="sequence length values",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=300,
        help="Monte Carlo samples per true class",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "analysis" / "6_sequence_ceiling",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="use no-reset mid-measurement (default: reset)",
    )
    parser.add_argument(
        "--reload-lookup",
        action="store_true",
        help="force recompute the single-round lookup",
    )
    return parser.parse_args()


def load_or_compute_lookup(out_dir: Path, reset: bool, force: bool) -> np.ndarray:
    name = f"lookup_{'reset' if reset else 'noreset'}.npy"
    path = out_dir / name
    if path.exists() and not force:
        print(f"[ceiling] loading cached lookup from {path}")
        return np.load(path)
    print(f"[ceiling] computing single-round lookup (reset={reset})...")
    contributions = compute_single_round_lookup(reset=reset)
    np.save(path, contributions)
    print(f"  saved -> {path}  shape={contributions.shape}")
    return contributions


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reset = not args.no_reset
    mode_tag = "reset" if reset else "noreset"

    contributions = load_or_compute_lookup(out_dir, reset, args.reload_lookup)

    rng = np.random.default_rng(args.seed)

    rows = []
    for p_bg in args.p_bg:
        for ratio in args.ratio:
            p_high = p_bg * ratio
            if p_high > 1.0:
                print(f"  skip p_bg={p_bg} ratio={ratio} (p_high>1)")
                continue
            dists = compute_all_distributions(p_bg, p_high, contributions)

            for T in args.T:
                confusion = bayes_classifier_confusion(
                    dists=dists, T=T, n_test=args.n_test, rng=rng
                )
                acc = overall_accuracy(confusion)
                pcls = per_class_accuracy(confusion)

                row = {
                    "p_bg": p_bg,
                    "p_high": p_high,
                    "ratio": ratio,
                    "T": T,
                    "accuracy": acc,
                    "per_class_min": float(pcls.min()),
                    "per_class_max": float(pcls.max()),
                    "per_class_mean": float(pcls.mean()),
                }
                rows.append(row)

                print(
                    f"  p_bg={p_bg:.1e}  p_high={p_high:.1e}  T={T:>4d}  "
                    f"acc={acc:.3f}  per-class [{pcls.min():.2f},"
                    f" {pcls.max():.2f}]"
                )

                conf_name = (
                    f"confusion_pbg{p_bg}_phigh{p_high:.4g}_T{T}_{mode_tag}.csv"
                )
                np.savetxt(
                    out_dir / conf_name,
                    confusion,
                    fmt="%d",
                    delimiter=",",
                )

    # Save grid summary CSV
    grid_path = out_dir / f"ceiling_grid_{mode_tag}.csv"
    if rows:
        with open(grid_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[ceiling] grid -> {grid_path}")

    # Plot accuracy vs T
    if rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        seen = set()
        for r in rows:
            key = (r["p_bg"], r["ratio"])
            seen.add(key)
        for p_bg, ratio in sorted(seen):
            sub = [r for r in rows if r["p_bg"] == p_bg and r["ratio"] == ratio]
            sub.sort(key=lambda r: r["T"])
            Ts = [r["T"] for r in sub]
            accs = [r["accuracy"] for r in sub]
            p_high = p_bg * ratio
            ax.plot(
                Ts,
                accs,
                marker="o",
                label=f"p_bg={p_bg:.0e}  p_high={p_high:.0e}",
            )
        ax.axhline(1 / 24, color="gray", linestyle="--", label="random (1/24)")
        ax.axhline(1.0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("T (rounds)")
        ax.set_ylabel("Bayes-optimal accuracy")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"R1 ceiling: dominant-CNOT identification ({mode_tag})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"ceiling_curves_{mode_tag}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[ceiling] curves -> ceiling_curves_{mode_tag}.png")


if __name__ == "__main__":
    main()
