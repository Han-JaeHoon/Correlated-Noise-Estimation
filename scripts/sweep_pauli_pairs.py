# scripts/sweep_pauli_pairs.py

"""
Sweep all 9 two-wire Pauli fault pairs on (control, target) at a fixed
n_rounds / error_rounds / measurement mode, and report degeneracy statistics
per fault type.

Reuses helpers from analyze_round1_degeneracy.py.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_round1_degeneracy import (
    collect_syndromes,
    group_by_syndrome,
    parse_error_rounds,
)
from src.io_utils import ensure_dir, save_csv


PAULI_PAIRS = [
    ("X", "X"), ("X", "Y"), ("X", "Z"),
    ("Y", "X"), ("Y", "Y"), ("Y", "Z"),
    ("Z", "X"), ("Z", "Y"), ("Z", "Z"),
]


def sweep(n_rounds, error_rounds, reset_after_measure):
    """
    Run all 9 Pauli pairs and collect degeneracy stats.
    """
    rows = []
    n_bits = n_rounds * 8
    zero_syndrome = "0" * n_bits

    for pair in PAULI_PAIRS:
        results = collect_syndromes(
            n_rounds=n_rounds,
            reset_after_measure=reset_after_measure,
            error_types=pair,
            error_rounds=error_rounds,
        )
        groups = group_by_syndrome(results)

        unique = len(groups)
        silent = len(groups.get(zero_syndrome, []))
        max_class = max(len(m) for m in groups.values())
        n_dup_classes = sum(1 for m in groups.values() if len(m) > 1)

        rows.append(
            {
                "fault_type": "".join(pair),
                "unique_syndromes": unique,
                "silent_cnots": silent,
                "max_class_size": max_class,
                "degeneracy_classes": n_dup_classes,
            }
        )

    return rows


def print_table(rows, n_rounds, error_rounds, reset_after_measure):
    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)

    print(
        f"Pauli pair sweep   n_rounds={n_rounds}  "
        f"error_rounds=[{rounds_label}]  mode={mode_label}"
    )
    print()
    print(
        f"{'fault':>6}  {'unique':>6}  {'silent':>6}  "
        f"{'max_class':>9}  {'deg_classes':>11}"
    )
    print("-" * 50)
    for r in rows:
        print(
            f"{r['fault_type']:>6}  {r['unique_syndromes']:>6}  "
            f"{r['silent_cnots']:>6}  {r['max_class_size']:>9}  "
            f"{r['degeneracy_classes']:>11}"
        )


def plot_bar(rows, n_rounds, error_rounds, reset_after_measure, out_path):
    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)
    labels = [r["fault_type"] for r in rows]
    uniques = [r["unique_syndromes"] for r in rows]
    silents = [r["silent_cnots"] for r in rows]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, uniques, width, label="unique syndromes",
           color="#4C78A8")
    ax.bar(x + width / 2, silents, width, label="silent CNOTs",
           color="#E45756")

    ax.axhline(24, color="gray", linewidth=0.8, linestyle=":")
    ax.text(len(labels) - 0.5, 24.3, "24 (all CNOTs)",
            color="gray", fontsize=8, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_xlabel("fault Pauli pair (control, target)")
    ax.set_title(
        f"Pauli sweep  n_rounds={n_rounds}  "
        f"error_rounds=[{rounds_label}]  mode={mode_label}"
    )
    ax.set_ylim(0, 26)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep 9 Pauli fault pairs and report degeneracy stats."
        )
    )
    parser.add_argument(
        "--n-rounds", type=int, default=2,
        help="Number of stabilizer rounds. Default: 2",
    )
    parser.add_argument(
        "--error-rounds", type=str, default="0",
        help='Comma-separated rounds where fault is injected. Default: 0',
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Do NOT reset ancilla after mid-circuit measurement.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    n_rounds = args.n_rounds
    error_rounds = parse_error_rounds(args.error_rounds)
    reset_after_measure = not args.no_reset

    for r in error_rounds:
        if r < 0 or r >= n_rounds:
            raise SystemExit(
                f"error_rounds {r} out of range [0, {n_rounds - 1}]"
            )

    rows = sweep(n_rounds, error_rounds, reset_after_measure)

    print_table(rows, n_rounds, error_rounds, reset_after_measure)

    mode_tag = "reset" if reset_after_measure else "noreset"
    rounds_tag = "-".join(str(r) for r in error_rounds) if error_rounds else "none"
    tag = f"pauli_sweep_nrounds{n_rounds}_r{rounds_tag}_{mode_tag}"

    out_dir = PROJECT_ROOT / "data" / "analysis" / "2_pauli_sweep_summary"
    ensure_dir(out_dir)

    csv_path = out_dir / f"{tag}.csv"
    save_csv(rows, csv_path)
    print(f"\nSaved table  : {csv_path}")

    png_path = out_dir / f"{tag}.png"
    plot_bar(rows, n_rounds, error_rounds, reset_after_measure, png_path)
    print(f"Saved figure : {png_path}")


if __name__ == "__main__":
    main()
