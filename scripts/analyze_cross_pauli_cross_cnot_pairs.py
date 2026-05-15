# scripts/analyze_cross_pauli_cross_cnot_pairs.py

"""
Cross-Pauli cross-CNOT pair analysis.

For all 24 CNOTs x 9 Pauli pairs = 216 (CNOT, Pauli) cases, enumerate every
unordered pair of cases that produce identical syndromes and categorize:

  same_cnot_diff_pauli   (CNOT_i, alpha) and (CNOT_i, beta), alpha != beta
                          -- intra-CNOT route degeneracy; harmless for
                          identifying the faulty CNOT
  diff_cnot_same_pauli   (CNOT_i, alpha) and (CNOT_j, alpha), i != j
                          -- within-Pauli degeneracy; resolvable if Pauli
                          type is known
  diff_cnot_diff_pauli   (CNOT_i, alpha) and (CNOT_j, beta), i != j,
                          alpha != beta
                          -- the "user's question": fundamentally cannot
                          be resolved from syndrome alone if Pauli is
                          unknown

Default mode is no-reset.
"""

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_round1_degeneracy import (
    collect_syndromes,
    parse_error_rounds,
)
from src.io_utils import ensure_dir, save_csv


PAULI_PAIRS = [
    ("X", "X"), ("X", "Y"), ("X", "Z"),
    ("Y", "X"), ("Y", "Y"), ("Y", "Z"),
    ("Z", "X"), ("Z", "Y"), ("Z", "Z"),
]


def collect_all_cases(n_rounds, reset_after_measure, error_rounds):
    records = []
    for pair in PAULI_PAIRS:
        results = collect_syndromes(
            n_rounds=n_rounds,
            reset_after_measure=reset_after_measure,
            error_types=pair,
            error_rounds=error_rounds,
        )
        for r in results:
            records.append(
                {
                    "cnot_idx": r["cnot_idx"],
                    "stab_type": r["stab_type"],
                    "stab_index": r["stab_index"],
                    "control": r["control"],
                    "target": r["target"],
                    "pauli": "".join(pair),
                    "syndrome_bits": r["syndrome_bits"],
                }
            )
    return records


def categorize(a, b):
    same_cnot = a["cnot_idx"] == b["cnot_idx"]
    same_pauli = a["pauli"] == b["pauli"]
    if same_cnot and same_pauli:
        return "trivial"  # cannot happen since records are unique
    if same_cnot:
        return "same_cnot_diff_pauli"
    if same_pauli:
        return "diff_cnot_same_pauli"
    return "diff_cnot_diff_pauli"


def enumerate_pairs(records):
    by_syndrome = defaultdict(list)
    for r in records:
        by_syndrome[r["syndrome_bits"]].append(r)

    pairs = []
    for syndrome, members in by_syndrome.items():
        if len(members) < 2:
            continue
        for a, b in combinations(members, 2):
            cat = categorize(a, b)
            pairs.append(
                {
                    "category": cat,
                    "syndrome": syndrome,
                    "cnot_a": a["cnot_idx"],
                    "pauli_a": a["pauli"],
                    "label_a": (
                        f"CNOT{a['cnot_idx']:02d} {a['stab_type']}{a['stab_index']} "
                        f"{a['control']}->{a['target']}"
                    ),
                    "cnot_b": b["cnot_idx"],
                    "pauli_b": b["pauli"],
                    "label_b": (
                        f"CNOT{b['cnot_idx']:02d} {b['stab_type']}{b['stab_index']} "
                        f"{b['control']}->{b['target']}"
                    ),
                }
            )
    return pairs


def plot_cnot_pair_matrix(diff_pairs, n_rounds, error_rounds,
                         reset_after_measure, out_path):
    """
    24 x 24 symmetric matrix.
    Cell (i, j) = number of (Pauli_alpha, Pauli_beta) pairs with alpha != beta
                  such that (CNOT_i, alpha) and (CNOT_j, beta) have identical
                  syndromes.
    """
    matrix = np.zeros((24, 24), dtype=int)
    for p in diff_pairs:
        i, j = p["cnot_a"], p["cnot_b"]
        matrix[i, j] += 1
        matrix[j, i] += 1

    max_val = max(1, int(matrix.max()))

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal",
                   vmin=0, vmax=max_val)

    for i in range(24):
        for j in range(24):
            v = int(matrix[i, j])
            if v > 0:
                ax.text(
                    j, i, str(v),
                    ha="center", va="center",
                    color="black" if v <= max_val * 0.55 else "white",
                    fontsize=6,
                )

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{i:02d}" for i in range(24)], fontsize=7,
                       rotation=0)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"CNOT{i:02d}" for i in range(24)], fontsize=7)
    ax.set_xlabel("CNOT_j  (index)")
    ax.set_ylabel("CNOT_i")

    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)
    ax.set_title(
        f"Cross-Pauli cross-CNOT collision counts\n"
        f"Cell (i,j) = # (Pauli_alpha, Pauli_beta) pairs with alpha != beta\n"
        f"such that (CNOT_i, alpha) and (CNOT_j, beta) produce identical syndromes\n"
        f"n_rounds={n_rounds}  error_rounds=[{rounds_label}]  mode={mode_label}"
    )

    plt.colorbar(im, ax=ax, shrink=0.7,
                 label="# (Pauli_a, Pauli_b) pairs colliding")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Find pairs (CNOT_i, Pauli_alpha) and (CNOT_j, Pauli_beta) with "
            "i != j and alpha != beta that produce identical syndromes."
        )
    )
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--error-rounds", type=str, default="0")
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset ancilla after mid-circuit measurement. "
             "Default is NO reset.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    n_rounds = args.n_rounds
    error_rounds = parse_error_rounds(args.error_rounds)
    reset_after_measure = args.reset

    for r in error_rounds:
        if r < 0 or r >= n_rounds:
            raise SystemExit(
                f"error_rounds {r} out of range [0, {n_rounds - 1}]"
            )

    mode_tag = "reset" if reset_after_measure else "noreset"
    rounds_tag = (
        "-".join(str(r) for r in error_rounds) if error_rounds else "none"
    )

    records = collect_all_cases(n_rounds, reset_after_measure, error_rounds)
    pairs = enumerate_pairs(records)

    counts = defaultdict(int)
    for p in pairs:
        counts[p["category"]] += 1

    mode_label = "reset" if reset_after_measure else "no-reset"
    print(f"Setup: n_rounds={n_rounds}  "
          f"error_rounds=[{rounds_tag}]  mode={mode_label}")
    print(f"Total (CNOT, Pauli) cases  : {len(records)}")
    print(f"Total syndrome-sharing pairs: {len(pairs)}")
    print()
    print(f"Category breakdown:")
    print(f"  same_cnot_diff_pauli   : {counts['same_cnot_diff_pauli']:>4}  "
          f"(intra-CNOT route degeneracy; harmless)")
    print(f"  diff_cnot_same_pauli   : {counts['diff_cnot_same_pauli']:>4}  "
          f"(within-Pauli degeneracy; needs Pauli to resolve)")
    print(f"  diff_cnot_diff_pauli   : {counts['diff_cnot_diff_pauli']:>4}  "
          f"(*** the user's case: fundamentally ambiguous ***)")
    print()

    diff_pairs = [p for p in pairs if p["category"] == "diff_cnot_diff_pauli"]

    if diff_pairs:
        print(f"First 10 cross-Pauli cross-CNOT collisions:")
        for p in diff_pairs[:10]:
            print(f"  {p['pauli_a']}-{p['label_a']:24}  <->  "
                  f"{p['pauli_b']}-{p['label_b']:24}")
        print()

    out_dir = (
        PROJECT_ROOT / "data" / "analysis" / "5_cross_pauli_pair_collisions"
    )
    ensure_dir(out_dir)

    tag = f"cross_pauli_pairs_nrounds{n_rounds}_r{rounds_tag}_{mode_tag}"

    csv_path = out_dir / f"{tag}.csv"
    save_csv(diff_pairs, csv_path)
    print(f"Saved table  (diff_cnot_diff_pauli only): {csv_path}")

    png_path = out_dir / f"{tag}.png"
    plot_cnot_pair_matrix(diff_pairs, n_rounds, error_rounds,
                          reset_after_measure, png_path)
    print(f"Saved figure : {png_path}")


if __name__ == "__main__":
    main()
