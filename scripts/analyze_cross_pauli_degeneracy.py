# scripts/analyze_cross_pauli_degeneracy.py

"""
Cross-Pauli degeneracy analysis.

For all 24 CNOTs * 9 Pauli pairs = 216 (CNOT, Pauli) cases, generate the
syndrome and check whether any syndrome is shared by faults at different
CNOT locations (possibly under different Pauli types).

If two cases (CNOT_i, Pauli_alpha) and (CNOT_j, Pauli_beta) with i != j
produce the same syndrome, then observing that syndrome cannot distinguish
CNOT_i from CNOT_j without knowing the Pauli type.

This is the fundamental ambiguity that limits "identify the faulty CNOT
from syndrome alone" estimators.
"""

import argparse
import sys
from collections import defaultdict
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
PAULI_LABELS = ["".join(p) for p in PAULI_PAIRS]


def collect_all_cases(n_rounds, reset_after_measure, error_rounds):
    """
    Returns: list of dicts, one per (CNOT_idx, Pauli_pair) case.
    """
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


def compute_conflict_info(records):
    """
    Group by syndrome. For each record, attach:
      - conflict_size: number of DISTINCT CNOTs sharing this syndrome
      - conflict_cnots: sorted list of those CNOT indices

    A syndrome with conflict_size == 1 uniquely identifies its CNOT.
    A syndrome with conflict_size > 1 is cross-CNOT ambiguous: that exact
    syndrome could come from multiple different CNOTs (possibly via
    different Pauli pairs).
    """
    by_syndrome = defaultdict(list)
    for r in records:
        by_syndrome[r["syndrome_bits"]].append(r)

    syndrome_to_distinct_cnots = {
        s: sorted({m["cnot_idx"] for m in members})
        for s, members in by_syndrome.items()
    }

    for r in records:
        cnots = syndrome_to_distinct_cnots[r["syndrome_bits"]]
        r["conflict_size"] = len(cnots)
        r["conflict_cnots"] = cnots

    return by_syndrome, syndrome_to_distinct_cnots


def print_summary(records, by_syndrome, n_rounds, error_rounds,
                  reset_after_measure):
    total = len(records)
    n_unique = len(by_syndrome)
    n_bits = n_rounds * 8

    zero_syndrome = "0" * n_bits
    silent_members = by_syndrome.get(zero_syndrome, [])
    silent_cnots = sorted({m["cnot_idx"] for m in silent_members})

    # Per-syndrome conflict
    ambig_syndromes = {
        s: members for s, members in by_syndrome.items()
        if len({m["cnot_idx"] for m in members}) > 1
    }
    n_ambig = len(ambig_syndromes)

    # Per-CNOT conflict graph
    cnot_conflicts = defaultdict(set)
    for s, members in ambig_syndromes.items():
        cnots = sorted({m["cnot_idx"] for m in members})
        for i in cnots:
            for j in cnots:
                if i != j:
                    cnot_conflicts[i].add(j)

    n_clean = 24 - len(cnot_conflicts)

    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)

    print(f"n_rounds={n_rounds}  error_rounds=[{rounds_label}]  "
          f"mode={mode_label}")
    print()
    print(f"Total (CNOT, Pauli) cases   : {total}")
    print(f"Unique syndromes            : {n_unique}")
    print(f"Cross-CNOT ambig syndromes  : {n_ambig}  "
          f"(syndromes shared by >=2 distinct CNOTs)")
    print(f"All-zero silent class       : {len(silent_members)} cases, "
          f"{len(silent_cnots)} distinct CNOTs")
    print(f"CNOTs cleanly identifiable  : {n_clean} / 24  "
          f"(no conflict with any other CNOT under any Pauli)")
    print(f"CNOTs with cross-Pauli conf : {len(cnot_conflicts)} / 24")
    print()

    if cnot_conflicts:
        print("Cross-CNOT conflict graph (CNOT i -> {CNOTs it can be confused with}):")
        for i in sorted(cnot_conflicts.keys()):
            partners = sorted(cnot_conflicts[i])
            print(f"  CNOT{i:02d} <-> {partners}")
        print()

    # Largest ambiguous classes
    if ambig_syndromes:
        print("Largest cross-CNOT ambiguous syndromes (top 5):")
        sorted_amb = sorted(
            ambig_syndromes.items(),
            key=lambda kv: -len({m["cnot_idx"] for m in kv[1]}),
        )
        for syn, members in sorted_amb[:5]:
            cnots = sorted({m["cnot_idx"] for m in members})
            print(f"  syndrome {syn}")
            print(f"      {len(cnots)} distinct CNOTs, "
                  f"{len(members)} (CNOT, Pauli) cases")
            for m in members:
                print(f"        CNOT{m['cnot_idx']:02d} "
                      f"{m['stab_type']}{m['stab_index']} "
                      f"{m['control']:2d}->{m['target']:2d}   "
                      f"Pauli {m['pauli']}")
        print()


def save_table(records, out_path):
    rows = []
    for r in records:
        rows.append(
            {
                "cnot_idx": r["cnot_idx"],
                "stab_type": r["stab_type"],
                "stab_index": r["stab_index"],
                "control": r["control"],
                "target": r["target"],
                "pauli": r["pauli"],
                "syndrome_bits": r["syndrome_bits"],
                "conflict_size": r["conflict_size"],
                "conflict_cnots": "|".join(str(c) for c in r["conflict_cnots"]),
            }
        )
    save_csv(rows, out_path)


def plot_conflict_matrix(records, n_rounds, error_rounds,
                         reset_after_measure, out_path):
    """
    24 x 9 matrix. Cell value = number of distinct CNOTs sharing the
    syndrome of (CNOT_i, Pauli_k).

    1 = uniquely identifies the CNOT (good)
    >1 = ambiguous (bad)
    """
    pauli_to_col = {label: i for i, label in enumerate(PAULI_LABELS)}
    matrix = np.zeros((24, 9), dtype=int)
    for r in records:
        matrix[r["cnot_idx"], pauli_to_col[r["pauli"]]] = r["conflict_size"]

    max_val = max(2, int(matrix.max()))

    fig, ax = plt.subplots(figsize=(7.5, 10))
    cmap = plt.get_cmap("YlOrRd")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=1, vmax=max_val)

    for i in range(24):
        for j in range(9):
            v = int(matrix[i, j])
            ax.text(
                j, i, str(v),
                ha="center", va="center",
                color="black" if v <= max_val * 0.6 else "white",
                fontsize=7,
            )

    ax.set_xticks(range(9))
    ax.set_xticklabels(PAULI_LABELS, fontsize=9)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"CNOT{i:02d}" for i in range(24)], fontsize=8)
    ax.set_xlabel("fault Pauli pair (control, target)")
    ax.set_ylabel("CNOT index")

    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)
    ax.set_title(
        f"Cross-Pauli degeneracy: # distinct CNOTs sharing this syndrome\n"
        f"n_rounds={n_rounds}  error_rounds=[{rounds_label}]  mode={mode_label}"
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("conflict size (distinct CNOTs per syndrome)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-Pauli degeneracy analysis: for each unique syndrome "
            "across 24 CNOTs * 9 Pauli pairs, count how many distinct CNOTs "
            "produce it."
        )
    )
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--error-rounds", type=str, default="0")
    parser.add_argument("--no-reset", action="store_true")
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

    mode_tag = "reset" if reset_after_measure else "noreset"
    rounds_tag = "-".join(str(r) for r in error_rounds) if error_rounds else "none"

    records = collect_all_cases(n_rounds, reset_after_measure, error_rounds)
    by_syndrome, _ = compute_conflict_info(records)

    print_summary(records, by_syndrome, n_rounds, error_rounds,
                  reset_after_measure)

    out_dir = PROJECT_ROOT / "data" / "analysis" / "3_cross_pauli_conflict"
    ensure_dir(out_dir)

    tag = f"cross_pauli_nrounds{n_rounds}_r{rounds_tag}_{mode_tag}"

    csv_path = out_dir / f"{tag}.csv"
    save_table(records, csv_path)
    print(f"Saved table : {csv_path}")

    png_path = out_dir / f"{tag}.png"
    plot_conflict_matrix(records, n_rounds, error_rounds,
                         reset_after_measure, png_path)
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
