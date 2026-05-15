# scripts/visualize_cross_pauli_per_pauli.py

"""
Per-Pauli heatmap visualization with cross-Pauli class id.

For each of the 9 Pauli pairs, produce a heatmap in the same format as
`nrounds{N}_e{XZ}_r{rounds}_{mode}_degeneracy.png`, but the top stripe
text shows a GLOBAL class id (assigned across all 216 = 24 CNOT x 9 Pauli
cases), and the stripe color encodes "conflict size" — how many distinct
CNOTs share that syndrome across all Pauli pairs.

How to read:
  * Stripe color
      white  = conflict size 1 (no other CNOT shares this syndrome under any
               Pauli)  -> uniquely identifies the CNOT
      yellow = 2,  orange = 3,  red = 4+
  * Stripe text "<gid>/<csize>"
      gid   = global class id (0..N_unique-1, same id across files == same
              syndrome)
      csize = number of distinct CNOTs in this global class

  Two cells in different Pauli files with the SAME gid -> same syndrome
  -> cross-Pauli ambiguity.

Default mode is no-reset (lower experimental cost, no information loss).
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
    stab_labels_for_rounds,
)
from src.io_utils import ensure_dir, save_csv


PAULI_PAIRS = [
    ("X", "X"), ("X", "Y"), ("X", "Z"),
    ("Y", "X"), ("Y", "Y"), ("Y", "Z"),
    ("Z", "X"), ("Z", "Y"), ("Z", "Z"),
]


def collect_per_pauli(n_rounds, reset_after_measure, error_rounds):
    """
    Returns dict mapping pauli_str -> list of records (one per CNOT).
    """
    by_pauli = {}
    for pair in PAULI_PAIRS:
        results = collect_syndromes(
            n_rounds=n_rounds,
            reset_after_measure=reset_after_measure,
            error_types=pair,
            error_rounds=error_rounds,
        )
        by_pauli["".join(pair)] = results
    return by_pauli


def compute_global_classes(by_pauli):
    """
    Group all 216 records by syndrome. Assign each unique syndrome a
    global class id (sorted so larger conflicts get smaller ids).

    Returns:
      syndrome_to_gid     : dict[str -> int]
      syndrome_to_csize   : dict[str -> int]  (# distinct CNOTs sharing it)
      syndrome_to_members : dict[str -> list of (pauli, record)]
    """
    by_syndrome = defaultdict(list)
    for pauli_str, results in by_pauli.items():
        for r in results:
            by_syndrome[r["syndrome_bits"]].append((pauli_str, r))

    syndrome_to_csize = {
        s: len({rec["cnot_idx"] for _, rec in members})
        for s, members in by_syndrome.items()
    }

    # Sort: larger csize first, then by bit string.
    syndrome_keys = sorted(
        by_syndrome.keys(),
        key=lambda s: (-syndrome_to_csize[s], s),
    )
    syndrome_to_gid = {s: i for i, s in enumerate(syndrome_keys)}

    return syndrome_to_gid, syndrome_to_csize, dict(by_syndrome)


def plot_one_pauli(
    results,
    pauli_str,
    syndrome_to_gid,
    syndrome_to_csize,
    n_rounds,
    error_rounds,
    reset_after_measure,
    out_path,
):
    order = sorted(results, key=lambda r: r["cnot_idx"])
    matrix = np.array([m["bits"] for m in order]).T
    n_stab = matrix.shape[0]

    col_labels = [
        f"CNOT{m['cnot_idx']:02d}  {m['stab_type']}{m['stab_index']}  "
        f"{m['control']:2d}->{m['target']:2d}"
        for m in order
    ]
    y_labels = stab_labels_for_rounds(n_rounds)

    col_csize = np.array(
        [syndrome_to_csize[m["syndrome_bits"]] for m in order]
    )
    col_gid = np.array(
        [syndrome_to_gid[m["syndrome_bits"]] for m in order]
    )

    max_csize = max(2, int(col_csize.max()))

    fig_height = 1.5 + 0.32 * n_stab
    fig, (ax_g, ax) = plt.subplots(
        2, 1,
        figsize=(13, fig_height),
        gridspec_kw={"height_ratios": [1, n_stab]},
    )

    # Top stripe: color = conflict size, text = "<gid>/<csize>".
    cmap = plt.get_cmap("YlOrRd")
    ax_g.imshow(
        col_csize.reshape(1, -1),
        cmap=cmap, aspect="auto",
        vmin=1, vmax=max_csize,
    )
    for j in range(len(order)):
        gid = int(col_gid[j])
        csize = int(col_csize[j])
        ax_g.text(
            j, 0, f"{gid}/{csize}",
            ha="center", va="center",
            color="black" if csize <= max_csize * 0.6 else "white",
            fontsize=6,
        )
    ax_g.set_xticks([])
    ax_g.set_yticks([0])
    ax_g.set_yticklabels(["global id / csize"], fontsize=8)

    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)
    ax_g.set_title(
        f"n_rounds={n_rounds}  fault={pauli_str} at round(s) [{rounds_label}]"
        f"  mode={mode_label}   [cross-Pauli view]"
    )

    # Main syndrome heatmap.
    ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    cell_fontsize = 7 if n_rounds <= 2 else 6
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(
                j, i, str(int(v)),
                ha="center", va="center",
                color="white" if v == 1 else "black",
                fontsize=cell_fontsize,
            )

    for r in range(n_rounds):
        ax.axhline(r * 8 + 3.5, color="blue",
                   linewidth=0.9, linestyle="--")
        if r > 0:
            ax.axhline(r * 8 - 0.5, color="red", linewidth=1.5)

    ax.set_yticks(range(n_stab))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=90)
    ax.set_ylabel("Stabilizer ancilla per round")
    ax.set_xlabel(
        f"CNOT location with {pauli_str} fault at round(s) [{rounds_label}]"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_match_table(syndrome_to_members, syndrome_to_gid,
                     syndrome_to_csize, out_path):
    """
    Save one CSV row per (CNOT, Pauli) case. Sorted by global_id so that
    rows with the same global_id are adjacent (= same syndrome).
    """
    rows = []
    for syndrome, members in syndrome_to_members.items():
        gid = syndrome_to_gid[syndrome]
        csize = syndrome_to_csize[syndrome]
        for pauli_str, rec in members:
            rows.append({
                "global_id": gid,
                "conflict_size": csize,
                "pauli": pauli_str,
                "cnot_idx": rec["cnot_idx"],
                "stab_type": rec["stab_type"],
                "stab_index": rec["stab_index"],
                "control": rec["control"],
                "target": rec["target"],
                "syndrome_bits": syndrome,
            })
    rows.sort(key=lambda r: (r["global_id"], r["pauli"], r["cnot_idx"]))
    save_csv(rows, out_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Per-Pauli syndrome heatmaps with cross-Pauli global class ids."
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

    by_pauli = collect_per_pauli(n_rounds, reset_after_measure, error_rounds)
    (
        syndrome_to_gid,
        syndrome_to_csize,
        syndrome_to_members,
    ) = compute_global_classes(by_pauli)

    out_dir = (
        PROJECT_ROOT / "data" / "analysis" / "4_cross_pauli_per_pauli_view"
        / f"nrounds{n_rounds}_r{rounds_tag}_{mode_tag}"
    )
    ensure_dir(out_dir)

    # Per-Pauli heatmaps.
    for pair in PAULI_PAIRS:
        pair_str = "".join(pair)
        out_path = out_dir / f"e{pair_str}.png"
        plot_one_pauli(
            by_pauli[pair_str],
            pair_str,
            syndrome_to_gid,
            syndrome_to_csize,
            n_rounds,
            error_rounds,
            reset_after_measure,
            out_path,
        )
        print(f"Saved: {out_path}")

    # Match table (sorted by global id).
    csv_path = out_dir / "matches_by_global_id.csv"
    save_match_table(
        syndrome_to_members, syndrome_to_gid, syndrome_to_csize, csv_path
    )
    print(f"Saved: {csv_path}")

    # Summary.
    n_unique = len(syndrome_to_gid)
    n_ambig = sum(1 for s, c in syndrome_to_csize.items() if c > 1)
    print()
    print(f"Total (CNOT, Pauli) cases  : {sum(len(v) for v in by_pauli.values())}")
    print(f"Unique syndromes (global)  : {n_unique}")
    print(f"Ambiguous syndromes        : {n_ambig}")
    print(f"Output directory           : {out_dir}")


if __name__ == "__main__":
    main()
