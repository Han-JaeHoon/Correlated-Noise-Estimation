# scripts/analyze_round1_degeneracy.py

"""
Single-CNOT Pauli fault syndrome degeneracy analysis.

For each of the 24 directed CNOT locations, inject the Pauli fault
"<error_types[0]> on control, <error_types[1]> on target" immediately
after that CNOT at every round listed in error_rounds. The circuit then
runs for n_rounds total stabilizer rounds.

Group CNOT locations by identical syndrome and visualize.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fault_schedule import make_fixed_cnot_fault_schedule
from src.io_utils import ensure_dir, save_csv
from src.postprocess import reshape_mid_measure_syndrome
from src.simulator import make_repeated_stabilizer_qnode
from src.surface_code_layout import enumerate_stabilizer_cnots


SHOTS = 1
STAB_LABELS_BASE = ["Z0", "Z1", "Z2", "Z3", "X0", "X1", "X2", "X3"]


def stab_labels_for_rounds(n_rounds):
    if n_rounds == 1:
        return list(STAB_LABELS_BASE)
    return [f"{s} r{r}" for r in range(n_rounds) for s in STAB_LABELS_BASE]


def parse_error_types(s):
    """
    "XZ" or "X,Z" -> ("X", "Z")
    """
    if "," in s:
        parts = s.split(",")
    else:
        parts = list(s)
    parts = tuple(p.strip().upper() for p in parts if p.strip())
    if len(parts) != 2:
        raise ValueError(
            f"error_types must contain two Pauli types. Got: {s}"
        )
    for p in parts:
        if p not in {"X", "Y", "Z"}:
            raise ValueError(f"Invalid Pauli type: {p}")
    return parts


def parse_error_rounds(s):
    """
    "0" -> [0]
    "0,1" -> [0, 1]
    """
    parts = [int(p.strip()) for p in s.split(",") if p.strip()]
    return sorted(set(parts))


def collect_syndromes(n_rounds, reset_after_measure, error_types, error_rounds):
    qnode = make_repeated_stabilizer_qnode(
        n_rounds=n_rounds,
        shots=SHOTS,
        mid_measure=True,
        reset_after_measure=reset_after_measure,
    )

    cnot_locations = enumerate_stabilizer_cnots()
    results = []

    for idx, loc in enumerate(cnot_locations):
        control, target = loc["control"], loc["target"]

        schedule = make_fixed_cnot_fault_schedule(
            control=control,
            target=target,
            error_rounds=error_rounds,
            error_types=error_types,
        )

        raw = qnode(fault_schedule=schedule)
        syndrome = reshape_mid_measure_syndrome(
            raw, n_rounds=n_rounds, shots=SHOTS
        )
        bits = syndrome[0].flatten().astype(int)

        results.append(
            {
                "cnot_idx": idx,
                "stab_type": loc["stab_type"],
                "stab_index": loc["stab_index"],
                "control": control,
                "target": target,
                "syndrome_bits": "".join(str(b) for b in bits),
                "bits": bits,
            }
        )

    return results


def group_by_syndrome(results):
    groups = defaultdict(list)
    for r in results:
        groups[r["syndrome_bits"]].append(r)
    return dict(
        sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )


def print_summary(results, groups, n_rounds, reset_after_measure,
                  error_types, error_rounds):
    duplicate_classes = [g for g in groups.values() if len(g) > 1]
    mode_label = "reset" if reset_after_measure else "no-reset"

    print(f"n_rounds           : {n_rounds}")
    print(f"fault Paulis       : {error_types[0]} on control / "
          f"{error_types[1]} on target")
    print(f"fault rounds       : {error_rounds}")
    print(f"measurement mode   : {mode_label}")
    print(f"Total CNOTs        : {len(results)}")
    print(f"Unique syndromes   : {len(groups)}")
    print(f"Degeneracy classes : {len(duplicate_classes)} "
          f"(classes with >1 member)")
    print()

    if not duplicate_classes:
        print(f"All {len(results)} CNOTs produce distinct syndromes.")
        return

    print("Degeneracy classes (size > 1)")
    print("-" * 72)
    for syndrome_bits, members in groups.items():
        if len(members) <= 1:
            continue
        print(f"  syndrome  {syndrome_bits}   ({len(members)} CNOTs)")
        for m in members:
            print(
                f"      CNOT{m['cnot_idx']:02d}  "
                f"{m['stab_type']}{m['stab_index']}  "
                f"{m['control']:2d} -> {m['target']:2d}"
            )
        print()


def save_table(results, groups, out_path):
    group_id = {key: i for i, key in enumerate(groups.keys())}
    rows = []
    for r in results:
        rows.append(
            {
                "cnot_idx": r["cnot_idx"],
                "stab_type": r["stab_type"],
                "stab_index": r["stab_index"],
                "control": r["control"],
                "target": r["target"],
                "syndrome_bits": r["syndrome_bits"],
                "group_id": group_id[r["syndrome_bits"]],
                "group_size": len(groups[r["syndrome_bits"]]),
            }
        )
    save_csv(rows, out_path)


def plot_heatmap(results, groups, n_rounds, reset_after_measure,
                 error_types, error_rounds, out_path):
    order = sorted(results, key=lambda r: r["cnot_idx"])
    matrix = np.array([m["bits"] for m in order]).T
    n_stab = matrix.shape[0]

    col_labels = [
        f"CNOT{m['cnot_idx']:02d}  {m['stab_type']}{m['stab_index']}  "
        f"{m['control']:2d}->{m['target']:2d}"
        for m in order
    ]
    y_labels = stab_labels_for_rounds(n_rounds)

    group_id = {key: i for i, key in enumerate(groups.keys())}
    group_size = {key: len(members) for key, members in groups.items()}
    col_group_id = np.array(
        [group_id[m["syndrome_bits"]] for m in order]
    )

    fig_height = 1.5 + 0.32 * n_stab
    fig, (ax_g, ax) = plt.subplots(
        2, 1,
        figsize=(13, fig_height),
        gridspec_kw={"height_ratios": [1, n_stab]},
    )

    cmap = plt.get_cmap("tab20", max(20, len(groups)))
    ax_g.imshow(
        col_group_id.reshape(1, -1),
        cmap=cmap, aspect="auto",
        vmin=0, vmax=cmap.N - 1,
    )
    for j, m in enumerate(order):
        gid = group_id[m["syndrome_bits"]]
        gsize = group_size[m["syndrome_bits"]]
        ax_g.text(
            j, 0, f"{gid}/{gsize}",
            ha="center", va="center",
            color="black",
            fontsize=6,
        )
    ax_g.set_xticks([])
    ax_g.set_yticks([0])
    ax_g.set_yticklabels(["class id / size"], fontsize=8)

    mode_label = "reset" if reset_after_measure else "no-reset"
    rounds_label = ",".join(str(r) for r in error_rounds)
    etype_label = "".join(error_types)
    ax_g.set_title(
        f"n_rounds={n_rounds}  fault={etype_label} at round(s) [{rounds_label}]"
        f"  mode={mode_label}   "
        f"({len(groups)} unique / {len(results)} CNOTs)"
    )

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
        f"CNOT location with {etype_label} fault at round(s) [{rounds_label}]"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def filename_tag(n_rounds, error_types, error_rounds, mode_tag):
    etype = "".join(error_types)
    rounds = "-".join(str(r) for r in error_rounds) if error_rounds else "none"
    return f"nrounds{n_rounds}_e{etype}_r{rounds}_{mode_tag}"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Single-CNOT Pauli fault syndrome degeneracy analysis. "
            "Fault is injected at the specified rounds; the circuit then runs "
            "for n_rounds total stabilizer rounds."
        )
    )
    parser.add_argument(
        "--n-rounds", type=int, default=1,
        help="Number of stabilizer rounds. Default: 1",
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Do NOT reset ancilla after mid-circuit measurement.",
    )
    parser.add_argument(
        "--error-types", type=str, default="XZ",
        help='Pauli pair on (control, target). E.g. "XZ", "YY", "ZX". '
             'Default: XZ',
    )
    parser.add_argument(
        "--error-rounds", type=str, default="0",
        help='Comma-separated rounds where fault is injected. '
             'E.g. "0" or "0,1". Default: 0',
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    n_rounds = args.n_rounds
    reset_after_measure = not args.no_reset
    error_types = parse_error_types(args.error_types)
    error_rounds = parse_error_rounds(args.error_rounds)

    for r in error_rounds:
        if r < 0 or r >= n_rounds:
            raise SystemExit(
                f"error_rounds contains {r}, which is out of range "
                f"[0, {n_rounds - 1}]"
            )

    mode_tag = "reset" if reset_after_measure else "noreset"

    out_dir = PROJECT_ROOT / "data" / "analysis" / "1_per_pauli_degeneracy"
    ensure_dir(out_dir)

    results = collect_syndromes(
        n_rounds, reset_after_measure, error_types, error_rounds
    )
    groups = group_by_syndrome(results)

    print_summary(results, groups, n_rounds, reset_after_measure,
                  error_types, error_rounds)

    tag = filename_tag(n_rounds, error_types, error_rounds, mode_tag)
    csv_path = out_dir / f"{tag}_syndromes.csv"
    save_table(results, groups, csv_path)
    print(f"Saved table  : {csv_path}")

    png_path = out_dir / f"{tag}_degeneracy.png"
    plot_heatmap(results, groups, n_rounds, reset_after_measure,
                 error_types, error_rounds, png_path)
    print(f"Saved figure : {png_path}")


if __name__ == "__main__":
    main()
