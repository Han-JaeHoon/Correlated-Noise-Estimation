"""
CNOT x Pauli dispersion analysis (Category C only).

Each of the 24 CNOTs has 15 nontrivial 2-qubit Pauli faults injected post-CNOT
at round 0.  This script asks, *for each CNOT*, how dispersed those 15
syndromes are: is the CNOT highly identifiable (15 distinct syndromes) or
ambiguous (many faults give the same bit pattern)?

We also cross-check against the R1 ambiguity groups discovered in §13:
  G1 = {6, 7},  G2 = {14, 15, 17},  G3 = {18, 19, 20},  G4 = {22, 23}.

For an R1 ambiguity group to exist, the 15-Pauli *multiset* of syndromes must
match across group members.  We compute, for every pair (k, k') of CNOTs,
the multiset Jaccard / overlap statistic on their 15 single-fault syndromes
and verify the R1 groups light up.

Outputs (data/analysis/10_fault_enumeration_patterns/):
    per_cnot_unique_syndromes.csv
    per_cnot_pauli_bitmap_reset.png
    per_cnot_pauli_bitmap_noreset.png
    cnot_multiset_overlap_reset.png
    cnot_multiset_overlap_noreset.png
    r1_group_multiset_check.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fault_enum_analysis._common import (  # noqa: E402
    N_BITS,
    N_CNOT,
    N_ROUNDS,
    N_STAB,
    STAB_LABELS,
    TWO_QUBIT_PAULIS,
    ensure_out_dir,
    flatten_syndrome,
    load_table,
    packbits24,
    select,
)


R1_AMBIGUITY_GROUPS = {
    "G1": [6, 7],
    "G2": [14, 15, 17],
    "G3": [18, 19, 20],
    "G4": [22, 23],
}


def per_cnot_syndromes(sub_C: dict) -> dict[int, list[tuple[str, int]]]:
    """Returns {cnot_id: [(pauli, syndrome_key), ...]} in Pauli lex order."""

    sf = flatten_syndrome(sub_C["syndrome"])
    keys = packbits24(sf)
    out: dict[int, list[tuple[str, int]]] = {k: [] for k in range(N_CNOT)}
    for k, p, key in zip(sub_C["qubit_or_cnot"], sub_C["pauli"], keys):
        out[int(k)].append((str(p), int(key)))
    # Sort each CNOT's list by Pauli lex order (already lex by construction).
    for k in out:
        out[k].sort(key=lambda kp: TWO_QUBIT_PAULIS.index(kp[0]))
    return out


def per_cnot_unique_csv(per_cnot: dict, out_path: Path, mode_label: str) -> None:
    with open(out_path, "w") as f:
        f.write("mode,cnot_id,n_unique_syndromes,n_silent,most_common_size\n")
        for k in range(N_CNOT):
            keys = [kp[1] for kp in per_cnot[k]]
            uniq = len(set(keys))
            silent = sum(1 for x in keys if x == 0)
            most_common = Counter(keys).most_common(1)[0][1]
            f.write(f"{mode_label},{k},{uniq},{silent},{most_common}\n")


def pauli_bitmap_plot(sub_C: dict, out_path: Path, title: str) -> None:
    """Per-CNOT heatmap of (Pauli x bit) -> bit value.

    Stack of 24 small 15 x 24 heatmaps arranged in a 6x4 grid.
    """

    fig, axes = plt.subplots(6, 4, figsize=(15, 18), sharex=True, sharey=True)
    sf = flatten_syndrome(sub_C["syndrome"])

    for k in range(N_CNOT):
        # 15 paulis x 24 bits matrix.
        mask = sub_C["qubit_or_cnot"] == k
        sub_k = sf[mask]
        ps_k = sub_C["pauli"][mask]
        # Sort by Pauli lex order.
        order = np.argsort([TWO_QUBIT_PAULIS.index(p) for p in ps_k])
        sub_k = sub_k[order]
        ps_k = ps_k[order]

        ax = axes[k // 4, k % 4]
        ax.imshow(sub_k, cmap="Greys", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"CNOT {k:02d}", fontsize=9)
        ax.set_yticks(range(15))
        ax.set_yticklabels(ps_k.tolist(), fontsize=6)
        # x-axis: 3 rounds x 8 stabs.
        xticks = list(range(0, N_BITS, 4))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{i // N_STAB}.{STAB_LABELS[i % N_STAB]}"
                            for i in xticks], rotation=90, fontsize=5)
        for r in range(1, N_ROUNDS):
            ax.axvline(r * N_STAB - 0.5, color="red", lw=0.5, alpha=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def multiset_overlap_matrix(sub_C: dict) -> np.ndarray:
    """For each pair (k, k') compute |multiset overlap| / 15.

    overlap(M1, M2) = sum_x min(M1[x], M2[x]) where M_i is the multiset of
    syndrome keys for CNOT i.  Returns (24, 24) float in [0, 1].
    """

    sf = flatten_syndrome(sub_C["syndrome"])
    keys = packbits24(sf)
    per_cnot_counter: list[Counter] = [Counter() for _ in range(N_CNOT)]
    for k, key in zip(sub_C["qubit_or_cnot"], keys):
        per_cnot_counter[int(k)][int(key)] += 1

    M = np.zeros((N_CNOT, N_CNOT), dtype=np.float32)
    for i in range(N_CNOT):
        for j in range(N_CNOT):
            inter = 0
            for x, c in per_cnot_counter[i].items():
                inter += min(c, per_cnot_counter[j].get(x, 0))
            M[i, j] = inter / 15.0
    return M


def plot_overlap_matrix(M: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="multiset overlap / 15")
    ax.set_title(title)
    ax.set_xlabel("CNOT j")
    ax.set_ylabel("CNOT i")
    ax.set_xticks(range(N_CNOT))
    ax.set_yticks(range(N_CNOT))
    ax.set_xticklabels([str(k) for k in range(N_CNOT)], fontsize=7)
    ax.set_yticklabels([str(k) for k in range(N_CNOT)], fontsize=7)

    # Highlight R1 groups with red rectangles around clusters.
    for name, members in R1_AMBIGUITY_GROUPS.items():
        mn, mx = min(members), max(members)
        # Outer rectangle
        ax.add_patch(plt.Rectangle((mn - 0.5, mn - 0.5), mx - mn + 1, mx - mn + 1,
                                   fill=False, edgecolor="red", lw=1.5))
        ax.text(mx + 0.6, mn + 0.5, name, color="red", fontsize=9,
                fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def r1_group_check_csv(M_reset: np.ndarray, M_noreset: np.ndarray, out_path: Path) -> None:
    """For each R1 group and each within-group pair, log multiset overlap
    in both modes."""

    with open(out_path, "w") as f:
        f.write("group,cnot_i,cnot_j,overlap_reset,overlap_noreset\n")
        for name, members in R1_AMBIGUITY_GROUPS.items():
            for i in members:
                for j in members:
                    if i >= j:
                        continue
                    f.write(
                        f"{name},{i},{j},{M_reset[i, j]:.4f},{M_noreset[i, j]:.4f}\n"
                    )


def main() -> None:
    out_dir = ensure_out_dir()
    t = load_table()

    for mode in ["reset", "noreset"]:
        sub_C = select(t, cat="C", mode=mode)
        per_cnot = per_cnot_syndromes(sub_C)

        per_cnot_unique_csv(
            per_cnot, out_dir / f"per_cnot_unique_syndromes_{mode}.csv", mode
        )
        print(f"Wrote per_cnot_unique_syndromes_{mode}.csv")

        bitmap_path = out_dir / f"per_cnot_pauli_bitmap_{mode}.png"
        pauli_bitmap_plot(
            sub_C, bitmap_path,
            f"Per-CNOT 15-Pauli syndrome bitmaps (rows=Paulis, cols=24 bits) — mode={mode}",
        )
        print(f"Wrote {bitmap_path}")

    # Multiset overlap matrices (both modes) for R1-group cross-check.
    sub_C_reset = select(t, cat="C", mode="reset")
    sub_C_noreset = select(t, cat="C", mode="noreset")
    M_reset = multiset_overlap_matrix(sub_C_reset)
    M_noreset = multiset_overlap_matrix(sub_C_noreset)

    plot_overlap_matrix(M_reset, out_dir / "cnot_multiset_overlap_reset.png",
                        "CNOT-pair 15-Pauli multiset overlap — mode=reset (R1 groups outlined)")
    plot_overlap_matrix(M_noreset, out_dir / "cnot_multiset_overlap_noreset.png",
                        "CNOT-pair 15-Pauli multiset overlap — mode=noreset (R1 groups outlined)")

    r1_group_check_csv(M_reset, M_noreset, out_dir / "r1_group_multiset_check.csv")
    print(f"Wrote r1_group_multiset_check.csv")

    print("\n=== R1 ambiguity group multiset overlaps ===")
    for name, members in R1_AMBIGUITY_GROUPS.items():
        for i in members:
            for j in members:
                if i >= j:
                    continue
                print(
                    f"  {name}  ({i:2d},{j:2d})  reset={M_reset[i, j]:.3f}  "
                    f"noreset={M_noreset[i, j]:.3f}"
                )


if __name__ == "__main__":
    main()
