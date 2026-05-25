"""
Syndrome collision / similarity analysis on the 774-record fault enumeration.

What this script answers:
  (1) For each (category, mode), how many distinct 24-bit syndromes do the
      single-fault cases collapse into? Which faults are bit-identical?
  (2) Pairwise Hamming distance distribution across all syndromes within a
      (category, mode). What is the median nearest-neighbour distance?
  (3) Heatmap of Hamming distances sorted by category/CNOT/Pauli, so that
      block structure (one CNOT's 15 Paulis vs another CNOT) is visible.

Outputs (data/analysis/10_fault_enumeration_patterns/):
    collision_summary.csv
    collision_groups_C_reset.csv
    collision_groups_C_noreset.csv
    collision_groups_A_reset.csv
    collision_groups_A_noreset.csv
    hamming_distance_hist.png
    hamming_distance_heatmap_C_reset.png
    hamming_distance_heatmap_C_noreset.png
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fault_enum_analysis._common import (  # noqa: E402
    ensure_out_dir,
    flatten_syndrome,
    hamming_pairwise,
    load_table,
    packbits24,
    select,
)


def group_by_syndrome(sub: dict) -> dict[int, list[str]]:
    sf = flatten_syndrome(sub["syndrome"])
    keys = packbits24(sf)
    groups: dict[int, list[str]] = defaultdict(list)
    for k, e in zip(keys, sub["error"]):
        groups[int(k)].append(str(e))
    return groups


def write_groups_csv(groups: dict[int, list[str]], path: Path) -> None:
    items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    with open(path, "w") as f:
        f.write("syndrome_key_uint24,size,members\n")
        for key, members in items:
            f.write(f"{key},{len(members)},{';'.join(sorted(members))}\n")


def collision_summary_row(sub: dict, label: str) -> dict:
    sf = flatten_syndrome(sub["syndrome"])
    keys = packbits24(sf)
    uniq, counts = np.unique(keys, return_counts=True)
    silent = int((keys == 0).sum())
    return dict(
        label=label,
        n_cases=int(len(keys)),
        n_unique=int(len(uniq)),
        n_silent=silent,
        max_collision_size=int(counts.max()),
        n_singleton_classes=int((counts == 1).sum()),
        n_multi_classes=int((counts > 1).sum()),
        compression_ratio=float(len(uniq)) / float(len(keys)),
    )


def heatmap_hamming(sub: dict, out_path: Path, title: str) -> None:
    sf = flatten_syndrome(sub["syndrome"])
    # Sort by (cnot, pauli) so the 15-Pauli block per CNOT is visible.
    order = np.lexsort((sub["pauli"], sub["qubit_or_cnot"]))
    sf_sorted = sf[order]
    D = hamming_pairwise(sf_sorted)

    n = D.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(D, cmap="magma", origin="lower", vmin=0, vmax=24)
    ax.set_title(title)
    ax.set_xlabel("case index (sorted by CNOT, then Pauli)")
    ax.set_ylabel("case index")
    fig.colorbar(im, ax=ax, label="Hamming distance (24-bit syndrome)")

    # Draw faint lines every 15 (CNOT block boundaries) for C-category.
    if "C" in title:
        for k in range(15, n, 15):
            ax.axvline(k - 0.5, color="cyan", lw=0.3, alpha=0.5)
            ax.axhline(k - 0.5, color="cyan", lw=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def hamming_hist(t: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, mode in zip(axes, ["reset", "noreset"]):
        for cat, color in [("A", "tab:blue"), ("C", "tab:orange")]:
            sub = select(t, cat=cat, mode=mode)
            sf = flatten_syndrome(sub["syndrome"])
            D = hamming_pairwise(sf)
            iu = np.triu_indices_from(D, k=1)
            d_vals = D[iu]
            ax.hist(
                d_vals,
                bins=np.arange(0, 25) - 0.5,
                histtype="step",
                lw=2,
                color=color,
                label=f"Cat {cat} (n_pairs={len(d_vals)})",
                density=True,
            )
        ax.set_title(f"mode = {mode}")
        ax.set_xlabel("Hamming distance between syndrome pairs (bits / 24)")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Pairwise Hamming distance distribution across single-fault syndromes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = ensure_out_dir()
    t = load_table()

    summary_rows = []
    for cat in ["A", "C"]:
        for mode in ["reset", "noreset"]:
            sub = select(t, cat=cat, mode=mode)
            label = f"{cat}_{mode}"
            summary_rows.append(collision_summary_row(sub, label))

            groups = group_by_syndrome(sub)
            write_groups_csv(groups, out_dir / f"collision_groups_{label}.csv")

    # Write summary CSV.
    summary_path = out_dir / "collision_summary.csv"
    with open(summary_path, "w") as f:
        cols = list(summary_rows[0].keys())
        f.write(",".join(cols) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"Wrote {summary_path}")

    # Histogram across cat/mode.
    hist_path = out_dir / "hamming_distance_hist.png"
    hamming_hist(t, hist_path)
    print(f"Wrote {hist_path}")

    # Heatmaps for the larger C category (more interesting).
    for mode in ["reset", "noreset"]:
        sub = select(t, mode=mode, cat="C")
        title = f"Pairwise Hamming distance — Category C, mode={mode}"
        path = out_dir / f"hamming_distance_heatmap_C_{mode}.png"
        heatmap_hamming(sub, path, title)
        print(f"Wrote {path}")

    print("\n=== Collision summary ===")
    for r in summary_rows:
        print(
            f"  {r['label']:>11s} : cases={r['n_cases']:3d}  unique={r['n_unique']:3d}"
            f"  silent={r['n_silent']:2d}  max_grp={r['max_collision_size']:2d}"
            f"  ratio={r['compression_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
