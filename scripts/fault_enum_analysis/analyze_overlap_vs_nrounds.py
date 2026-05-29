"""
Overlap-vs-n_rounds scan.

Uses the n_rounds=3 (baseline) and n_rounds=5 (rebuilt) enumeration tables
to plot R1-ambiguity-group detection-event multiset overlap as a function
of how many rounds are folded into the syndrome string.

For each n_rounds budget T in {1, 2, 3, 4, 5} (where T <= the underlying
table's n_rounds), we slice the first T rounds of detection-event syndromes,
build the T*8-bit multiset for each CNOT, and compute the R1-pair overlap.

Source tables:
    data/analysis/fault_enumeration/fault_enumeration_table.npz   (n_rounds=3)
    data/analysis/fault_enumeration_nrounds5/fault_enumeration_table.npz

Output (data/analysis/10_fault_enumeration_patterns/):
    r1_group_overlap_vs_nrounds.csv
    r1_group_overlap_vs_nrounds.png
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
    N_CNOT,
    OUT_DIR,
    PROJECT_ROOT,
    ensure_out_dir,
    load_table,
    select,
)

R1_AMBIGUITY_GROUPS = {
    "G1": [6, 7],
    "G2": [14, 15, 17],
    "G3": [18, 19, 20],
    "G4": [22, 23],
}

NROUNDS5_PATH = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration_nrounds5" / "fault_enumeration_table.npz"


def raw_to_detection(syn: np.ndarray) -> np.ndarray:
    det = np.empty_like(syn)
    det[:, 0, :] = syn[:, 0, :]
    det[:, 1:, :] = syn[:, 1:, :] ^ syn[:, :-1, :]
    return det


def pack_bits(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape[:-1], dtype=np.uint64)
    for j in range(arr.shape[-1]):
        out = (out << np.uint64(1)) | arr[..., j].astype(np.uint64)
    return out


def per_cnot_multisets(det: np.ndarray, qubit_or_cnot: np.ndarray, T: int) -> dict[int, Counter]:
    """{cnot_id: Counter(packed key over first T rounds -> count)}."""

    sliced = det[:, :T, :].reshape(det.shape[0], -1)
    keys = pack_bits(sliced)
    out: dict[int, Counter] = {k: Counter() for k in range(N_CNOT)}
    for k, key in zip(qubit_or_cnot, keys):
        out[int(k)][int(key)] += 1
    return out


def overlap(a: Counter, b: Counter) -> float:
    inter = 0
    for x, c in a.items():
        inter += min(c, b.get(x, 0))
    return inter / 15.0


def collect_overlaps(table_path: Path, T_max: int, mode: str = "reset") -> dict:
    t = load_table(table_path)
    sub_C = select(t, cat="C", mode=mode)
    det = raw_to_detection(sub_C["syndrome"])

    rows = []
    for T in range(1, T_max + 1):
        ms = per_cnot_multisets(det, sub_C["qubit_or_cnot"], T)
        for name, members in R1_AMBIGUITY_GROUPS.items():
            for i in members:
                for j in members:
                    if i >= j:
                        continue
                    o = overlap(ms[i], ms[j])
                    rows.append((T, name, i, j, o))
    return rows


def main() -> None:
    out_dir = ensure_out_dir()

    rows_3 = collect_overlaps(
        PROJECT_ROOT / "data" / "analysis" / "fault_enumeration" / "fault_enumeration_table.npz",
        T_max=3,
    )
    rows_5 = collect_overlaps(NROUNDS5_PATH, T_max=5)

    # Prefer n_rounds=5 results for T=1..5; n_rounds=3 used only as cross-check
    # for T=1,2,3 (should match exactly).
    by_key_3 = {(T, name, i, j): o for T, name, i, j, o in rows_3}
    by_key_5 = {(T, name, i, j): o for T, name, i, j, o in rows_5}

    cross_check_pass = True
    for (T, name, i, j), o3 in by_key_3.items():
        o5 = by_key_5.get((T, name, i, j))
        if o5 is None or abs(o5 - o3) > 1e-9:
            cross_check_pass = False
            print(f"  MISMATCH T={T} {name} ({i},{j}): n3={o3:.4f} n5={o5}")

    print(f"\nT=1..3 cross-check (n_rounds=3 vs first-3-rounds of n_rounds=5): "
          f"{'PASS' if cross_check_pass else 'FAIL'}")

    # CSV.
    csv_path = out_dir / "r1_group_overlap_vs_nrounds.csv"
    with open(csv_path, "w") as f:
        f.write("T,group,cnot_i,cnot_j,overlap\n")
        for T, name, i, j, o in rows_5:
            f.write(f"{T},{name},{i},{j},{o:.4f}\n")
    print(f"Wrote {csv_path}")

    # Plot — one line per (group, pair).
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"G1": "tab:blue", "G2": "tab:orange", "G3": "tab:green", "G4": "tab:red"}
    seen_label = set()
    for name, members in R1_AMBIGUITY_GROUPS.items():
        for i in members:
            for j in members:
                if i >= j:
                    continue
                ys = [by_key_5[(T, name, i, j)] for T in range(1, 6)]
                label = name if name not in seen_label else None
                seen_label.add(name)
                ax.plot(range(1, 6), ys, "-o", color=colors[name], alpha=0.6, label=label,
                        markersize=6)

    ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4, label="R1 ceiling (T=1)")
    ax.set_xlabel("n_rounds T included in detection-event syndrome")
    ax.set_ylabel("multiset overlap / 15 (lower = more distinguishable)")
    ax.set_title("R1 ambiguity-group multiset overlap shrinks with n_rounds (reset mode)")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    out_png = out_dir / "r1_group_overlap_vs_nrounds.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_png}")

    print("\n=== R1 overlap as function of n_rounds (reset) ===")
    print(f"{'T':>2s} | " + " | ".join(f"{name}" for name in R1_AMBIGUITY_GROUPS) + " | mean")
    for T in range(1, 6):
        per_group_mean = []
        for name, members in R1_AMBIGUITY_GROUPS.items():
            vals = [by_key_5[(T, name, i, j)] for i in members for j in members if i < j]
            per_group_mean.append(np.mean(vals))
        overall = np.mean(per_group_mean)
        print(f"{T:>2d} | " + " | ".join(f"{x:.3f}" for x in per_group_mean) + f" | {overall:.3f}")


if __name__ == "__main__":
    main()
