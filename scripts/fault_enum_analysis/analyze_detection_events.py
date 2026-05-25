"""
Detection-event view of the 774-record fault enumeration.

The raw table stores per-round ancilla measurements.  §13's R1 framework
operates on *detection events* — per-round XOR of consecutive raw rounds,
with round 0's detection event taken vs. a zero baseline.  This script
recomputes the collision / multiset analyses on detection events, which is
the correct frame for cross-checking §13's ambiguity-group claim.

Two views of "detection event" are reported:

  (a) round_0_only  : 8-bit detection event of round 0.  Bit-for-bit equal
      to the §13 R1 lookup table.  This is what defines the R1 ambiguity
      groups.

  (b) all_rounds_det : (3, 8) detection event with the convention
        det[0] = raw[0]
        det[t] = raw[t] XOR raw[t-1], t >= 1
      i.e. 24 bits per case.  The round-0 slice matches (a); rounds 1, 2
      reflect the frame propagation downstream of the round-0 fault.

We then re-derive
  - per-CNOT round-0 multiset (8-bit)
  - pair-wise multiset overlap matrix
  - R1 group cross-check (should be 1.000 for all G1..G4 pairs by §13)
  - 24-bit (a) all-rounds-detection multiset overlap, to show how much extra
    discrimination the frame propagation gives us.

Outputs (data/analysis/10_fault_enumeration_patterns/):
    detection_round0_multisets.csv
    detection_round0_overlap_reset.png
    detection_round0_overlap_noreset.png
    detection_allrounds_overlap_reset.png
    detection_allrounds_overlap_noreset.png
    r1_group_detection_check.csv
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
    N_ROUNDS,
    N_STAB,
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


def raw_to_detection(syn: np.ndarray) -> np.ndarray:
    """(N, 3, 8) int8 raw → (N, 3, 8) int8 detection events.

    det[0] = raw[0],  det[t] = raw[t] XOR raw[t-1]  for t in {1, 2}.
    """

    det = np.empty_like(syn)
    det[:, 0, :] = syn[:, 0, :]
    det[:, 1:, :] = syn[:, 1:, :] ^ syn[:, :-1, :]
    return det


def pack_bits(arr: np.ndarray) -> np.ndarray:
    """(..., n_bits) bits -> (...,) uint64 packed key."""

    out = np.zeros(arr.shape[:-1], dtype=np.uint64)
    for j in range(arr.shape[-1]):
        out = (out << np.uint64(1)) | arr[..., j].astype(np.uint64)
    return out


def per_cnot_multiset_round0(sub_C: dict) -> dict[int, Counter]:
    """{cnot_id: Counter(8-bit key -> count over the 15 Paulis)}."""

    det = raw_to_detection(sub_C["syndrome"])
    round0 = det[:, 0, :]  # (n, 8)
    keys = pack_bits(round0)
    out: dict[int, Counter] = {k: Counter() for k in range(N_CNOT)}
    for k, key in zip(sub_C["qubit_or_cnot"], keys):
        out[int(k)][int(key)] += 1
    return out


def per_cnot_multiset_24bit(sub_C: dict) -> dict[int, Counter]:
    """{cnot_id: Counter(24-bit detection-event key -> count over the 15 Paulis)}."""

    det = raw_to_detection(sub_C["syndrome"])
    flat = det.reshape(det.shape[0], -1)  # (n, 24)
    keys = pack_bits(flat)
    out: dict[int, Counter] = {k: Counter() for k in range(N_CNOT)}
    for k, key in zip(sub_C["qubit_or_cnot"], keys):
        out[int(k)][int(key)] += 1
    return out


def overlap_matrix(multisets: dict[int, Counter]) -> np.ndarray:
    M = np.zeros((N_CNOT, N_CNOT), dtype=np.float32)
    for i in range(N_CNOT):
        for j in range(N_CNOT):
            inter = 0
            ci = multisets[i]
            cj = multisets[j]
            for x, c in ci.items():
                inter += min(c, cj.get(x, 0))
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

    for name, members in R1_AMBIGUITY_GROUPS.items():
        mn, mx = min(members), max(members)
        ax.add_patch(plt.Rectangle((mn - 0.5, mn - 0.5), mx - mn + 1, mx - mn + 1,
                                   fill=False, edgecolor="red", lw=1.6))
        ax.text(mx + 0.6, mn + 0.5, name, color="red", fontsize=9,
                fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_round0_multisets_csv(ms_reset: dict, ms_noreset: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        f.write("mode,cnot_id,multiset\n")
        for label, ms in [("reset", ms_reset), ("noreset", ms_noreset)]:
            for k in range(N_CNOT):
                items = sorted(ms[k].items())
                items_str = ";".join(f"{key}:{cnt}" for key, cnt in items)
                f.write(f"{label},{k},{items_str}\n")


def write_r1_group_check_csv(ms_round0_reset, ms_round0_noreset,
                             ms_24_reset, ms_24_noreset, out_path) -> None:
    with open(out_path, "w") as f:
        f.write("group,cnot_i,cnot_j,round0_overlap_reset,round0_overlap_noreset,"
                "all24_overlap_reset,all24_overlap_noreset\n")
        for name, members in R1_AMBIGUITY_GROUPS.items():
            for i in members:
                for j in members:
                    if i >= j:
                        continue
                    o0r = _ms_overlap(ms_round0_reset[i], ms_round0_reset[j])
                    o0n = _ms_overlap(ms_round0_noreset[i], ms_round0_noreset[j])
                    o24r = _ms_overlap(ms_24_reset[i], ms_24_reset[j])
                    o24n = _ms_overlap(ms_24_noreset[i], ms_24_noreset[j])
                    f.write(
                        f"{name},{i},{j},{o0r:.4f},{o0n:.4f},{o24r:.4f},{o24n:.4f}\n"
                    )


def _ms_overlap(a: Counter, b: Counter) -> float:
    inter = 0
    for x, c in a.items():
        inter += min(c, b.get(x, 0))
    return inter / 15.0


def main() -> None:
    out_dir = ensure_out_dir()
    t = load_table()

    ms_round0 = {}
    ms_24 = {}
    M_round0 = {}
    M_24 = {}
    for mode in ["reset", "noreset"]:
        sub_C = select(t, cat="C", mode=mode)
        ms_round0[mode] = per_cnot_multiset_round0(sub_C)
        ms_24[mode] = per_cnot_multiset_24bit(sub_C)
        M_round0[mode] = overlap_matrix(ms_round0[mode])
        M_24[mode] = overlap_matrix(ms_24[mode])

        plot_overlap_matrix(
            M_round0[mode], out_dir / f"detection_round0_overlap_{mode}.png",
            f"Round-0 detection-event multiset overlap (8-bit) — mode={mode}",
        )
        plot_overlap_matrix(
            M_24[mode], out_dir / f"detection_allrounds_overlap_{mode}.png",
            f"All-rounds detection-event multiset overlap (24-bit) — mode={mode}",
        )
        print(f"Wrote detection_round0_overlap_{mode}.png and detection_allrounds_overlap_{mode}.png")

    write_round0_multisets_csv(
        ms_round0["reset"], ms_round0["noreset"],
        out_dir / "detection_round0_multisets.csv",
    )
    print("Wrote detection_round0_multisets.csv")

    write_r1_group_check_csv(
        ms_round0["reset"], ms_round0["noreset"],
        ms_24["reset"], ms_24["noreset"],
        out_dir / "r1_group_detection_check.csv",
    )
    print("Wrote r1_group_detection_check.csv")

    print("\n=== R1 group overlap on detection events ===")
    for name, members in R1_AMBIGUITY_GROUPS.items():
        for i in members:
            for j in members:
                if i >= j:
                    continue
                o0 = M_round0["reset"][i, j]
                o24 = M_24["reset"][i, j]
                print(
                    f"  {name}  ({i:2d},{j:2d})  round0={o0:.3f}  all_24={o24:.3f}  (reset mode)"
                )


if __name__ == "__main__":
    main()
