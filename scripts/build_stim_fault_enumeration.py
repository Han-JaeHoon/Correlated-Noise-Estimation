"""Single-round single-fault syndrome enumeration on the realistic Stim circuit.

For one stabilizer-measurement round (no repetition), enumerate every elementary
fault and record the syndrome (raw ancilla flip pattern) it produces:

* **data single-Pauli**: X / Y / Z on each data qubit, injected at the start of
  the round (``pre``);
* **CNOT two-Pauli**: each of the 15 non-identity two-qubit Paulis on
  ``(control, target)`` immediately after each directed CNOT (``post_cx``), so it
  propagates through the remaining ticks of the same round (hook error).

Syndrome extraction is the repo's validated differential method (see
``build_viz_data.py``): sample the noiseless and the fault-injected circuit with
the **same seed** and XOR the raw ancilla records. Pauli faults consume no
randomness, so the streams stay aligned and the XOR isolates exactly the
stabilizers the fault flips. With a single round reset/no-reset are identical, so
only one table is produced per distance.

Outputs (per distance, under data/analysis/stim_fault_enumeration/d{d}/):
* enumeration.csv  — one human-readable row per fault case
* syndromes.npz    — syndrome matrix (N x d^2-1) int8 + metadata arrays
* summary.txt      — case counts, unique syndromes, silent + degeneracy stats
"""
from __future__ import annotations

import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend_stim.surface_code import RotatedSurfaceCode  # noqa: E402

ROUNDS = 2          # round 0 projects |0...0> -> |0_L>; round 1 carries the fault
INJECT_ROUND = 1    # the (single) fault round; round 0 is fault-free projection
SEED = 12345

# 15 non-identity two-qubit Paulis (control, target), II excluded.
TWO_PAULIS = ["".join(p) for p in product("IXYZ", repeat=2) if p != ("I", "I")]


def raw_ancilla(code: RotatedSurfaceCode, injections) -> np.ndarray:
    """One seeded shot -> (ROUNDS, n_anc) raw ancilla measurements."""
    circ = code.build_circuit(rounds=ROUNDS, basis="Z", reset=True,
                              injections=injections)
    n_anc = len(code.anc_coords)
    samp = circ.compile_sampler(seed=SEED).sample(shots=1)[0]
    return samp[: n_anc * ROUNDS].reshape(ROUNDS, n_anc).astype(np.int8)


def enumerate_distance(d: int) -> dict:
    code = RotatedSurfaceCode(d)
    n_anc = len(code.anc_coords)
    anc_labels = [f"{code.anc_type[c]}{k}" for k, c in enumerate(code.anc_coords)]

    base = raw_ancilla(code, [])  # noiseless reference (same seed)

    rows = []          # dict per case
    syndromes = []     # flattened (n_anc,) per case

    def add_case(case_id, kind, location, pauli, tick, control, target,
                 stab_type, injections):
        err = raw_ancilla(code, injections)
        # syndrome = the fault round's change vs the fault-free reference (with
        # round 0 having projected into |0_L>, this is the detection-event
        # pattern: all-zero when no fault, the fault's flips otherwise).
        flip = (base ^ err)[INJECT_ROUND]  # (n_anc,)
        lit = [anc_labels[k] for k in range(n_anc) if flip[k]]
        rows.append({
            "case_id": case_id, "kind": kind, "location": location,
            "pauli": pauli, "tick": tick, "control": control, "target": target,
            "stab_type": stab_type,
            "syndrome": "".join(str(int(b)) for b in flip),
            "weight": int(flip.sum()),
            "lit_ancillas": ";".join(lit),
        })
        syndromes.append(flip)

    # (1) data single-Pauli
    for i, c in enumerate(code.data_coords):
        for p in ("X", "Y", "Z"):
            add_case(f"D{i}_{p}", "data_single", f"D{i}{tuple(c)}", p,
                     "", "", "", "",
                     [{"round": INJECT_ROUND, "pos": "pre", "coord": c, "pauli": p}])

    # (2) CNOT two-Pauli (post-CNOT, propagates through the rest of the round)
    for cid, (tick, ctrl, tgt, typ) in enumerate(code.cnot_enumeration()):
        for pp in TWO_PAULIS:
            add_case(f"C{cid}_{pp}", "cnot_two", f"C{cid}", pp,
                     tick, str(tuple(ctrl)), str(tuple(tgt)), typ,
                     [{"round": INJECT_ROUND, "pos": "post_cx", "tick": tick,
                       "control": ctrl, "target": tgt, "pauli": pp}])

    return {
        "code": code, "rows": rows,
        "syndromes": np.array(syndromes, dtype=np.int8),
        "anc_labels": anc_labels, "n_anc": n_anc,
    }


def write_outputs(d: int, res: dict, out_root: Path) -> None:
    out = out_root / f"d{d}"
    out.mkdir(parents=True, exist_ok=True)
    rows, S = res["rows"], res["syndromes"]
    fields = ["case_id", "kind", "location", "pauli", "tick", "control",
              "target", "stab_type", "syndrome", "weight", "lit_ancillas"]

    with (out / "enumeration.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(
        out / "syndromes.npz",
        syndromes=S,
        case_id=np.array([r["case_id"] for r in rows]),
        kind=np.array([r["kind"] for r in rows]),
        pauli=np.array([r["pauli"] for r in rows]),
        weight=np.array([r["weight"] for r in rows]),
        anc_labels=np.array(res["anc_labels"]),
    )

    # summary / degeneracy
    n = len(rows)
    silent = [r["case_id"] for r in rows if r["weight"] == 0]
    keys = [r["syndrome"] for r in rows]
    uniq = {}
    for k in keys:
        uniq[k] = uniq.get(k, 0) + 1
    n_unique = len(uniq)
    max_deg = max(uniq.values())
    n_data = sum(r["kind"] == "data_single" for r in rows)
    n_cnot = sum(r["kind"] == "cnot_two" for r in rows)
    uniq_data = len({r["syndrome"] for r in rows if r["kind"] == "data_single"})
    uniq_cnot = len({r["syndrome"] for r in rows if r["kind"] == "cnot_two"})

    with (out / "summary.txt").open("w") as f:
        f.write(f"Stim single-fault syndrome enumeration  d={d}\n")
        f.write(f"  round 0 = fault-free projection (|0...0> -> |0_L>), "
                f"round 1 = single fault; Z-basis, seed={SEED}\n")
        f.write(f"  ancillas (syndrome length) = {res['n_anc']}\n\n")
        f.write(f"total cases            : {n}\n")
        f.write(f"  data single-Pauli    : {n_data}  (unique syndromes {uniq_data})\n")
        f.write(f"  CNOT two-Pauli       : {n_cnot}  (unique syndromes {uniq_cnot})\n")
        f.write(f"unique syndromes (all) : {n_unique}\n")
        f.write(f"silent (all-zero) cases: {len(silent)}\n")
        f.write(f"max degeneracy class   : {max_deg}\n")
        if silent:
            f.write("\nsilent cases:\n  " + ", ".join(silent) + "\n")

    print(f"[d={d}] {n} cases -> {out}  "
          f"(unique {n_unique}, silent {len(silent)}, max-deg {max_deg})")


def main() -> int:
    out_root = Path("data/analysis/stim_fault_enumeration")
    for d in (3, 5):
        res = enumerate_distance(d)
        write_outputs(d, res, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
