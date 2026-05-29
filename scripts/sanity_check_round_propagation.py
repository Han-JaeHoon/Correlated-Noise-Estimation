"""
Regression test for the symbolic Pauli-frame propagator in
`src/round_propagation.py`.

Three checks:

  A. Single-round single-fault syndrome lookup (24×15) matches the PennyLane
     circuit's lookup (`src/ceiling.compute_single_round_lookup`) bit-for-bit.

  B. With no faults, every single-data-qubit Pauli (3 × 9 = 27) is preserved
     in the data part of the frame across one round.

  C. Two-round end-to-end: fault at round 0, idle round 1, single-shot QNode
     with shots=1. Symbolic two-round syndrome stream must equal the PennyLane
     syndrome stream for all 24 × 15 = 360 (CNOT, Pauli) cases.

This file is the runtime safety net for the LookupDecoder and the
window-by-window sequence runner — both depend on the propagator being
exactly right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.ceiling import compute_single_round_lookup
from src.pauli_frame import PauliFrame
from src.postprocess import reshape_mid_measure_syndrome
from src.round_propagation import (
    PAULI_PAIRS_15,
    build_single_fault_syndrome_lookup,
    propagate_round,
    propagate_round_with_syndrome,
)
from src.simulator import make_repeated_stabilizer_qnode
from src.stochastic_faults import _pauli_pair_to_error_lists
from src.surface_code_layout import enumerate_stabilizer_cnots, n_data, n_qubits


def check_A_single_round_lookup() -> bool:
    print("=" * 72)
    print("A. Single-round 24×15 syndrome lookup: PennyLane vs symbolic")
    pl = compute_single_round_lookup(reset=True)
    sym = build_single_fault_syndrome_lookup(reset=True)
    ok = bool(np.array_equal(pl, sym))
    print(f"   shapes: PL={pl.shape}, sym={sym.shape}")
    if ok:
        print("   PASS  (360/360 entries match)")
    else:
        diff = (pl != sym)
        n = int(diff.sum())
        print(f"   FAIL  ({n}/{pl.size} entries differ)")
        for k, alpha in zip(*np.where(diff)):
            print(f"     CNOT {k}, Pauli {alpha}: PL={pl[k,alpha]:08b} sym={sym[k,alpha]:08b}")
            if k >= 3:
                break
    return ok


def check_B_data_preservation() -> bool:
    print("=" * 72)
    print("B. Empty round preserves every single-data-qubit Pauli")
    n_pass = n_fail = 0
    for q in range(n_data):
        for pauli in ("X", "Y", "Z"):
            f = PauliFrame(n=n_qubits)
            f.apply_pauli(pauli, q)
            x0, z0 = f.data_xz()
            post = propagate_round(f, [], reset_after_measure=True)
            x1, z1 = post.data_xz()
            if np.array_equal(x0, x1) and np.array_equal(z0, z1):
                n_pass += 1
            else:
                n_fail += 1
                print(f"   FAIL: {pauli}_{q} not preserved")
    if n_fail == 0:
        print(f"   PASS  ({n_pass}/27 single-data Paulis preserved)")
    else:
        print(f"   FAIL  ({n_fail}/27)")
    return n_fail == 0


def check_C_two_round_end_to_end() -> bool:
    print("=" * 72)
    print("C. Two-round end-to-end (fault r0, idle r1) — PennyLane vs symbolic")
    cnots = enumerate_stabilizer_cnots()
    qnode = make_repeated_stabilizer_qnode(
        n_rounds=2, shots=1, mid_measure=True, reset_after_measure=True,
        return_probs=False,
    )
    n_pass = n_fail = 0
    fail_cases = []
    for k, loc in enumerate(cnots):
        c, t = int(loc["control"]), int(loc["target"])
        for alpha, pair in enumerate(PAULI_PAIRS_15):
            types, wires = _pauli_pair_to_error_lists(pair, c, t)
            schedule = [{
                "round": 0, "control": c, "target": t,
                "error_wires": wires, "error_types": types,
            }]
            raw = qnode(fault_schedule=schedule)
            pl_syn = np.asarray(
                reshape_mid_measure_syndrome(raw, n_rounds=2, shots=1, n_stabilizers=8)[0],
                dtype=np.uint8,
            )

            empty = PauliFrame(n=n_qubits)
            f1, s1 = propagate_round_with_syndrome(empty, [schedule[0]], reset_after_measure=True)
            f2, s2 = propagate_round_with_syndrome(f1, [], reset_after_measure=True)
            sym_syn = np.stack([s1, s2], axis=0)

            if np.array_equal(pl_syn, sym_syn):
                n_pass += 1
            else:
                n_fail += 1
                if len(fail_cases) < 5:
                    fail_cases.append((k, alpha, pair, pl_syn.tolist(), sym_syn.tolist()))
    if n_fail == 0:
        print(f"   PASS  ({n_pass}/360 two-round (cnot, pauli) cases match)")
    else:
        print(f"   FAIL  ({n_fail}/360)")
        for c in fail_cases:
            print(f"     {c}")
    return n_fail == 0


def main() -> None:
    print("Pauli-frame round propagator — regression suite")
    print()
    a = check_A_single_round_lookup()
    b = check_B_data_preservation()
    c = check_C_two_round_end_to_end()
    print("=" * 72)
    print(f"OVERALL: {'PASS' if (a and b and c) else 'FAIL'}")
    sys.exit(0 if (a and b and c) else 1)


if __name__ == "__main__":
    main()
