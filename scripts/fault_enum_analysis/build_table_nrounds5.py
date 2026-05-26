"""
Rebuild the deterministic single-fault enumeration table at n_rounds = 5
instead of 3.  Saves to a sibling directory so the n_rounds=3 baseline
is untouched.

Standalone: copies the relevant logic from scripts/build_fault_enumeration_table.py
so we don't need to make scripts/ importable.

Output:
    data/analysis/fault_enumeration_nrounds5/fault_enumeration_table.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.postprocess import reshape_mid_measure_syndrome
from src.simulator import make_repeated_stabilizer_qnode
from src.surface_code_layout import data_qubits, enumerate_stabilizer_cnots


N_ROUNDS = 5
N_STABILIZERS = 8
SHOTS = 1

SINGLE_QUBIT_PAULIS = ["X", "Y", "Z"]
TWO_QUBIT_PAULIS = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]
MODE_SPECS = [("reset", True), ("noreset", False)]


def make_two_qubit_post_cnot_fault(control, target, two_qubit_pauli, round_idx=0):
    p_c, p_t = two_qubit_pauli[0], two_qubit_pauli[1]
    error_wires, error_types = [], []
    if p_c != "I":
        error_wires.append(int(control)); error_types.append(p_c)
    if p_t != "I":
        error_wires.append(int(target)); error_types.append(p_t)
    return [{"round": int(round_idx),
             "control": int(control), "target": int(target),
             "error_wires": error_wires, "error_types": error_types}]


def enumerate_cases():
    for d in data_qubits:
        for p in SINGLE_QUBIT_PAULIS:
            yield f"A:data{d}:{p}", [], [p], [int(d)]
    cnot_locations = enumerate_stabilizer_cnots()
    for k, loc in enumerate(cnot_locations):
        for p2 in TWO_QUBIT_PAULIS:
            fs = make_two_qubit_post_cnot_fault(loc["control"], loc["target"], p2, 0)
            yield f"C:cnot{k:02d}:{p2}", fs, [], []


def simulate_case(qnode, fault_schedule, init_errs, init_wires):
    raw = qnode(fault_schedule=fault_schedule,
                initial_error_list=init_errs,
                initial_error_wires=init_wires)
    syn = reshape_mid_measure_syndrome(raw, n_rounds=N_ROUNDS,
                                       shots=SHOTS, n_stabilizers=N_STABILIZERS)
    return syn[0].astype(np.int8)


def main():
    out_dir = PROJECT_ROOT / "data" / "analysis" / f"fault_enumeration_nrounds{N_ROUNDS}"
    ensure_dir(out_dir)

    cases = list(enumerate_cases())
    n_cases = len(cases)
    n_records = n_cases * len(MODE_SPECS)
    print(f"n_rounds={N_ROUNDS}  n_cases={n_cases}  n_records={n_records}")

    qnodes = {
        label: make_repeated_stabilizer_qnode(
            n_rounds=N_ROUNDS, shots=SHOTS, mid_measure=True,
            reset_after_measure=reset_flag,
        )
        for label, reset_flag in MODE_SPECS
    }

    rows_error, rows_mode = [], []
    syndromes = np.zeros((n_records, N_ROUNDS, N_STABILIZERS), dtype=np.int8)

    idx = 0
    for case_i, (descriptor, fs, init_errs, init_wires) in enumerate(cases):
        for mode_label, _ in MODE_SPECS:
            syn = simulate_case(qnodes[mode_label], fs, init_errs, init_wires)
            rows_error.append(descriptor); rows_mode.append(mode_label)
            syndromes[idx] = syn
            idx += 1
        if (case_i + 1) % 50 == 0 or case_i + 1 == n_cases:
            print(f"  processed {case_i + 1:4d} / {n_cases}")

    npz_path = out_dir / "fault_enumeration_table.npz"
    np.savez(
        npz_path,
        error=np.array(rows_error), mode=np.array(rows_mode),
        syndrome=syndromes,
        n_rounds=np.int32(N_ROUNDS), n_stabilizers=np.int32(N_STABILIZERS),
        stab_labels=np.array(["Z0","Z1","Z2","Z3","X0","X1","X2","X3"]),
    )
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
