# scripts/build_fault_enumeration_table.py

"""
Deterministic single-fault syndrome enumeration table.

Two fault categories are enumerated, with all other noise turned off
(deterministic 1-shot simulation):

    Category A (round-start data idle)
        At round 0, before any gate, a single Pauli (X / Y / Z) is
        inserted on one of the 9 data qubits.
        9 data * 3 Pauli = 27 cases.

    Category C (post-CNOT 2-qubit Pauli)
        Immediately after one of the 24 directed CNOTs in round 0, one
        of the 15 nontrivial 2-qubit Pauli operators is inserted.
        24 CNOTs * 15 Pauli = 360 cases.

Each case is simulated twice:
    mode = "reset"    -> ancilla measured + reset every round
    mode = "noreset"  -> ancilla measured without reset

Total records:
    (27 + 360) * 2 = 774.

For every record the saved fields are exactly:
    error     : str descriptor of the inserted fault
    mode      : "reset" or "noreset"
    syndrome  : (n_rounds, 8) int8 array of raw ancilla measurements

Output:
    data/analysis/fault_enumeration/fault_enumeration_table.npz
    data/analysis/fault_enumeration/fault_enumeration_preview.csv
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.postprocess import reshape_mid_measure_syndrome
from src.simulator import make_repeated_stabilizer_qnode
from src.surface_code_layout import data_qubits, enumerate_stabilizer_cnots


N_ROUNDS = 3
N_STABILIZERS = 8
SHOTS = 1

SINGLE_QUBIT_PAULIS = ["X", "Y", "Z"]

# Lexicographic over {I, X, Y, Z}^2 minus II.
TWO_QUBIT_PAULIS = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]

MODE_SPECS = [
    ("reset", True),
    ("noreset", False),
]


def make_two_qubit_post_cnot_fault(control, target, two_qubit_pauli, round_idx=0):
    """
    Build a fault schedule for one round-0 post-CNOT 2-qubit Pauli.

    "I" leg is omitted from error_wires/error_types.
    """

    p_control, p_target = two_qubit_pauli[0], two_qubit_pauli[1]

    error_wires = []
    error_types = []

    if p_control != "I":
        error_wires.append(int(control))
        error_types.append(p_control)

    if p_target != "I":
        error_wires.append(int(target))
        error_types.append(p_target)

    return [
        {
            "round": int(round_idx),
            "control": int(control),
            "target": int(target),
            "error_wires": error_wires,
            "error_types": error_types,
        }
    ]


def enumerate_cases():
    """
    Yield (descriptor, fault_schedule, initial_error_types, initial_error_wires).
    """

    # Category A: round-0 data idle single-qubit Pauli.
    for d in data_qubits:
        for p in SINGLE_QUBIT_PAULIS:
            descriptor = f"A:data{d}:{p}"
            yield descriptor, [], [p], [int(d)]

    # Category C: round-0 post-CNOT 2-qubit Pauli.
    cnot_locations = enumerate_stabilizer_cnots()
    for k, loc in enumerate(cnot_locations):
        for p2 in TWO_QUBIT_PAULIS:
            descriptor = f"C:cnot{k:02d}:{p2}"
            fs = make_two_qubit_post_cnot_fault(
                control=loc["control"],
                target=loc["target"],
                two_qubit_pauli=p2,
                round_idx=0,
            )
            yield descriptor, fs, [], []


def simulate_case(qnode, fault_schedule, init_errs, init_wires):
    """
    Return (N_ROUNDS, 8) int8 syndrome.
    """

    raw = qnode(
        fault_schedule=fault_schedule,
        initial_error_list=init_errs,
        initial_error_wires=init_wires,
    )

    syndrome = reshape_mid_measure_syndrome(
        raw, n_rounds=N_ROUNDS, shots=SHOTS, n_stabilizers=N_STABILIZERS
    )

    return syndrome[0].astype(np.int8)


def main():
    out_dir = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration"
    ensure_dir(out_dir)

    cases = list(enumerate_cases())
    n_cases = len(cases)
    n_records = n_cases * len(MODE_SPECS)

    print(f"n_rounds   : {N_ROUNDS}")
    print(f"n_cases    : {n_cases}  (expected 387 = 27 A + 360 C)")
    print(f"n_records  : {n_records}  (cases * 2 modes)")
    print()

    qnodes = {
        label: make_repeated_stabilizer_qnode(
            n_rounds=N_ROUNDS,
            shots=SHOTS,
            mid_measure=True,
            reset_after_measure=reset_flag,
        )
        for label, reset_flag in MODE_SPECS
    }

    rows_error = []
    rows_mode = []
    syndromes = np.zeros((n_records, N_ROUNDS, N_STABILIZERS), dtype=np.int8)

    idx = 0
    for case_i, (descriptor, fs, init_errs, init_wires) in enumerate(cases):
        for mode_label, _ in MODE_SPECS:
            qnode = qnodes[mode_label]
            syn = simulate_case(qnode, fs, init_errs, init_wires)

            rows_error.append(descriptor)
            rows_mode.append(mode_label)
            syndromes[idx] = syn

            idx += 1

        if (case_i + 1) % 50 == 0 or case_i + 1 == n_cases:
            print(f"  processed {case_i + 1:4d} / {n_cases} cases")

    error_arr = np.array(rows_error)
    mode_arr = np.array(rows_mode)

    npz_path = out_dir / "fault_enumeration_table.npz"
    np.savez(
        npz_path,
        error=error_arr,
        mode=mode_arr,
        syndrome=syndromes,
        n_rounds=np.int32(N_ROUNDS),
        n_stabilizers=np.int32(N_STABILIZERS),
        stab_labels=np.array(["Z0", "Z1", "Z2", "Z3", "X0", "X1", "X2", "X3"]),
    )
    print(f"\nSaved npz   : {npz_path}")

    csv_path = out_dir / "fault_enumeration_preview.csv"
    with open(csv_path, "w") as f:
        f.write("error,mode")
        for r in range(N_ROUNDS):
            f.write(f",syndrome_r{r}")
        f.write("\n")
        for e, m, s in zip(error_arr, mode_arr, syndromes):
            f.write(f"{e},{m}")
            for r in range(N_ROUNDS):
                bits = "".join(str(int(b)) for b in s[r])
                f.write(f",{bits}")
            f.write("\n")
    print(f"Saved csv   : {csv_path}")

    # Quick sanity stats.
    print()
    print("Sanity")
    print("------")
    n_zero = np.sum(np.all(syndromes.reshape(n_records, -1) == 0, axis=1))
    print(f"  records with all-zero syndrome : {n_zero} / {n_records}")
    cat_a_mask = np.array([e.startswith("A:") for e in error_arr])
    cat_c_mask = np.array([e.startswith("C:") for e in error_arr])
    print(f"  Category A records              : {int(cat_a_mask.sum())}")
    print(f"  Category C records              : {int(cat_c_mask.sum())}")


if __name__ == "__main__":
    main()
