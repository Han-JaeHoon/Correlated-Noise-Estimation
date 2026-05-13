# src/stabilizer_circuit.py

import pennylane as qml

from .fault_schedule import get_faults_for_round
from .surface_code_layout import (
    Z_stabilizers,
    X_stabilizers,
    anc_z,
    anc_x,
)


def apply_pauli(pauli_type, wire):
    """
    Apply a Pauli operator to a wire.
    """

    if pauli_type == "X":
        qml.PauliX(wires=wire)
    elif pauli_type == "Y":
        qml.PauliY(wires=wire)
    elif pauli_type == "Z":
        qml.PauliZ(wires=wire)
    else:
        raise ValueError(f"Unknown Pauli type: {pauli_type}")


def apply_pauli_error(error_types, error_wires):
    """
    Apply multiple Pauli errors.

    Example:
        error_types = ["X", "Z"]
        error_wires = [0, 9]

        applies X_0 Z_9.
    """

    if len(error_types) != len(error_wires):
        raise ValueError(
            "error_types and error_wires must have the same length. "
            f"Got {len(error_types)} and {len(error_wires)}."
        )

    for pauli_type, wire in zip(error_types, error_wires):
        apply_pauli(pauli_type, wire)


def inject_faults_after_cnot(control, target, current_faults):
    """
    Direction-sensitive CNOT fault insertion.

    A fault is injected only if both control and target match exactly.

    Fault format:
        {
            "round": int,
            "control": int,
            "target": int,
            "error_wires": [...],
            "error_types": [...]
        }
    """

    for fault in current_faults:
        if int(fault["control"]) == int(control) and int(fault["target"]) == int(target):
            apply_pauli_error(
                error_types=fault["error_types"],
                error_wires=fault["error_wires"],
            )


def apply_Z_stabilizer(
    anc,
    qubits,
    current_faults,
    mid_measure=True,
    reset_after_measure=True,
):
    """
    Apply one Z stabilizer circuit.

    Convention:
        data q -> ancilla CNOT

    If mid_measure=True:
        measure ancilla and return the PennyLane mid-circuit measurement object.

    If reset_after_measure=True:
        qml.measure(anc, reset=True)

    If reset_after_measure=False:
        qml.measure(anc, reset=False)
    """

    for q in qubits:
        qml.CNOT(wires=[q, anc])

        inject_faults_after_cnot(
            control=q,
            target=anc,
            current_faults=current_faults,
        )

    if mid_measure:
        return qml.measure(
            anc,
            reset=reset_after_measure,
        )

    return None


def apply_X_stabilizer(
    anc,
    qubits,
    current_faults,
    mid_measure=True,
    reset_after_measure=True,
):
    """
    Apply one X stabilizer circuit.

    Convention:
        H on ancilla
        ancilla -> data q CNOT
        H on ancilla

    If mid_measure=True:
        measure ancilla and return the PennyLane mid-circuit measurement object.
    """

    qml.Hadamard(wires=anc)

    for q in qubits:
        qml.CNOT(wires=[anc, q])

        inject_faults_after_cnot(
            control=anc,
            target=q,
            current_faults=current_faults,
        )

    qml.Hadamard(wires=anc)

    if mid_measure:
        return qml.measure(
            anc,
            reset=reset_after_measure,
        )

    return None


def stabilizer_round(
    current_round,
    fault_schedule,
    mid_measure=True,
    reset_after_measure=True,
):
    """
    Apply one full stabilizer round.

    Order:
        Z stabilizers first
        X stabilizers second

    Returned order when mid_measure=True:
        [Z0, Z1, Z2, Z3, X0, X1, X2, X3]

    If mid_measure=False:
        returns an empty list.
    """

    current_faults = get_faults_for_round(
        fault_schedule=fault_schedule,
        current_round=current_round,
    )

    round_record = []

    for i, stab in enumerate(Z_stabilizers):
        m = apply_Z_stabilizer(
            anc=anc_z[i],
            qubits=stab,
            current_faults=current_faults,
            mid_measure=mid_measure,
            reset_after_measure=reset_after_measure,
        )

        if mid_measure:
            round_record.append(m)

    for i, stab in enumerate(X_stabilizers):
        m = apply_X_stabilizer(
            anc=anc_x[i],
            qubits=stab,
            current_faults=current_faults,
            mid_measure=mid_measure,
            reset_after_measure=reset_after_measure,
        )

        if mid_measure:
            round_record.append(m)

    return round_record