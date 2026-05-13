# src/simulator.py

import pennylane as qml

from .logical_state import prepare_logical_zero
from .stabilizer_circuit import stabilizer_round, apply_pauli
from .surface_code_layout import n_qubits, DATA_WIRES, ANC_WIRES


def make_repeated_stabilizer_qnode(
    n_rounds,
    shots=1,
    mid_measure=True,
    reset_after_measure=True,
    return_probs=False,
):
    """
    Create a QNode for repeated stabilizer circuits.

    Mode 1:
        mid_measure=True, reset_after_measure=True
        -> mid-circuit syndrome measurement + reset

    Mode 2:
        mid_measure=True, reset_after_measure=False
        -> mid-circuit syndrome measurement without reset

    Mode 3:
        mid_measure=False
        -> no mid-circuit measurement
        -> final measurement on ANC_WIRES only

    Args:
        n_rounds: number of repeated stabilizer rounds.
        shots:
            finite integer for qml.sample.
            None is appropriate for qml.probs.
        mid_measure:
            whether to measure ancilla during each stabilizer round.
        reset_after_measure:
            whether to reset ancilla after mid-circuit measurement.
        return_probs:
            only meaningful when mid_measure=False.
            If True, return final probability distribution over ANC_WIRES.
            If False, return final sample over ANC_WIRES.

    Returns:
        QNode
    """

    if mid_measure and shots is None:
        raise ValueError("mid_measure=True requires finite shots for qml.sample.")

    if (not mid_measure) and (not return_probs) and shots is None:
        raise ValueError(
            "mid_measure=False and return_probs=False requires finite shots "
            "for final qml.sample."
        )

    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def repeated_stabilizer_circuit(
        fault_schedule=None,
        initial_error_list=None,
        initial_error_wires=None,
    ):
        if fault_schedule is None:
            fault_schedule = []

        if initial_error_list is None:
            initial_error_list = []

        if initial_error_wires is None:
            initial_error_wires = []

        measurement_record = []

        prepare_logical_zero()

        qml.Barrier(wires=DATA_WIRES)

        for pauli_type, wire in zip(initial_error_list, initial_error_wires):
            apply_pauli(pauli_type, wire)

        for t in range(n_rounds):
            round_record = stabilizer_round(
                current_round=t,
                fault_schedule=fault_schedule,
                mid_measure=mid_measure,
                reset_after_measure=reset_after_measure,
            )

            measurement_record.extend(round_record)

        if mid_measure:
            return [qml.sample(m) for m in measurement_record]

        if return_probs:
            return qml.probs(wires=ANC_WIRES)

        return qml.sample(wires=ANC_WIRES)

    return repeated_stabilizer_circuit