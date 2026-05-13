# src/logical_state.py

import pennylane as qml


def prepare_logical_zero():
    """
    Prepare |0_L> of the d=3 rotated surface code
    with logical Z_L = Z0 Z1 Z2.

    This is copied from the current notebook implementation.
    """

    # Independent variables:
    # a -> qubit 0
    # b -> qubit 2
    # c -> qubit 3
    # d -> qubit 8
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.Hadamard(wires=8)

    # q1 = a xor b
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 1])

    # q4 = b xor c
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[3, 4])

    # q5 = b
    qml.CNOT(wires=[2, 5])

    # q6 = c
    qml.CNOT(wires=[3, 6])

    # q7 = c xor d
    qml.CNOT(wires=[3, 7])
    qml.CNOT(wires=[8, 7])