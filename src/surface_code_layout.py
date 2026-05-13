# src/surface_code_layout.py

"""
d=3 rotated surface code layout.

Data qubits:
    0, 1, ..., 8

Z ancilla:
    9, 10, 11, 12

X ancilla:
    13, 14, 15, 16

Stabilizer measurement convention:
    Z stabilizer:
        data -> ancilla CNOT

    X stabilizer:
        H on ancilla
        ancilla -> data CNOT
        H on ancilla
"""

n_data = 9
n_anc = 8
n_qubits = n_data + n_anc

data_qubits = list(range(n_data))

anc_z = list(range(n_data, n_data + 4))
anc_x = list(range(n_data + 4, n_qubits))

DATA_WIRES = data_qubits
ANC_WIRES = anc_z + anc_x

X_stabilizers = [
    [0, 1],
    [1, 2, 4, 5],
    [3, 4, 6, 7],
    [7, 8],
]

Z_stabilizers = [
    [0, 1, 3, 4],
    [2, 5],
    [3, 6],
    [4, 5, 7, 8],
]


def enumerate_stabilizer_cnots():
    """
    Return all directed CNOT locations appearing in one stabilizer round.

    Z stabilizer:
        data -> ancilla

    X stabilizer:
        ancilla -> data

    Returns:
        list[dict]
    """

    cnot_locations = []

    for i, stab in enumerate(Z_stabilizers):
        anc = anc_z[i]
        for q in stab:
            cnot_locations.append(
                {
                    "stab_type": "Z",
                    "stab_index": i,
                    "control": q,
                    "target": anc,
                }
            )

    for i, stab in enumerate(X_stabilizers):
        anc = anc_x[i]
        for q in stab:
            cnot_locations.append(
                {
                    "stab_type": "X",
                    "stab_index": i,
                    "control": anc,
                    "target": q,
                }
            )

    return cnot_locations