# src/round_propagation.py

"""
Symbolic propagation of one stabilizer round on a PauliFrame.

Mirrors `src/stabilizer_circuit.py:stabilizer_round` gate-for-gate so that the
Pauli frame we maintain is by construction consistent with what the PennyLane
circuit actually does. The output is the post-round Pauli frame; the
data-qubit part is what we carry into the next round, the ancilla part can be
read off to predict the syndrome under reset mode.

Used for two purposes:

1. **Building the LookupDecoder's residual table.** For each (CNOT k, Pauli α),
   the post-round residual on the 9 data qubits (with fresh ancillas and no
   prior frame) is precomputed once.

2. **Live frame book-keeping inside `sequence_runner.run_sequence` for non-
   Identity decoders.** Between rounds, the frame accumulates corrections and
   residuals so the next round's QNode can be primed with the correct
   `initial_error_list`.
"""

from typing import Iterable, List

import numpy as np

from .pauli_frame import PauliFrame
from .stochastic_faults import _pauli_pair_to_error_lists, _PAULI_PAIRS as _PAULI_PAIRS_SAMPLER
from .surface_code_layout import (
    DATA_WIRES,
    X_stabilizers,
    Z_stabilizers,
    anc_x,
    anc_z,
    n_data,
    n_qubits,
)


PAULI_PAIRS_15 = tuple(_PAULI_PAIRS_SAMPLER)
assert len(PAULI_PAIRS_15) == 15


def _faults_for(c: int, t: int, round_faults: List[dict]) -> List[dict]:
    """Return all fault entries firing at CNOT (c, t) — order preserved."""
    matches = []
    for f in round_faults:
        if int(f["control"]) == int(c) and int(f["target"]) == int(t):
            matches.append(f)
    return matches


def _apply_fault_to_frame(frame: PauliFrame, fault: dict) -> None:
    """XOR a fault's Pauli string into the frame."""
    for ptype, wire in zip(fault["error_types"], fault["error_wires"]):
        frame.apply_pauli(ptype, int(wire))


def propagate_round(
    start_frame: PauliFrame,
    round_faults: Iterable[dict],
    reset_after_measure: bool = True,
) -> PauliFrame:
    """
    Apply one stabilizer round's circuit symbolically to a copy of `start_frame`.

    Gate order replicates `src/stabilizer_circuit.py:stabilizer_round`:

        for i in 0..3 (Z stabs, ancilla = anc_z[i]):
            for q in Z_stabilizers[i]:
                CNOT(q, anc_z[i])
                [apply matching fault Paulis to the frame]
        for i in 0..3 (X stabs, ancilla = anc_x[i]):
            H(anc_x[i])
            for q in X_stabilizers[i]:
                CNOT(anc_x[i], q)
                [apply matching fault Paulis to the frame]
            H(anc_x[i])
        (measurement + optional reset on ancillas)

    Under reset mode, each ancilla's (x, z) is forced to (0, 0) at the end
    so the frame entering the next round has clean ancillas.

    Returns:
        A new PauliFrame (start_frame is not mutated).
    """
    round_faults = list(round_faults)
    frame = start_frame.copy()

    # Z stabilizers
    for i, stab in enumerate(Z_stabilizers):
        anc = anc_z[i]
        for q in stab:
            frame.apply_cnot(int(q), int(anc))
            for fault in _faults_for(q, anc, round_faults):
                _apply_fault_to_frame(frame, fault)

    # X stabilizers
    for i, stab in enumerate(X_stabilizers):
        anc = anc_x[i]
        frame.apply_h(int(anc))
        for q in stab:
            frame.apply_cnot(int(anc), int(q))
            for fault in _faults_for(anc, q, round_faults):
                _apply_fault_to_frame(frame, fault)
        frame.apply_h(int(anc))

    # Measurement + reset: ancilla frame components don't propagate to next
    # round (the physical ancilla is forced to |0⟩). We zero them out here so
    # the returned frame is exactly what enters the next round's QNode.
    if reset_after_measure:
        for q in (*anc_z, *anc_x):
            frame.reset_qubit(int(q))

    return frame


def predicted_syndrome(post_round_frame: PauliFrame) -> np.ndarray:
    """
    Read the syndrome bits predicted by the post-round frame *before* the
    reset zeroes the ancilla components.

    NOTE: pass a frame from `propagate_round(..., reset_after_measure=False)`
    or accept that calling this on a reset-zeroed frame always gives zero.

    Z-basis measurement outcome on a qubit q reads off whether the frame
    Pauli on q anti-commutes with Z_q, i.e. whether the frame has an X
    component (X or Y) on q. So:

        outcome[q] = x[q]

    For both Z- and X-stab ancillas this is the right answer: the X-stab
    circuit conjugates the measurement by Hadamards so that the post-round
    frame's x[anc_x] is exactly what is read on the wire.

    Returns:
        (8,) uint8 in order [Z0, Z1, Z2, Z3, X0, X1, X2, X3].
    """
    bits = np.zeros(8, dtype=np.uint8)
    for i, anc in enumerate(anc_z):
        bits[i] = int(post_round_frame.x[anc]) & 1
    for i, anc in enumerate(anc_x):
        bits[4 + i] = int(post_round_frame.x[anc]) & 1
    return bits


def propagate_round_with_syndrome(
    start_frame: PauliFrame,
    round_faults: Iterable[dict],
    reset_after_measure: bool = True,
) -> tuple:
    """
    Convenience: compute the post-round frame AND its predicted syndrome.

    Internally we run the propagator with reset=False, read off the
    syndrome, then optionally apply the reset for the returned frame.
    """
    post_no_reset = propagate_round(
        start_frame, round_faults, reset_after_measure=False
    )
    syndrome = predicted_syndrome(post_no_reset)
    if reset_after_measure:
        post = post_no_reset.copy()
        for q in (*anc_z, *anc_x):
            post.reset_qubit(int(q))
    else:
        post = post_no_reset
    return post, syndrome


def build_single_fault_residual_lookup(reset: bool = True) -> np.ndarray:
    """
    For each (CNOT index k ∈ 0..23, Pauli pair index α ∈ 0..14) compute the
    post-round residual on the 9 data qubits, starting from an empty frame.

    Returns:
        residuals: (24, 15, 2, 9) uint8. residuals[k, α, 0, :] = X-part,
        residuals[k, α, 1, :] = Z-part on data qubits.
    """
    from .surface_code_layout import enumerate_stabilizer_cnots

    cnots = enumerate_stabilizer_cnots()
    n_cnot = len(cnots)
    n_pauli = len(PAULI_PAIRS_15)

    out = np.zeros((n_cnot, n_pauli, 2, n_data), dtype=np.uint8)
    for k, loc in enumerate(cnots):
        c, t = int(loc["control"]), int(loc["target"])
        for alpha, pair in enumerate(PAULI_PAIRS_15):
            err_types, err_wires = _pauli_pair_to_error_lists(pair, c, t)
            fault = {
                "round": 0,
                "control": c,
                "target": t,
                "error_wires": err_wires,
                "error_types": err_types,
            }
            empty = PauliFrame(n=n_qubits)
            post = propagate_round(empty, [fault], reset_after_measure=reset)
            x_data, z_data = post.data_xz()
            out[k, alpha, 0, :] = x_data
            out[k, alpha, 1, :] = z_data
    return out


def build_single_fault_syndrome_lookup(reset: bool = True) -> np.ndarray:
    """
    Symbolically compute the (24, 15) single-fault syndrome lookup, packed as
    int (bit b = stabilizer b in the order [Z0,Z1,Z2,Z3,X0,X1,X2,X3]).

    This is the symbolic counterpart of `ceiling.compute_single_round_lookup`
    (which uses the PennyLane circuit). We use it ONLY for testing: the two
    should agree bit-for-bit if the propagator is correct.
    """
    from .surface_code_layout import enumerate_stabilizer_cnots

    cnots = enumerate_stabilizer_cnots()
    n_cnot = len(cnots)
    n_pauli = len(PAULI_PAIRS_15)

    out = np.zeros((n_cnot, n_pauli), dtype=np.int32)
    for k, loc in enumerate(cnots):
        c, t = int(loc["control"]), int(loc["target"])
        for alpha, pair in enumerate(PAULI_PAIRS_15):
            err_types, err_wires = _pauli_pair_to_error_lists(pair, c, t)
            fault = {
                "round": 0,
                "control": c,
                "target": t,
                "error_wires": err_wires,
                "error_types": err_types,
            }
            empty = PauliFrame(n=n_qubits)
            _, bits = propagate_round_with_syndrome(
                empty, [fault], reset_after_measure=reset
            )
            packed = int(np.dot(bits.astype(np.int64), 1 << np.arange(bits.size)))
            out[k, alpha] = packed
    return out
