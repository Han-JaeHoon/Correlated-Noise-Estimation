# src/sequence_runner.py

"""
Run a single T-round stabilizer sequence with an arbitrary Decoder.

Two code paths:

  - Fast path (IdentityDecoder + force_window_path=False): one QNode call
    over T rounds. Used by the R1 scenario.

  - Window-by-window path (any other Decoder, or force_window_path=True):
    each round is a separate 1-round QNode call. Between rounds, the data-
    qubit Pauli frame is tracked explicitly and seeded back into the next
    round's QNode via `initial_error_list/wires`. The Decoder is consulted
    every `decoder.window_size` rounds; the returned Correction is XOR'd
    into the frame. Used by the R3b scenario.

The window path passes `force_window_path=True` through as a regression
check: running an IdentityDecoder through the window path must yield the
same syndrome stream as the fast path for the same `fault_schedule`. The
overnight sanity script exercises this.
"""

from typing import List

import numpy as np

from .decoder import Correction, Decoder, IdentityDecoder
from .pauli_frame import PauliFrame
from .postprocess import reshape_mid_measure_syndrome
from .round_propagation import propagate_round
from .simulator import make_repeated_stabilizer_qnode
from .stochastic_faults import BackgroundElevatedSampler
from .surface_code_layout import n_data, n_qubits


def run_sequence(
    T: int,
    sampler: BackgroundElevatedSampler,
    decoder: Decoder,
    mid_measure: bool = True,
    reset_after_measure: bool = True,
    force_window_path: bool = False,
) -> dict:
    """
    Execute one T-round sequence and return the syndrome stream.

    Args:
        T: number of rounds.
        sampler: fault sampler used per round.
        decoder: Decoder instance. If IdentityDecoder and not
            `force_window_path`, the fast (single-QNode) path is used.
        mid_measure: must be True.
        reset_after_measure: passed through to the QNode and the propagator.
        force_window_path: if True, always use the window path (useful for
            regression-testing the window path against the fast path).

    Returns (dict):
        'syndromes':       (T, 8) uint8
        'fault_schedule':  list of fault dicts (oracle log, all rounds)

      Additional keys for the window path only:
        'corrections_log': list of (round_idx, Correction) — what the
                           decoder returned at each window boundary
        'final_frame':     final PauliFrame (data + ancilla parts)
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if not mid_measure:
        raise ValueError("sequence runner requires mid_measure=True")

    decoder.reset()

    fast_path = isinstance(decoder, IdentityDecoder) and not force_window_path

    if fast_path:
        fault_schedule = sampler.sample_schedule(T)

        qnode = make_repeated_stabilizer_qnode(
            n_rounds=T,
            shots=1,
            mid_measure=True,
            reset_after_measure=reset_after_measure,
            return_probs=False,
        )

        raw = qnode(fault_schedule=fault_schedule)

        syndromes = reshape_mid_measure_syndrome(
            raw=raw,
            n_rounds=T,
            shots=1,
            n_stabilizers=8,
        )[0]  # (T, 8)
        syndromes = np.asarray(syndromes, dtype=np.uint8)

        return {
            "syndromes": syndromes,
            "fault_schedule": fault_schedule,
        }

    # --- window-by-window path -----------------------------------------------
    window_size = int(decoder.window_size)
    if window_size != 1:
        raise NotImplementedError(
            f"window_size {window_size} not yet supported (only 1)"
        )

    qnode = make_repeated_stabilizer_qnode(
        n_rounds=1,
        shots=1,
        mid_measure=True,
        reset_after_measure=reset_after_measure,
        return_probs=False,
    )

    frame = PauliFrame(n=n_qubits)
    syndromes = np.zeros((T, 8), dtype=np.uint8)
    full_fault_schedule = []
    corrections_log: List = []

    for t in range(T):
        round_faults = sampler.sample_round(t)
        full_fault_schedule.extend(round_faults)

        # Local copy with round=0 so the 1-round QNode picks them up
        round_faults_local = [dict(f, round=0) for f in round_faults]

        init_types, init_wires = frame.to_initial_error_lists()
        raw = qnode(
            fault_schedule=round_faults_local,
            initial_error_list=init_types,
            initial_error_wires=init_wires,
        )
        round_syn = np.asarray(
            reshape_mid_measure_syndrome(
                raw=raw, n_rounds=1, shots=1, n_stabilizers=8
            )[0, 0],
            dtype=np.uint8,
        )
        syndromes[t] = round_syn

        corr = decoder.decode_window(round_syn.reshape(1, -1))
        corrections_log.append((t, corr))

        # Symbolic propagation of (frame + this round's faults)
        post = propagate_round(
            frame, round_faults_local, reset_after_measure=reset_after_measure
        )
        # Apply correction: XOR residual cancellation on data qubits
        for q in range(n_data):
            if corr.x_correction[q]:
                post.x[q] ^= 1
            if corr.z_correction[q]:
                post.z[q] ^= 1
        frame = post

    return {
        "syndromes": syndromes,
        "fault_schedule": full_fault_schedule,
        "corrections_log": corrections_log,
        "final_frame": frame,
    }
