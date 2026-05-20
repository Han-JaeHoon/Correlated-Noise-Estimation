# src/fast_simulator.py

"""
Pure-Pauli-frame simulator for stabilizer sequences.

Equivalent to running the PennyLane circuit through `sequence_runner`, but
typically 100–1000× faster because there is no statevector at all — every
gate is an in-place symplectic update on F_2^{17} × F_2^{17}.

Correctness rests on `src/round_propagation.py`'s exact agreement with the
PennyLane circuit (proven by scripts/sanity_check_round_propagation.py:
A. 24x15 single-fault lookups bit-identical; B. data-Pauli preservation;
C. 2-round end-to-end bit-identical).

Used by `src/ceiling_r3b.py` to make MC ceiling sweeps tractable.
"""

import numpy as np

from .decoder import Decoder, IdentityDecoder, LookupDecoder
from .pauli_frame import PauliFrame
from .round_propagation import propagate_round_with_syndrome
from .stochastic_faults import BackgroundElevatedSampler
from .surface_code_layout import n_data, n_qubits


def run_sequence_symbolic(
    T: int,
    sampler: BackgroundElevatedSampler,
    decoder: Decoder,
    reset_after_measure: bool = True,
) -> dict:
    """
    Pure-symbolic stand-in for `sequence_runner.run_sequence`.

    Same signature, same return-key shape (minus the optional 'final_frame'
    key for the IdentityDecoder fast path). Faster by orders of magnitude
    because there are no PennyLane QNode calls.

    Returns:
        {
            'syndromes':      (T, 8) uint8,
            'fault_schedule': list[dict] (all rounds, original round numbers),
            'corrections_log': list[(round_idx, Correction)],
            'final_frame':    PauliFrame,
        }
    """
    decoder.reset()

    frame = PauliFrame(n=n_qubits)
    syndromes = np.zeros((T, 8), dtype=np.uint8)
    full_fault_schedule = []
    corrections_log = []

    for t in range(T):
        round_faults = sampler.sample_round(t)
        full_fault_schedule.extend(round_faults)

        round_faults_local = [dict(f, round=0) for f in round_faults]

        # Propagate the frame through this round's circuit + faults, and
        # read the predicted ancilla outcomes.
        post, syn = propagate_round_with_syndrome(
            frame, round_faults_local, reset_after_measure=reset_after_measure
        )
        syndromes[t] = syn

        # Decoder pass (skipped for IdentityDecoder for speed)
        if isinstance(decoder, IdentityDecoder):
            corr = decoder.decode_window(syn.reshape(1, -1))
        else:
            corr = decoder.decode_window(syn.reshape(1, -1))
            for q in range(n_data):
                if corr.x_correction[q]:
                    post.x[q] ^= 1
                if corr.z_correction[q]:
                    post.z[q] ^= 1
        corrections_log.append((t, corr))

        frame = post

    return {
        "syndromes": syndromes,
        "fault_schedule": full_fault_schedule,
        "corrections_log": corrections_log,
        "final_frame": frame,
    }
