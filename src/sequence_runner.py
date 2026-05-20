# src/sequence_runner.py

"""
Run a single T-round stabilizer sequence with an arbitrary Decoder.

R1 path (IdentityDecoder): a single QNode is built for T rounds and executed
once. This is the fastest path and matches the existing simulator interface.

Future path (window-by-window): when the decoder needs to apply corrections
between windows, the runner must re-prepare the state from the (corrected)
end of the previous window. That is not implemented yet — left as a clear
extension point.
"""

import numpy as np

from .decoder import Decoder, IdentityDecoder
from .postprocess import reshape_mid_measure_syndrome
from .simulator import make_repeated_stabilizer_qnode
from .stochastic_faults import BackgroundElevatedSampler


def run_sequence(
    T: int,
    sampler: BackgroundElevatedSampler,
    decoder: Decoder,
    mid_measure: bool = True,
    reset_after_measure: bool = True,
) -> dict:
    """
    Execute one T-round sequence and return the syndrome stream.

    Returns:
        {
            'syndromes':      (T, 8) uint8,
            'fault_schedule': list[dict] (full oracle log),
        }
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if not mid_measure:
        raise ValueError("sequence runner requires mid_measure=True")

    decoder.reset()

    if isinstance(decoder, IdentityDecoder):
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
        )  # (1, T, 8)
        syndromes = np.asarray(syndromes[0], dtype=np.uint8)  # (T, 8)

        return {
            "syndromes": syndromes,
            "fault_schedule": fault_schedule,
        }

    raise NotImplementedError(
        "Non-identity decoders require window-by-window execution with "
        "state hand-off between windows. This path is not implemented yet."
    )
