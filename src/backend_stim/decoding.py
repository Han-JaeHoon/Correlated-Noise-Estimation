"""Surface-code decoding — MWPM (minimum-weight perfect matching) baseline.

MWPM is the standard accuracy-oriented reference decoder for the surface code
(Dennis et al. 2002; Fowler et al. 2012). We use **PyMatching 2** (Higgott &
Gidney), the de-facto implementation, driven by the **detector error model
(DEM)** that ``RotatedSurfaceCode.build_circuit`` already produces — so the
decoder is wired to our circuit with no extra modeling.

Pipeline
--------
    circuit ──DEM──▶ pymatching.Matching ──▶ decode detection events ──▶
    predicted logical flips ──compare to true observables──▶ logical error rate

The decoder is **noise-model agnostic**: it consumes whatever DEM the circuit
compiles to. Add gate noise (``after_cnot_depolarize``), measurement noise
(``measure_flip``), or — later — a correlated-noise channel, and the same
``MatchingDecoder`` decodes it, as long as the DEM is graphlike
(``decompose_errors=True``). For non-graphlike correlated errors PyMatching
still runs on the decomposed graph; a correlation-aware decoder (BP+OSD) would
be a separate, heavier option.

Other decoders, for reference (not implemented here):
  - Union-Find — near-MWPM accuracy, much faster (good for large d / real time);
  - BP+OSD — for general / correlated (qLDPC) codes, higher cost;
  - neural (e.g. AlphaQubit) — highest accuracy, learned, expensive.
MWPM is the representative baseline and the natural benchmark for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim

from .surface_code import RotatedSurfaceCode


@dataclass
class MatchingDecoder:
    """MWPM decoder bound to one (noisy) circuit.

    Construct from a circuit directly, or from a code via :meth:`from_code`.
    """

    circuit: stim.Circuit

    def __post_init__(self) -> None:
        import pymatching  # imported lazily so the module loads without it

        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(self.dem)
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_code(
        cls,
        code: RotatedSurfaceCode,
        rounds: Optional[int] = None,
        basis: str = "Z",
        after_cnot_depolarize: float = 1e-3,
        measure_flip: float = 1e-3,
        reset: bool = True,
    ) -> "MatchingDecoder":
        """Build a decoder for ``code`` under a simple circuit-level noise model.

        Defaults: ``rounds = d`` and gate+measurement noise at 1e-3.
        """
        rounds = rounds if rounds is not None else code.d
        circ = code.build_circuit(
            rounds=rounds, basis=basis,
            after_cnot_depolarize=after_cnot_depolarize,
            measure_flip=measure_flip, reset=reset,
        )
        return cls(circ)

    # ----------------------------------------------------------------- decode
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode a batch of detection events.

        ``detection_events`` : ``(shots, num_detectors)`` bool/uint8.
        Returns predicted observable flips ``(shots, num_observables)`` uint8.
        """
        return self.matching.decode_batch(detection_events)

    def sample_and_decode(self, shots: int, seed: int = 0):
        """Sample the circuit, decode, and return (predicted, actual) observables."""
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        det, obs = sampler.sample(shots=shots, separate_observables=True)
        pred = self.matching.decode_batch(det)
        return pred, obs

    def logical_error_rate(self, shots: int, seed: int = 0) -> float:
        """Monte-Carlo logical error rate (any observable mispredicted)."""
        pred, obs = self.sample_and_decode(shots, seed=seed)
        return float(np.mean(np.any(pred != obs, axis=1)))


def logical_error_rate(
    code: RotatedSurfaceCode,
    p: float,
    rounds: Optional[int] = None,
    shots: int = 20000,
    seed: int = 0,
    measure_flip: Optional[float] = None,
) -> float:
    """Convenience: per-shot logical error rate for ``code`` at physical rate ``p``.

    Uses depolarizing CNOT noise ``p`` and measurement noise ``measure_flip``
    (defaults to ``p``). ``rounds`` defaults to ``d``.
    """
    dec = MatchingDecoder.from_code(
        code, rounds=rounds, after_cnot_depolarize=p,
        measure_flip=p if measure_flip is None else measure_flip,
    )
    return dec.logical_error_rate(shots, seed=seed)
