# src/decoder.py

"""
Decoder interface for surface code sequence experiments.

The sequence runner is decoder-agnostic. Different regimes (R1 = no decoder,
R3b = lookup decoder, etc.) plug in by providing a Decoder implementation.

R1 uses IdentityDecoder, which never applies a correction. The runner can
detect this and take a fast path (single QNode call over T rounds).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .surface_code_layout import n_data


@dataclass
class Correction:
    """
    Pauli correction on data qubits.

    x_correction[i] == 1 means apply X to data qubit i.
    z_correction[i] == 1 means apply Z to data qubit i.
    Y is represented as X and Z both set on the same qubit.
    """

    x_correction: np.ndarray  # shape (n_data,), uint8
    z_correction: np.ndarray  # shape (n_data,), uint8

    @classmethod
    def identity(cls, n: int = n_data) -> "Correction":
        return cls(
            x_correction=np.zeros(n, dtype=np.uint8),
            z_correction=np.zeros(n, dtype=np.uint8),
        )

    def is_identity(self) -> bool:
        return (not self.x_correction.any()) and (not self.z_correction.any())


class Decoder(ABC):
    """
    Abstract decoder.

    The sequence runner calls decode_window() every `window_size` rounds,
    passing the detection events accumulated in that window. The returned
    Correction is applied to the data qubits before the next window starts.
    """

    @property
    @abstractmethod
    def window_size(self) -> int:
        """Number of rounds the decoder consumes per call."""

    @abstractmethod
    def decode_window(self, detection_events: np.ndarray) -> Correction:
        """
        Args:
            detection_events: shape (window_size, n_stabilizers), {0,1}.

        Returns:
            Correction to apply on data qubits.
        """

    def reset(self) -> None:
        """Called at the start of every sequence. Override if stateful."""


class IdentityDecoder(Decoder):
    """
    No-op decoder for R1 (no decoding applied).

    window_size is set to 1 by default but is irrelevant: the decoder always
    returns identity, so the sequence runner can short-circuit and execute the
    whole T-round circuit in a single QNode call.
    """

    def __init__(self, n: int = n_data):
        self._n = n

    @property
    def window_size(self) -> int:
        return 1

    def decode_window(self, detection_events: np.ndarray) -> Correction:
        return Correction.identity(self._n)


class LookupDecoder(Decoder):
    """
    Single-round-window lookup decoder for the R3b scenario.

    Built once from the symbolic single-fault lookups:
      - syndrome lookup:  (CNOT k, Pauli α) → 8-bit syndrome σ_{k,α}
      - residual lookup:  (k, α) → data-qubit Pauli residual at round end.

    At each round, the decoder is handed a single round's detection event d.
    It picks the lex-minimum (k, α) consistent with d and returns the
    matching residual as a Correction (applying it cancels the residual under
    the single-fault hypothesis).

    Defaults follow `docs/r3b_design.md`:
      - window_size = 1
      - tie-break = lex min (k, α)
      - d == 0 → identity
      - d ∈ no lookup match (multi-fault round) → identity (conservative)
    """

    def __init__(self, reset: bool = True, n: int = n_data):
        from collections import defaultdict
        from .round_propagation import (
            build_single_fault_residual_lookup,
            build_single_fault_syndrome_lookup,
        )

        self._n = int(n)
        self._reset = bool(reset)

        syn_lookup = build_single_fault_syndrome_lookup(reset=reset)
        res_lookup = build_single_fault_residual_lookup(reset=reset)

        inverse = defaultdict(list)
        n_cnot, n_pauli = syn_lookup.shape
        for k in range(n_cnot):
            for alpha in range(n_pauli):
                s = int(syn_lookup[k, alpha])
                inverse[s].append((k, alpha))
        # tie-break: lex min (k, α) per syndrome
        self._lex_min = {s: min(v) for s, v in inverse.items()}
        self._all_matches = {s: sorted(v) for s, v in inverse.items()}
        self._residuals = res_lookup
        self._syn_lookup = syn_lookup

    @property
    def window_size(self) -> int:
        return 1

    @property
    def residuals(self) -> np.ndarray:
        """(24, 15, 2, 9) uint8 — exposed for diagnostics / tests."""
        return self._residuals

    @property
    def syndrome_lookup(self) -> np.ndarray:
        """(24, 15) int — exposed for diagnostics / tests."""
        return self._syn_lookup

    def decode_window(self, detection_events: np.ndarray) -> Correction:
        d = np.asarray(detection_events[0], dtype=np.int64).reshape(-1)
        if d.size != 8:
            raise ValueError(f"LookupDecoder expects 8-bit detection event, got size {d.size}")
        s = int(np.dot(d, 1 << np.arange(d.size)))

        if s == 0 or s not in self._lex_min:
            return Correction.identity(self._n)

        k, alpha = self._lex_min[s]
        x_corr = self._residuals[k, alpha, 0].copy()
        z_corr = self._residuals[k, alpha, 1].copy()
        return Correction(x_correction=x_corr, z_correction=z_corr)


class HammingNearestDecoder(LookupDecoder):
    """
    Variant of LookupDecoder for multi-fault rounds.

    On round t the observed detection event d is often not in the single-fault
    lookup (multi-fault round). LookupDecoder defaults to identity in that
    case. HammingNearestDecoder instead picks the lookup syndrome σ closest
    to d in Hamming distance and applies its (lex-min) residual.

    For d == 0 still returns identity. For d in the lookup, behavior is
    identical to LookupDecoder.

    This is intended only as a quick exploration of whether the "do something
    for multi-fault rounds" branch matters for the R3b ceiling — not as a
    physically motivated decoder.
    """

    def __init__(self, reset: bool = True, n: int = n_data):
        super().__init__(reset=reset, n=n)
        # Pre-compute lex-min syndrome list once (in lookup order)
        self._lookup_syndromes_sorted = sorted(self._lex_min.keys())

    def decode_window(self, detection_events: np.ndarray) -> Correction:
        d = np.asarray(detection_events[0], dtype=np.int64).reshape(-1)
        if d.size != 8:
            raise ValueError(f"HammingNearestDecoder expects 8-bit detection event, got size {d.size}")
        s = int(np.dot(d, 1 << np.arange(d.size)))

        if s == 0:
            return Correction.identity(self._n)

        if s in self._lex_min:
            k, alpha = self._lex_min[s]
        else:
            # find nearest lookup syndrome by Hamming distance, with lex-min
            # syndrome int as a second-key tie-break for determinism
            best_dist = 9
            best_s = None
            for cand in self._lookup_syndromes_sorted:
                d_h = bin(cand ^ s).count("1")
                if d_h < best_dist or (d_h == best_dist and best_s is None):
                    best_dist = d_h
                    best_s = cand
            assert best_s is not None
            k, alpha = self._lex_min[best_s]

        x_corr = self._residuals[k, alpha, 0].copy()
        z_corr = self._residuals[k, alpha, 1].copy()
        return Correction(x_correction=x_corr, z_correction=z_corr)
