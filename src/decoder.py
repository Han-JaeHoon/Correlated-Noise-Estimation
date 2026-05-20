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
