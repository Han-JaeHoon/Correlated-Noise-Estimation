# src/pauli_frame.py

"""
Symplectic Pauli frame on 17 qubits (9 data + 8 ancilla) over F_2.

A Pauli on a single qubit is represented by (x, z) in {0,1}^2:
    I = (0, 0), X = (1, 0), Z = (0, 1), Y = (1, 1).

The frame stores the X and Z components as two length-n uint8 arrays.

Gate updates use the Heisenberg-picture symplectic rules:
    CNOT(c, t):
        x[t] ^= x[c]      # X spreads control → target
        z[c] ^= z[t]      # Z spreads target → control
    H(q):
        swap x[q] and z[q]
    Reset(q) (after measurement; we don't track measurement outcomes here):
        x[q] = z[q] = 0

Ancilla measurement outcomes are *not* tracked in this module — the actual
PennyLane circuit handles them. This module exists only to propagate the
data-qubit Pauli frame between rounds and to compute the post-round residual
of single faults for use by the LookupDecoder.
"""

import numpy as np

from .surface_code_layout import n_data, n_qubits


class PauliFrame:
    """Mutable symplectic Pauli frame on n qubits."""

    __slots__ = ("n", "x", "z")

    def __init__(self, n: int = n_qubits):
        self.n = int(n)
        self.x = np.zeros(self.n, dtype=np.uint8)
        self.z = np.zeros(self.n, dtype=np.uint8)

    # --- construction helpers --------------------------------------------------

    @classmethod
    def from_xz(cls, x: np.ndarray, z: np.ndarray) -> "PauliFrame":
        if x.shape != z.shape:
            raise ValueError(f"x and z shapes differ: {x.shape} vs {z.shape}")
        f = cls(n=x.size)
        f.x = np.asarray(x, dtype=np.uint8).copy()
        f.z = np.asarray(z, dtype=np.uint8).copy()
        return f

    def copy(self) -> "PauliFrame":
        return PauliFrame.from_xz(self.x, self.z)

    # --- gate updates ----------------------------------------------------------

    def apply_cnot(self, c: int, t: int) -> None:
        """In-place CNOT propagation. control=c, target=t."""
        self.x[t] ^= self.x[c]
        self.z[c] ^= self.z[t]

    def apply_h(self, q: int) -> None:
        """In-place Hadamard: swap X and Z on qubit q."""
        self.x[q], self.z[q] = self.z[q], self.x[q]

    def reset_qubit(self, q: int) -> None:
        """Used for ancilla reset after measurement."""
        self.x[q] = 0
        self.z[q] = 0

    def apply_pauli(self, pauli_type: str, wire: int) -> None:
        """XOR a single Pauli into the frame at the given wire."""
        if pauli_type == "I":
            return
        if pauli_type == "X":
            self.x[wire] ^= 1
        elif pauli_type == "Z":
            self.z[wire] ^= 1
        elif pauli_type == "Y":
            self.x[wire] ^= 1
            self.z[wire] ^= 1
        else:
            raise ValueError(f"Unknown Pauli type: {pauli_type!r}")

    def xor_(self, other: "PauliFrame") -> None:
        """In-place XOR with another frame of the same size."""
        if other.n != self.n:
            raise ValueError(f"size mismatch: {self.n} vs {other.n}")
        self.x ^= other.x
        self.z ^= other.z

    # --- queries ---------------------------------------------------------------

    def data_xz(self) -> tuple:
        """Return (x_data, z_data) copies restricted to the 9 data qubits."""
        return self.x[:n_data].copy(), self.z[:n_data].copy()

    def is_identity(self) -> bool:
        return (not self.x.any()) and (not self.z.any())

    def is_data_identity(self) -> bool:
        return (not self.x[:n_data].any()) and (not self.z[:n_data].any())

    def to_initial_error_lists(self):
        """
        Convert the data-qubit part of the frame into (error_types, error_wires)
        suitable for `make_repeated_stabilizer_qnode(initial_error_list=...,
        initial_error_wires=...)`.

        Only the 9 data qubits are emitted. Ancilla components are ignored on
        the assumption that ancilla qubits start each round in |0⟩ (they are
        reset at the end of the previous round).
        """
        types = []
        wires = []
        for q in range(n_data):
            xq = int(self.x[q])
            zq = int(self.z[q])
            if xq and zq:
                types.append("Y")
                wires.append(q)
            elif xq:
                types.append("X")
                wires.append(q)
            elif zq:
                types.append("Z")
                wires.append(q)
        return types, wires

    def __repr__(self) -> str:
        return f"PauliFrame(n={self.n}, x={self.x.tolist()}, z={self.z.tolist()})"
