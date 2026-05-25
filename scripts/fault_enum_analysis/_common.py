"""
Shared loaders and parsers for the fault-enumeration pattern analyses.

The underlying table comes from scripts/build_fault_enumeration_table.py and
contains 774 records = 387 deterministic single-fault cases x 2 ancilla modes.

    Category A (round-start data idle)   : 27 cases =  9 data * 3 Pauli
    Category C (post-CNOT 2-qubit Pauli) : 360 cases = 24 CNOT * 15 Pauli

Each record:
    error     str  e.g. "A:data4:Y" or "C:cnot12:XZ"
    mode      str  "reset" or "noreset"
    syndrome  (3, 8) int8   raw ancilla outcomes per round (Z0..Z3, X0..X3)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration" / "fault_enumeration_table.npz"
OUT_DIR = PROJECT_ROOT / "data" / "analysis" / "10_fault_enumeration_patterns"

STAB_LABELS = ["Z0", "Z1", "Z2", "Z3", "X0", "X1", "X2", "X3"]
N_ROUNDS = 3
N_STAB = 8
N_BITS = N_ROUNDS * N_STAB  # 24

CAT_A_RE = re.compile(r"^A:data(\d+):([XYZ])$")
CAT_C_RE = re.compile(r"^C:cnot(\d{2}):([IXYZ]{2})$")

# Lexicographic order matching the generator.
TWO_QUBIT_PAULIS = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]
SINGLE_QUBIT_PAULIS = ["X", "Y", "Z"]
N_CNOT = 24
N_DATA = 9


def load_table(path: Optional[Path] = None) -> dict:
    """Load the fault enumeration table and parse descriptors."""

    if path is None:
        path = TABLE_PATH
    d = np.load(path, allow_pickle=True)

    error = d["error"]
    mode = d["mode"]
    syndrome = d["syndrome"]  # (774, 3, 8)
    assert syndrome.shape == (len(error), N_ROUNDS, N_STAB), syndrome.shape

    cat = np.empty(len(error), dtype="U1")
    qubit_or_cnot = np.full(len(error), -1, dtype=np.int32)
    pauli = np.empty(len(error), dtype="U2")

    for i, e in enumerate(error):
        e = str(e)
        m = CAT_A_RE.match(e)
        if m:
            cat[i] = "A"
            qubit_or_cnot[i] = int(m.group(1))
            pauli[i] = m.group(2)
            continue
        m = CAT_C_RE.match(e)
        if m:
            cat[i] = "C"
            qubit_or_cnot[i] = int(m.group(1))
            pauli[i] = m.group(2)
            continue
        raise ValueError(f"Unrecognized error descriptor at index {i}: {e!r}")

    return dict(
        error=error,
        mode=mode,
        syndrome=syndrome,
        cat=cat,
        qubit_or_cnot=qubit_or_cnot,
        pauli=pauli,
        stab_labels=STAB_LABELS,
    )


def flatten_syndrome(syn: np.ndarray) -> np.ndarray:
    """(N, 3, 8) -> (N, 24) uint8."""

    return syn.reshape(syn.shape[0], -1).astype(np.uint8)


def packbits24(syn_flat: np.ndarray) -> np.ndarray:
    """(N, 24) uint8 bits -> (N,) uint32 packed key for fast equality lookups."""

    out = np.zeros(syn_flat.shape[0], dtype=np.uint32)
    for j in range(syn_flat.shape[1]):
        out = (out << 1) | (syn_flat[:, j].astype(np.uint32) & 1)
    return out


def hamming_pairwise(syn_flat: np.ndarray) -> np.ndarray:
    """(N, 24) uint8 -> (N, N) int16 Hamming distance matrix."""

    a = syn_flat.astype(np.int16)
    # Use broadcasting; N up to 774 so 774x774x24 = 14.4M ops, fine.
    diff = (a[:, None, :] != a[None, :, :]).astype(np.int16)
    return diff.sum(axis=2)


def select(t: dict, *, mode: Optional[str] = None, cat: Optional[str] = None) -> dict:
    """Return a sub-dict filtered by mode and/or category."""

    mask = np.ones(len(t["error"]), dtype=bool)
    if mode is not None:
        mask &= t["mode"] == mode
    if cat is not None:
        mask &= t["cat"] == cat

    return dict(
        error=t["error"][mask],
        mode=t["mode"][mask],
        syndrome=t["syndrome"][mask],
        cat=t["cat"][mask],
        qubit_or_cnot=t["qubit_or_cnot"][mask],
        pauli=t["pauli"][mask],
        stab_labels=t["stab_labels"],
    )


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
