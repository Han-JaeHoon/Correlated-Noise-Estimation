"""Correlated-noise model for the realistic surface code (Phase 2 main task).

Physically-grounded correlated errors (see docs/CORRELATED_NOISE_SURVEY.md):
the candidate "correlated locations" are all qubit pairs within coordinate
distance <= 2, each given the correlated channel its dominant real mechanism
implies:

  * data <-> ancilla  (dist sqrt 2, == a CNOT pair) : stray ZZ during the gate
        -> CORRELATED_ERROR Z_data Z_anc, injected right after that CNOT.
  * data <-> data     (dist 2)                       : always-on residual ZZ
        -> CORRELATED_ERROR Z Z, injected once per round (round start).
  * ancilla <-> ancilla (dist 2)                     : readout crosstalk
        -> CORRELATED_ERROR X X, injected right before measurement
           (a Z on an ancilla would be invisible to its Z-basis readout).

On top of this, an i.i.d. circuit-level background (depolarizing CNOTs +
measurement flips) is the "None" class. The task: from syndrome data, classify
which pair (or None) carries the elevated correlation.
"""
from __future__ import annotations

import itertools
from typing import List, Optional

from .surface_code import RotatedSurfaceCode, Coord


def _kind(code: RotatedSurfaceCode, a: Coord, b: Coord) -> str:
    data = set(code.data_coords)
    da, db = a in data, b in data
    if da and db:
        return "data-data"
    if da or db:
        return "data-anc"
    return "anc-anc"


def enumerate_pairs(code: RotatedSurfaceCode) -> List[dict]:
    """All distance-<=2 candidate correlated pairs, canonically ordered.

    Each entry: {index, q1, q2, kind, channel, spec} where ``spec`` is the
    ready-to-use ``correlated`` element for ``build_circuit`` (with ``p``/index
    filled in by :func:`correlated_spec`).
    """
    # map a CNOT pair -> (tick, control, target) for data-anc injection timing
    cnot_by_pair = {}
    for tick, ctrl, tgt, _typ in code.cnot_enumeration():
        cnot_by_pair[frozenset((ctrl, tgt))] = (tick, ctrl, tgt)

    out: List[dict] = []
    allq = list(code.data_coords) + list(code.anc_coords)
    for a, b in itertools.combinations(allq, 2):
        d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        if d2 not in (2, 4):
            continue
        kind = _kind(code, a, b)
        if kind == "data-anc":
            tick, ctrl, tgt = cnot_by_pair[frozenset((a, b))]
            base = {"when": "post_cx", "tick": tick, "q1": ctrl, "q2": tgt,
                    "paulis": "ZZ", "channel": "ZZ@gate"}
        elif kind == "data-data":
            base = {"when": "pre", "q1": a, "q2": b,
                    "paulis": "ZZ", "channel": "ZZ@idle"}
        else:  # anc-anc
            base = {"when": "pre_measure", "q1": a, "q2": b,
                    "paulis": "XX", "channel": "XX@readout"}
        out.append({"q1": a, "q2": b, "kind": kind,
                    "channel": base["channel"], "_base": base})
    # stable ordering: by kind then coords, then assign index
    out.sort(key=lambda e: (e["kind"], e["q1"], e["q2"]))
    for i, e in enumerate(out):
        e["index"] = i
    return out


def correlated_spec(pair: Optional[dict], c: float) -> List[dict]:
    """Return the ``correlated`` list for ``build_circuit``.

    ``pair`` is an entry from :func:`enumerate_pairs`, or ``None`` (background
    only). ``c`` is the correlated-error probability.
    """
    if pair is None:
        return []
    spec = dict(pair["_base"])
    spec["p"] = c
    return [spec]


def num_classes(code: RotatedSurfaceCode) -> int:
    """Number of classification labels: one per pair + the None class."""
    return len(enumerate_pairs(code)) + 1
