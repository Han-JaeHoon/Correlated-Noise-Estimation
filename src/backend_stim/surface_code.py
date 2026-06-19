"""General distance-d rotated surface code — first-principles construction.

This module builds the rotated (``d`` x ``d``) surface code from explicit
geometric rules, *not* by calling :func:`stim.Circuit.generated`. The
construction is validated against Stim's generator in
``scripts/validate_surface_code.py`` (identical ancilla set + identical
hook-safe CNOT schedule + matching code distance and logical error rate).

Layout rules (derived from, and verified against, the standard rotated code)
----------------------------------------------------------------------------
Coordinates are integer ``(x, y)`` on a grid of side ``2d``:

* **Data qubits** sit at odd-odd points ``(2i+1, 2j+1)`` for ``i, j`` in
  ``0..d-1`` (``d^2`` of them).
* **Ancilla qubits** sit at even-even *face centers* ``(x, y)`` with
  ``x, y`` in ``0, 2, ..., 2d``. A face center carries an ancilla iff:

  - interior (``0 < x < 2d`` and ``0 < y < 2d``): always; or
  - top/bottom boundary (``y in {0, 2d}``): only if it is **X-type**; or
  - left/right boundary (``x in {0, 2d}``): only if it is **Z-type**.

  This yields exactly ``d^2 - 1`` ancillas: ``(d-1)^2`` interior (weight-4)
  plus ``2(d-1)`` boundary (weight-2).
* **Ancilla type** is ``X`` if ``(x//2 + y//2)`` is odd, else ``Z``.

Constant-depth schedule (hook-error safe)
-----------------------------------------
Every ancilla couples to its existing corner data qubits in a fixed **4-tick**
order, the same for all ancillas of a type, so the per-round CNOT depth is 4
regardless of ``d``. The order (relative corner ``(dx, dy)`` -> tick):

* **X** ancilla (control, post-H): ``(+,+)->0  (-,+)->1  (+,-)->2  (-,-)->3``
* **Z** ancilla (target):          ``(+,+)->0  (+,-)->1  (-,+)->2  (-,-)->3``

X and Z use transposed orderings so X-hooks and Z-hooks point along orthogonal
(harmless) directions — the standard hook-error-avoiding schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import stim

Coord = Tuple[int, int]

# corner (dx, dy) -> tick index, per ancilla type
_X_TICK = {(1, 1): 0, (-1, 1): 1, (1, -1): 2, (-1, -1): 3}
_Z_TICK = {(1, 1): 0, (1, -1): 1, (-1, 1): 2, (-1, -1): 3}


@dataclass
class RotatedSurfaceCode:
    """First-principles rotated distance-``d`` surface code.

    Attributes are pure geometry; :meth:`build_circuit` turns them into a
    ``stim.Circuit``.
    """

    d: int
    data_coords: List[Coord] = field(default_factory=list)
    anc_coords: List[Coord] = field(default_factory=list)
    anc_type: Dict[Coord, str] = field(default_factory=dict)
    # ancilla coord -> list of (tick, data_coord) couplings, tick-ordered
    schedule: Dict[Coord, List[Tuple[int, Coord]]] = field(default_factory=dict)
    qubit_index: Dict[Coord, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("distance d must be >= 2")
        self._build_layout()

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        d = self.d
        span = 2 * d  # boundary face centers live at 0 and 2d

        self.data_coords = [
            (2 * i + 1, 2 * j + 1) for j in range(d) for i in range(d)
        ]
        data_set = set(self.data_coords)

        for y in range(0, span + 1, 2):
            for x in range(0, span + 1, 2):
                typ = "X" if ((x // 2 + y // 2) % 2 == 1) else "Z"
                on_tb = y in (0, span)        # top / bottom boundary
                on_lr = x in (0, span)        # left / right boundary
                if on_tb and on_lr:
                    continue                   # corners: no ancilla
                if on_tb and typ != "X":
                    continue                   # top/bottom carry X only
                if on_lr and typ != "Z":
                    continue                   # left/right carry Z only
                # interior or an allowed boundary face
                self.anc_coords.append((x, y))
                self.anc_type[(x, y)] = typ

        # couplings + tick order
        for (ax, ay) in self.anc_coords:
            typ = self.anc_type[(ax, ay)]
            tickmap = _X_TICK if typ == "X" else _Z_TICK
            couplings: List[Tuple[int, Coord]] = []
            for (dx, dy), tick in tickmap.items():
                dcoord = (ax + dx, ay + dy)
                if dcoord in data_set:
                    couplings.append((tick, dcoord))
            couplings.sort()
            self.schedule[(ax, ay)] = couplings

        # qubit indices: data first, then ancillas (stable, readable)
        idx = 0
        for c in self.data_coords:
            self.qubit_index[c] = idx
            idx += 1
        for c in self.anc_coords:
            self.qubit_index[c] = idx
            idx += 1

        if len(self.anc_coords) != d * d - 1:
            raise AssertionError(
                f"expected {d*d-1} ancillas, built {len(self.anc_coords)}"
            )

    # -------------------------------------------------------------- accessors
    @property
    def x_ancillas(self) -> List[Coord]:
        return [a for a in self.anc_coords if self.anc_type[a] == "X"]

    @property
    def z_ancillas(self) -> List[Coord]:
        return [a for a in self.anc_coords if self.anc_type[a] == "Z"]

    def num_directed_cnots(self) -> int:
        return sum(len(v) for v in self.schedule.values())

    def cnot_enumeration(self) -> List[Tuple[int, Coord, Coord, str]]:
        """Canonical per-round directed-CNOT list ``(tick, control, target, type)``.

        X stabilizer: control = ancilla, target = data.
        Z stabilizer: control = data, target = ancilla.
        Ordered by (tick, control-index, target-index) — a stable index set for
        fault-injection sites.
        """
        out: List[Tuple[int, Coord, Coord, str]] = []
        for a in self.anc_coords:
            typ = self.anc_type[a]
            for tick, dcoord in self.schedule[a]:
                if typ == "X":
                    out.append((tick, a, dcoord, "X"))
                else:
                    out.append((tick, dcoord, a, "Z"))
        out.sort(key=lambda r: (r[0], self.qubit_index[r[1]], self.qubit_index[r[2]]))
        return out

    # ----------------------------------------------------------------- circuit
    def build_circuit(
        self,
        rounds: int,
        basis: str = "Z",
        after_cnot_depolarize: float = 0.0,
        measure_flip: float = 0.0,
        reset: bool = True,
        injections=None,
        correlated=None,
    ) -> stim.Circuit:
        """Build the memory-experiment circuit.

        Parameters
        ----------
        rounds : number of stabilizer-measurement rounds.
        basis : ``"Z"`` or ``"X"`` memory experiment.
        after_cnot_depolarize : two-qubit depolarizing rate after every CNOT
            (0 = noiseless; used for validation / threshold / decoder studies).
        measure_flip : probability of a classical bit-flip on each measurement
            (``X_ERROR`` before every ``MR``/``M``) — measurement noise. 0 = off.
        reset : if True, ancillas are measured-and-reset every round (``MR``);
            if False, measured without reset (``M``) — the no-reset mode.
        injections : optional list of deterministic fault dicts, each one of

            * ``{"round": r, "pos": "pre", "coord": (x, y), "pauli": "X"|"Y"|"Z"}``
              — a single-qubit Pauli on a data qubit at the start of round ``r``;
            * ``{"round": r, "pos": "post_cx", "tick": t, "control": c,
              "target": tg, "pauli": "PP"}`` — a 2-qubit Pauli on ``(c, tg)``
              immediately after their CNOT in tick ``t`` of round ``r``.

            Faults are applied as exact Pauli gates (deterministic), matching the
            Phase 0 post-CNOT injection convention.
        """
        injections = injections or []
        pre_inj: Dict[int, list] = {}
        post_inj: Dict[Tuple[int, int], list] = {}
        premeas_inj: Dict[int, list] = {}
        for f in injections:
            if f["pos"] == "pre":
                pre_inj.setdefault(f["round"], []).append(f)
            elif f["pos"] == "pre_measure":
                premeas_inj.setdefault(f["round"], []).append(f)
            else:
                post_inj.setdefault((f["round"], f["tick"]), []).append(f)

        def apply_pauli(circ, pauli, qubits):
            for p, q in zip(pauli, qubits):
                if p in ("X", "Y", "Z"):
                    circ.append(p, [q])

        # Correlated two-qubit noise channels, applied EVERY round at the given
        # position. Each spec: {"when": "pre"|"post_cx"|"pre_measure",
        # "tick": t (post_cx only), "q1": (x,y), "q2": (x,y),
        # "paulis": "ZZ"|"XX"|..., "p": prob}. Emitted as a Stim CORRELATED_ERROR
        # (the Pauli product fires jointly on both qubits with probability p).
        correlated = correlated or []
        pre_corr = [f for f in correlated if f["when"] == "pre"]
        premeas_corr = [f for f in correlated if f["when"] == "pre_measure"]
        postcx_corr: Dict[int, list] = {}
        for f in correlated:
            if f["when"] == "post_cx":
                postcx_corr.setdefault(f["tick"], []).append(f)

        _tmap = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}

        def append_corr(circ, f):
            tgts = [_tmap[p](qi[tuple(q)])
                    for p, q in zip(f["paulis"], (f["q1"], f["q2"]))]
            circ.append("CORRELATED_ERROR", tgts, f["p"])

        basis = basis.upper()
        if basis not in ("Z", "X"):
            raise ValueError("basis must be 'Z' or 'X'")
        d = self.d
        qi = self.qubit_index
        data_idx = [qi[c] for c in self.data_coords]
        anc_idx = [qi[c] for c in self.anc_coords]
        x_anc_idx = [qi[c] for c in self.x_ancillas]

        c = stim.Circuit()
        for coord, i in qi.items():
            c.append("QUBIT_COORDS", [i], list(coord))

        # initial reset; for X-basis memory, put data in |+>
        c.append("R", data_idx + anc_idx)
        if basis == "X":
            c.append("H", data_idx)
        c.append("TICK")

        # measurement-record bookkeeping: most-recent rec index for each ancilla
        meas_count = 0
        last_rec: Dict[Coord, int] = {}
        # detector basis: the deterministic stabilizers in this memory basis
        det_type = "Z" if basis == "Z" else "X"

        def one_round(first: bool, r: int) -> None:
            nonlocal meas_count
            for f in pre_inj.get(r, []):
                apply_pauli(c, f["pauli"], [qi[tuple(f["coord"])]])
            for f in pre_corr:
                append_corr(c, f)
            c.append("H", x_anc_idx)
            c.append("TICK")
            for tick in range(4):
                pairs: List[int] = []
                for a in self.anc_coords:
                    typ = self.anc_type[a]
                    for t, dcoord in self.schedule[a]:
                        if t != tick:
                            continue
                        if typ == "X":
                            pairs += [qi[a], qi[dcoord]]
                        else:
                            pairs += [qi[dcoord], qi[a]]
                if pairs:
                    c.append("CX", pairs)
                    if after_cnot_depolarize > 0:
                        c.append("DEPOLARIZE2", pairs, after_cnot_depolarize)
                for f in post_inj.get((r, tick), []):
                    apply_pauli(c, f["pauli"],
                                [qi[tuple(f["control"])], qi[tuple(f["target"])]])
                for f in postcx_corr.get(tick, []):
                    append_corr(c, f)
                c.append("TICK")
            c.append("H", x_anc_idx)
            c.append("TICK")
            for f in premeas_inj.get(r, []):
                apply_pauli(c, f["pauli"], [qi[tuple(f["q1"])], qi[tuple(f["q2"])]])
            for f in premeas_corr:
                append_corr(c, f)
            if measure_flip > 0:
                c.append("X_ERROR", anc_idx, measure_flip)
            c.append("MR" if reset else "M", anc_idx)
            # record indices for this round's ancilla measurements
            round_rec: Dict[Coord, int] = {}
            for k, a in enumerate(self.anc_coords):
                round_rec[a] = meas_count + k
            meas_count += len(anc_idx)
            # detectors
            for a in self.anc_coords:
                if self.anc_type[a] != det_type:
                    continue
                cur = round_rec[a]
                if first:
                    # deterministic vs reset value 0
                    c.append("DETECTOR",
                             [stim.target_rec(cur - meas_count)],
                             list(a) + [0])
                else:
                    prev = last_rec[a]
                    c.append("DETECTOR",
                             [stim.target_rec(cur - meas_count),
                              stim.target_rec(prev - meas_count)],
                             list(a) + [0])
            last_rec.update(round_rec)
            c.append("SHIFT_COORDS", [], [0, 0, 1])

        for r in range(rounds):
            one_round(first=(r == 0), r=r)

        # final data measurement + reconstruct det_type stabilizers
        if basis == "X":
            c.append("H", data_idx)
        if measure_flip > 0:
            c.append("X_ERROR", data_idx, measure_flip)
        c.append("M", data_idx)
        data_rec: Dict[Coord, int] = {}
        for k, dc in enumerate(self.data_coords):
            data_rec[dc] = meas_count + k
        meas_count += len(data_idx)

        for a in self.anc_coords:
            if self.anc_type[a] != det_type:
                continue
            recs = [stim.target_rec(last_rec[a] - meas_count)]
            for _, dcoord in self.schedule[a]:
                recs.append(stim.target_rec(data_rec[dcoord] - meas_count))
            c.append("DETECTOR", recs, list(a) + [1])

        # logical observable: a string of data qubits crossing the patch.
        # A Z-string is undetected only when it terminates on the Z-type
        # (left/right) boundaries, i.e. it runs horizontally (constant y).
        # X-string runs vertically (constant x). (Both pass through (1, 1).)
        if basis == "Z":
            logical = [dc for dc in self.data_coords if dc[1] == 1]  # row y = 1
        else:
            logical = [dc for dc in self.data_coords if dc[0] == 1]  # column x = 1
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(data_rec[dc] - meas_count) for dc in logical],
                 0)
        return c


def build(d: int, **kw) -> stim.Circuit:
    """Convenience: build a circuit for distance ``d``."""
    return RotatedSurfaceCode(d).build_circuit(**kw)
