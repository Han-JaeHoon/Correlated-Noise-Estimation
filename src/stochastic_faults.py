# src/stochastic_faults.py

"""
Stochastic fault sampling for sequence experiments.

Model: at every CNOT location in every round, a fault occurs independently with
some probability. If a fault occurs, one of the 15 non-identity two-qubit Pauli
operators is drawn uniformly at random and applied to (control, target).

Two probability levels:
    - p_bg: background probability, applied to all CNOTs.
    - p_high: elevated probability, applied to a single designated CNOT
              (faulty_cnot_id). If faulty_cnot_id is None, all CNOTs use p_bg.

The output is a fault_schedule (list of dicts) compatible with the existing
stabilizer circuit code in src/stabilizer_circuit.py.
"""

from typing import Optional

import numpy as np

from .surface_code_layout import enumerate_stabilizer_cnots


# 15 non-identity 2-qubit Paulis, ordered for reproducibility.
_PAULI_PAIRS = tuple(
    (a, b)
    for a in ("I", "X", "Y", "Z")
    for b in ("I", "X", "Y", "Z")
    if not (a == "I" and b == "I")
)
assert len(_PAULI_PAIRS) == 15


def _pauli_pair_to_error_lists(pair, control: int, target: int):
    """
    Convert ('X', 'Z') + (control=4, target=12) into the (error_types,
    error_wires) lists expected by apply_pauli_error. I components are dropped.
    """
    pc, pt = pair
    error_types = []
    error_wires = []
    if pc != "I":
        error_types.append(pc)
        error_wires.append(int(control))
    if pt != "I":
        error_types.append(pt)
        error_wires.append(int(target))
    return error_types, error_wires


class BackgroundElevatedSampler:
    """
    Per-round, per-CNOT independent fault sampler.

    Args:
        p_bg: background fault probability for non-faulty CNOTs.
        p_high: elevated fault probability for the faulty CNOT.
        faulty_cnot_id: index in enumerate_stabilizer_cnots() (0..23) that
            uses p_high. If None, all CNOTs use p_bg (baseline).
        rng: numpy Generator for reproducibility.
    """

    def __init__(
        self,
        p_bg: float,
        p_high: float,
        faulty_cnot_id: Optional[int],
        rng: np.random.Generator,
    ):
        if not 0.0 <= p_bg <= 1.0:
            raise ValueError(f"p_bg out of [0,1]: {p_bg}")
        if not 0.0 <= p_high <= 1.0:
            raise ValueError(f"p_high out of [0,1]: {p_high}")

        self._cnots = enumerate_stabilizer_cnots()
        n_cnots = len(self._cnots)

        if faulty_cnot_id is not None:
            if not 0 <= int(faulty_cnot_id) < n_cnots:
                raise ValueError(
                    f"faulty_cnot_id must be in [0,{n_cnots}) or None, "
                    f"got {faulty_cnot_id}"
                )
            faulty_cnot_id = int(faulty_cnot_id)

        self.p_bg = float(p_bg)
        self.p_high = float(p_high)
        self.faulty_cnot_id = faulty_cnot_id
        self.rng = rng

    @property
    def cnots(self):
        return self._cnots

    def sample_round(self, round_idx: int) -> list:
        """Sample fault entries for one round."""
        faults = []
        for cid, loc in enumerate(self._cnots):
            p = self.p_high if cid == self.faulty_cnot_id else self.p_bg
            if self.rng.random() >= p:
                continue
            pair = _PAULI_PAIRS[int(self.rng.integers(0, 15))]
            err_types, err_wires = _pauli_pair_to_error_lists(
                pair, loc["control"], loc["target"],
            )
            if not err_types:
                # II should be excluded; defensive only.
                continue
            faults.append(
                {
                    "round": int(round_idx),
                    "control": int(loc["control"]),
                    "target": int(loc["target"]),
                    "error_wires": err_wires,
                    "error_types": err_types,
                    "cnot_id": int(cid),
                    "pauli_pair": [pair[0], pair[1]],
                }
            )
        return faults

    def sample_schedule(self, T: int) -> list:
        """Sample fault entries for T rounds, concatenated."""
        schedule = []
        for t in range(int(T)):
            schedule.extend(self.sample_round(t))
        return schedule
