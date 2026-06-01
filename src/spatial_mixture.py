"""
Shared utilities for the bag-of-shots spatial mixture model.

Loads the deterministic single-fault atomic table, precomputes
round-shifted versions, and samples d-round syndrome bags under a
dominant-location-plus-background noise model.

The atomic NPZ stores fault@round 0 syndromes only; fault@round r is
modelled by zero-padded time translation (valid for reset and noreset
modes provided round-0 preparation is steady-state; not yet validated
against an explicit fault@round r simulation).
"""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATOM_NPZ = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration" / "fault_enumeration_table.npz"


def load_atoms(mode, npz_path=None):
    """Returns (atoms_dict, d_atomic).

    atoms_dict[location] = (array (n_paulis, d_atomic, 8) int8, [pauli_label])
    """
    z = np.load(npz_path or ATOM_NPZ, allow_pickle=True)
    sel = z["mode"] == mode
    err = z["error"][sel]
    syn = z["syndrome"][sel].astype(np.int8)
    d_atomic = syn.shape[1]

    by_loc = {}
    for e, s in zip(err, syn):
        _, loc, pauli = e.split(":")
        by_loc.setdefault(loc, []).append((pauli, s))
    atoms = {
        loc: (np.stack([s for _, s in items]),
              [p for p, _ in items])
        for loc, items in by_loc.items()
    }
    return atoms, d_atomic


def precompute_shifted_atoms(atoms, d):
    """Return shifted[loc] : (d, n_paulis, d, 8) int8.

    Axis 0 is the fault round r; axis 2 is the round position in the
    d-round shot. Entries before r are zero-padded.
    """
    shifted = {}
    for loc, (arr, _) in atoms.items():
        n_paulis, d_atomic, n_stab = arr.shape
        out = np.zeros((d, n_paulis, d, n_stab), dtype=np.int8)
        for r in range(d):
            end = min(d_atomic, d - r)
            if end > 0:
                out[r, :, r:r + end, :] = arr[:, :end, :]
        shifted[loc] = out
    return shifted


def sample_shots(shifted, all_loc_keys, dominant, p_high, p_bg, d, n_shots, rng):
    """Sample n_shots d-round syndrome bags.

    Events at the dominant location fire at rate p_high per round; at
    every other location, at rate p_bg per round. Pauli draws are
    uniform within each location's atomic pool. The shot syndrome is
    the XOR of all triggered round-shifted atoms.
    """
    n_stab = next(iter(shifted.values())).shape[-1]
    shots = np.zeros((n_shots, d, n_stab), dtype=np.int8)
    for loc in all_loc_keys:
        sh_arr = shifted[loc]
        n_paulis = sh_arr.shape[1]
        rate = p_high if loc == dominant else p_bg
        if rate <= 0.0:
            continue
        event_mask = rng.random((n_shots, d)) < rate
        n_events = int(event_mask.sum())
        if n_events == 0:
            continue
        shot_idx, round_idx = np.where(event_mask)
        pauli_idx = rng.integers(0, n_paulis, size=n_events)
        atoms_to_add = sh_arr[round_idx, pauli_idx]
        np.bitwise_xor.at(shots, shot_idx, atoms_to_add)
    return shots
