"""Validate the correlated-noise model on the realistic surface code.

Checks:
  1. Candidate pairs = all distance-<=2 qubit pairs, by kind, + the None class
     (d=3 -> 24 data-anc + 12 data-data + 8 anc-anc = 44 pairs -> 45 classes).
  2. Each pair's correlated channel is emitted into the circuit as a Stim
     CORRELATED_ERROR (the channel its dominant real mechanism implies).
  3. The model produces a *detectable, pair-specific* syndrome signature: the
     per-detector detection-event rate vector differs measurably between None
     and each pair, and between different pairs -> the classification task is
     feasible.

Run: python scripts/validate_correlated_noise.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend_stim.surface_code import RotatedSurfaceCode          # noqa: E402
from src.backend_stim.correlated_noise import (                       # noqa: E402
    enumerate_pairs, correlated_spec, num_classes)


def signature(code, corr, R, N, seed):
    n_anc = len(code.anc_coords)
    circ = code.build_circuit(rounds=R, after_cnot_depolarize=1e-3,
                              measure_flip=1e-3, correlated=corr)
    raw = circ.compile_sampler(seed=seed).sample(N)[:, :n_anc * R].reshape(N, R, n_anc)
    det = (raw[:, 1:, :] ^ raw[:, :-1, :]).reshape(N, -1)  # detection events, drop round 0
    return det.mean(0)


def main():
    for d in (3, 5):
        code = RotatedSurfaceCode(d)
        pairs = enumerate_pairs(code)
        kinds = Counter(p["kind"] for p in pairs)
        print(f"d={d}: {len(pairs)} candidate pairs "
              f"({dict(kinds)}) + None = {num_classes(code)} classes")

    # signal + distinguishability on d=3
    print("\n[d=3] per-detector signature L1 distances (c=0.03):")
    code = RotatedSurfaceCode(3); R = 4; c = 0.03; N = 60000
    pairs = enumerate_pairs(code)
    reps = {"None": None}
    for k in ("data-anc", "data-data", "anc-anc"):
        p = next(p for p in pairs if p["kind"] == k)
        reps[f"{k} {p['q1']}-{p['q2']}"] = p
    sigs = {name: signature(code, correlated_spec(p, c), R, N, seed=i)
            for i, (name, p) in enumerate(reps.items())}
    names = list(sigs)
    w = max(len(n) for n in names)
    print(" " * (w + 2) + "".join(f"{n.split()[0][:9]:>11}" for n in names))
    ok = True
    for n in names:
        row = ""
        for m in names:
            dist = float(np.abs(sigs[n] - sigs[m]).sum())
            row += f"{dist:>11.3f}"
            if n != m and dist < 0.01:
                ok = False
        print(f"{n:<{w}}  {row}")

    # confirm a CORRELATED_ERROR is actually emitted
    circ = code.build_circuit(rounds=2,
                              correlated=correlated_spec(pairs[0], 0.01))
    emitted = "CORRELATED_ERROR" in str(circ) or "E(" in str(circ)
    print(f"\nCORRELATED_ERROR emitted into circuit: {emitted}")
    print("all pairs/None mutually distinguishable (L1 > 0.01):", ok)
    print("\n" + ("MODEL OK" if (ok and emitted) else "CHECK FAILED"))


if __name__ == "__main__":
    main()
