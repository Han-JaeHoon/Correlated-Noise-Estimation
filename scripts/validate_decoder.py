"""Validate the MWPM decoder on our surface code: correctness + threshold.

Checks:
  1. Noiseless circuit -> the decoder makes zero logical errors (sanity).
  2. Below-threshold suppression -> at a fixed low physical error rate p, the
     logical error rate *decreases* as the distance d grows. This is the
     defining behavior of a working decoder on a real distance-d code, and
     simultaneously confirms the decoder is correctly wired to our circuit.
  3. A small (p x d) sweep, printed as a table, that brackets the threshold
     (curves for different d cross around p*).

Run: python scripts/validate_decoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend_stim.surface_code import RotatedSurfaceCode          # noqa: E402
from src.backend_stim.decoding import MatchingDecoder, logical_error_rate  # noqa: E402


def main() -> int:
    ok = True

    # 1. noiseless sanity
    print("[1] noiseless sanity (expect 0 logical errors):")
    for d in (3, 5, 7):
        dec = MatchingDecoder.from_code(
            RotatedSurfaceCode(d), after_cnot_depolarize=0.0, measure_flip=0.0)
        ler = dec.logical_error_rate(shots=20000, seed=0)
        good = ler == 0.0
        ok = ok and good
        print(f"    d={d}: LER={ler:.5f}  {'OK' if good else 'FAIL'}")

    # 2. below-threshold suppression at p = 1e-3
    print("\n[2] below-threshold suppression at p=1e-3 (expect LER decreasing in d):")
    p = 1e-3
    lers = {}
    for d in (3, 5, 7):
        lers[d] = logical_error_rate(RotatedSurfaceCode(d), p=p, shots=100000, seed=1)
        print(f"    d={d}: LER={lers[d]:.6f}")
    mono = lers[3] > lers[5] > lers[7]
    ok = ok and mono
    print(f"    monotone decreasing (d=3 > 5 > 7): {mono}  {'OK' if mono else 'FAIL'}")

    # 3. threshold-bracketing sweep
    print("\n[3] (p x d) sweep — curves for different d should cross near threshold:")
    ps = [3e-3, 6e-3, 1e-2, 1.5e-2, 2e-2]
    ds = [3, 5, 7]
    print("      p \\ d  " + "  ".join(f"d={d:>2}" for d in ds))
    for p in ps:
        row = []
        for d in ds:
            ler = logical_error_rate(RotatedSurfaceCode(d), p=p, shots=40000, seed=2)
            row.append(ler)
        print(f"    {p:7.4f}  " + "  ".join(f"{x:5.3f}" for x in row))

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
