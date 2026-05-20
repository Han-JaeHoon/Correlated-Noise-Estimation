"""
Regression test for `src/fast_simulator.run_sequence_symbolic`.

Verifies that the pure-symbolic Pauli-frame simulator produces bit-identical
syndrome streams to the PennyLane runner across both decoders, then
benchmarks the speedup.

This is the runtime safety net for `src/ceiling_r3b.py`, which uses the
symbolic simulator to make MC ceiling sweeps tractable.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.decoder import IdentityDecoder, LookupDecoder
from src.fast_simulator import run_sequence_symbolic
from src.sequence_runner import run_sequence
from src.stochastic_faults import BackgroundElevatedSampler


def _sampler(seed: int, k: int):
    return BackgroundElevatedSampler(
        p_bg=0.02, p_high=0.15, faulty_cnot_id=k,
        rng=np.random.default_rng(seed),
    )


def check_identity_agreement() -> bool:
    print("=" * 72)
    print("Identity decoder: PennyLane fast vs symbolic")
    ok = True
    for s in range(3):
        for k in (0, 8, 18):
            a = run_sequence(T=15, sampler=_sampler(s, k), decoder=IdentityDecoder())
            b = run_sequence_symbolic(T=15, sampler=_sampler(s, k), decoder=IdentityDecoder())
            if not np.array_equal(a["syndromes"], b["syndromes"]):
                print(f"  FAIL seed={s} k={k}")
                ok = False
    if ok:
        print("  PASS (9 cases)")
    return ok


def check_lookup_agreement() -> bool:
    print("=" * 72)
    print("LookupDecoder: PennyLane window-path vs symbolic")
    ok = True
    dec_pl = LookupDecoder(reset=True)
    dec_sym = LookupDecoder(reset=True)
    for s in range(3):
        for k in (0, 8, 18):
            a = run_sequence(T=15, sampler=_sampler(s, k), decoder=dec_pl)
            b = run_sequence_symbolic(T=15, sampler=_sampler(s, k), decoder=dec_sym)
            if not np.array_equal(a["syndromes"], b["syndromes"]):
                print(f"  FAIL seed={s} k={k}")
                ok = False
    if ok:
        print("  PASS (9 cases)")
    return ok


def benchmark() -> None:
    print("=" * 72)
    print("Benchmark: 50 sequences (T=30, LookupDecoder)")
    dec = LookupDecoder(reset=True)
    t0 = time.time()
    for _ in range(50):
        run_sequence_symbolic(T=30, sampler=_sampler(0, 8), decoder=dec)
    dt_sym = time.time() - t0
    print(f"  symbolic: {dt_sym:.2f}s ({dt_sym/50*1000:.1f} ms/seq)")

    t0 = time.time()
    for _ in range(5):
        run_sequence(T=30, sampler=_sampler(0, 8), decoder=dec)
    dt_pl = time.time() - t0
    print(f"  PennyLane (5 seqs only): {dt_pl:.2f}s ({dt_pl/5*1000:.1f} ms/seq)")
    print(f"  speedup ≈ {(dt_pl/5) / (dt_sym/50):.0f}x")


def main() -> None:
    a = check_identity_agreement()
    b = check_lookup_agreement()
    benchmark()
    print("=" * 72)
    print(f"OVERALL: {'PASS' if (a and b) else 'FAIL'}")
    sys.exit(0 if (a and b) else 1)


if __name__ == "__main__":
    main()
