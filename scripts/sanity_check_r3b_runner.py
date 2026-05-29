"""
End-to-end smoke test for the R3b sequence runner with LookupDecoder.

Verifies that:

  D. IdentityDecoder fast-path and IdentityDecoder window-path produce
     identical syndromes for the same sampler+seed. (Regression check that
     the window path doesn't drift from the existing R1 path.)

  E. LookupDecoder + window-path runs to completion on representative
     fault rates without raising. Output shapes are well-formed.

  F. LookupDecoder syndromes differ from IdentityDecoder syndromes on at
     least one round for at least one (faulty_cnot_id, seed) pair. (Trivial
     check that the decoder is actually doing something.)

This is the runtime safety net for the LookupDecoder + window-by-window
runner combination. The deeper question — does R3b break the ambiguity
groups? — is the subject of the R3b ceiling analysis (Task #6 in the
overnight plan).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.decoder import IdentityDecoder, LookupDecoder
from src.sequence_runner import run_sequence
from src.stochastic_faults import BackgroundElevatedSampler


def _sampler(seed: int, p_bg: float, p_high: float, faulty: int):
    return BackgroundElevatedSampler(
        p_bg=p_bg, p_high=p_high, faulty_cnot_id=faulty,
        rng=np.random.default_rng(seed),
    )


def check_D_fast_vs_window_identity() -> bool:
    print("=" * 72)
    print("D. IdentityDecoder: fast path vs window path syndromes")
    ok = True
    for s in range(8):
        a = run_sequence(
            T=20,
            sampler=_sampler(s, 0.05, 0.2, faulty=8),
            decoder=IdentityDecoder(),
            force_window_path=False,
        )
        b = run_sequence(
            T=20,
            sampler=_sampler(s, 0.05, 0.2, faulty=8),
            decoder=IdentityDecoder(),
            force_window_path=True,
        )
        if not np.array_equal(a["syndromes"], b["syndromes"]):
            print(f"   FAIL seed={s}")
            ok = False
    print(f"   {'PASS' if ok else 'FAIL'}  (8 seeds × T=20)")
    return ok


def check_E_lookup_decoder_smoke() -> bool:
    print("=" * 72)
    print("E. LookupDecoder runs to completion + well-formed output")
    dec = LookupDecoder(reset=True)
    out = run_sequence(
        T=30,
        sampler=_sampler(0, 0.01, 0.1, faulty=8),
        decoder=dec,
    )
    ok = (
        isinstance(out, dict)
        and out["syndromes"].shape == (30, 8)
        and out["syndromes"].dtype == np.uint8
        and len(out["corrections_log"]) == 30
        and out["final_frame"] is not None
    )
    print(f"   syndromes shape = {out['syndromes'].shape}, dtype = {out['syndromes'].dtype}")
    print(f"   corrections_log len = {len(out['corrections_log'])}")
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def check_F_lookup_changes_syndromes() -> bool:
    print("=" * 72)
    print("F. LookupDecoder syndromes differ from IdentityDecoder")
    dec_id = IdentityDecoder()
    dec_lk = LookupDecoder(reset=True)
    any_diff = False
    for s in range(5):
        for faulty in (0, 8, 18):
            a = run_sequence(T=40, sampler=_sampler(s, 0.02, 0.15, faulty), decoder=dec_id)
            b = run_sequence(T=40, sampler=_sampler(s, 0.02, 0.15, faulty), decoder=dec_lk)
            if not np.array_equal(a["syndromes"], b["syndromes"]):
                any_diff = True
                break
        if any_diff:
            break
    print(f"   {'PASS' if any_diff else 'FAIL'}  (decoder applies non-trivial corrections)")
    return any_diff


def main() -> None:
    print("R3b LookupDecoder + window-runner — smoke test")
    print()
    d = check_D_fast_vs_window_identity()
    e = check_E_lookup_decoder_smoke()
    f = check_F_lookup_changes_syndromes()
    print("=" * 72)
    print(f"OVERALL: {'PASS' if (d and e and f) else 'FAIL'}")
    sys.exit(0 if (d and e and f) else 1)


if __name__ == "__main__":
    main()
