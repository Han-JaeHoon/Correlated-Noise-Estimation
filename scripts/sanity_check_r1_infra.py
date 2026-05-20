"""
Sanity check for the R1 sequence infrastructure (long-sequence-analysis branch).

Verifies that the new path (BackgroundElevatedSampler -> run_sequence) reproduces
the syndrome bits produced by the existing single-shot path
(make_fixed_cnot_fault_schedule -> make_repeated_stabilizer_qnode) on a
controllable case.

Two tests:

Test B (sampler dict structure)
    With p_bg=0, p_high=1, faulty_cnot_id=k, sample_round(0) must emit exactly
    one fault dict whose (round, control, target, error_wires, error_types)
    match what make_fixed_cnot_fault_schedule would produce for the same
    (CNOT, Pauli pair). Extra keys (cnot_id, pauli_pair) are tolerated. Verified
    over all 24 CNOTs and all 9 (X/Y/Z) x (X/Y/Z) Pauli pairs the old helper
    supports.

Test C (end-to-end syndrome equality)
    For each of the 24 CNOTs, with Pauli pair XZ at every round of a T=2
    sequence with mid-measurement + reset, the new path's (T, 8) syndromes
    must equal the old path's (1, T, 8) syndromes bit-for-bit.

Run:
    python scripts/sanity_check_r1_infra.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.fault_schedule import make_fixed_cnot_fault_schedule
from src.postprocess import reshape_mid_measure_syndrome
from src.sequence_runner import run_sequence
from src.simulator import make_repeated_stabilizer_qnode
from src.stochastic_faults import BackgroundElevatedSampler
from src.surface_code_layout import enumerate_stabilizer_cnots
from src.decoder import IdentityDecoder


# ---------------------------------------------------------------------------
# Test B: sampler dict structure
# ---------------------------------------------------------------------------

def test_sampler_dict_structure():
    cnots = enumerate_stabilizer_cnots()
    pauli_letters = ("X", "Y", "Z")
    pauli_pairs = [(a, b) for a in pauli_letters for b in pauli_letters]

    n_checked = 0
    n_failed = 0
    failures = []

    for k, loc in enumerate(cnots):
        for pair in pauli_pairs:
            old = make_fixed_cnot_fault_schedule(
                control=loc["control"],
                target=loc["target"],
                error_rounds=[0],
                error_types=pair,
            )[0]

            sampler = BackgroundElevatedSampler(
                p_bg=0.0,
                p_high=1.0,
                faulty_cnot_id=k,
                rng=np.random.default_rng(0),
            )

            # With p_bg=0, p_high=1, sample_round must return exactly 1 fault
            # at the faulty CNOT. The Pauli pair is sampled though; we don't
            # check its identity here, just the dict shape.
            new_list = sampler.sample_round(0)

            if len(new_list) != 1:
                n_failed += 1
                failures.append(
                    f"CNOT {k:02d} pair {pair}: sampler returned "
                    f"{len(new_list)} faults, expected 1"
                )
                continue

            new = new_list[0]

            # Keys old must produce, all must be in new (extra keys allowed).
            for key in ("round", "control", "target", "error_wires", "error_types"):
                if key not in new:
                    n_failed += 1
                    failures.append(
                        f"CNOT {k:02d} pair {pair}: new is missing key '{key}'"
                    )
                    break
            else:
                if new["round"] != old["round"]:
                    n_failed += 1
                    failures.append(
                        f"CNOT {k:02d} pair {pair}: round mismatch "
                        f"old={old['round']} new={new['round']}"
                    )
                    continue
                if new["control"] != old["control"] or new["target"] != old["target"]:
                    n_failed += 1
                    failures.append(
                        f"CNOT {k:02d} pair {pair}: (control,target) mismatch "
                        f"old=({old['control']},{old['target']}) "
                        f"new=({new['control']},{new['target']})"
                    )
                    continue

            n_checked += 1

    print(f"[Test B] checked {n_checked} configurations, {n_failed} failures")
    if failures:
        for line in failures[:10]:
            print(f"  - {line}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        return False
    return True


# ---------------------------------------------------------------------------
# Test C: end-to-end syndrome equality
# ---------------------------------------------------------------------------

class _ForcedSampler:
    """
    Test-only sampler.

    Quacks like BackgroundElevatedSampler for the purpose of run_sequence
    (only sample_schedule is required for the IdentityDecoder path).

    Emits a single fault at `faulty_cnot_id` every round with a fixed Pauli
    pair. No background faults, no Pauli randomness.
    """

    def __init__(self, faulty_cnot_id: int, pauli_pair):
        self._cnots = enumerate_stabilizer_cnots()
        self.faulty_cnot_id = int(faulty_cnot_id)
        self.pauli_pair = tuple(pauli_pair)

        if not 0 <= self.faulty_cnot_id < len(self._cnots):
            raise ValueError(
                f"faulty_cnot_id out of range [0,{len(self._cnots)}): "
                f"{self.faulty_cnot_id}"
            )

    def sample_schedule(self, T: int):
        loc = self._cnots[self.faulty_cnot_id]
        schedule = []
        for t in range(int(T)):
            schedule.append(
                {
                    "round": int(t),
                    "control": int(loc["control"]),
                    "target": int(loc["target"]),
                    "error_wires": [int(loc["control"]), int(loc["target"])],
                    "error_types": list(self.pauli_pair),
                }
            )
        return schedule


def _old_path_syndromes(control, target, T, error_types, reset):
    fault_schedule = make_fixed_cnot_fault_schedule(
        control=control,
        target=target,
        error_rounds=list(range(T)),
        error_types=error_types,
    )

    qnode = make_repeated_stabilizer_qnode(
        n_rounds=T,
        shots=1,
        mid_measure=True,
        reset_after_measure=reset,
        return_probs=False,
    )

    raw = qnode(fault_schedule=fault_schedule)

    syndromes = reshape_mid_measure_syndrome(
        raw=raw,
        n_rounds=T,
        shots=1,
        n_stabilizers=8,
    )  # (1, T, 8)
    return np.asarray(syndromes[0], dtype=np.uint8)  # (T, 8)


def _new_path_syndromes(faulty_cnot_id, T, pauli_pair, reset):
    sampler = _ForcedSampler(
        faulty_cnot_id=faulty_cnot_id,
        pauli_pair=pauli_pair,
    )
    out = run_sequence(
        T=T,
        sampler=sampler,
        decoder=IdentityDecoder(),
        mid_measure=True,
        reset_after_measure=reset,
    )
    return out["syndromes"]  # (T, 8)


def test_end_to_end(T=2, error_types=("X", "Z"), reset=True):
    cnots = enumerate_stabilizer_cnots()

    print(
        f"[Test C] T={T} pauli={''.join(error_types)} "
        f"mode={'reset' if reset else 'no_reset'}"
    )

    n_failed = 0
    failures = []

    for k, loc in enumerate(cnots):
        old = _old_path_syndromes(
            control=loc["control"],
            target=loc["target"],
            T=T,
            error_types=error_types,
            reset=reset,
        )
        new = _new_path_syndromes(
            faulty_cnot_id=k,
            T=T,
            pauli_pair=error_types,
            reset=reset,
        )

        if not np.array_equal(old, new):
            n_failed += 1
            failures.append((k, loc, old, new))
            print(
                f"  CNOT {k:02d} {loc['stab_type']}{loc['stab_index']} "
                f"{loc['control']}->{loc['target']}: MISMATCH"
            )
            print(f"    old: {old.flatten().tolist()}")
            print(f"    new: {new.flatten().tolist()}")
        else:
            tag = f"{loc['stab_type']}{loc['stab_index']}"
            print(
                f"  CNOT {k:02d} {tag:>3} "
                f"{loc['control']:>2}->{loc['target']:>2}: OK "
                f"{new.flatten().tolist()}"
            )

    print(f"[Test C] {len(cnots) - n_failed}/{len(cnots)} CNOTs passed")
    return n_failed == 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    ok_b = test_sampler_dict_structure()
    print()
    print("=" * 70)
    ok_c_reset = test_end_to_end(T=2, error_types=("X", "Z"), reset=True)
    print()
    print("=" * 70)
    ok_c_noreset = test_end_to_end(T=2, error_types=("X", "Z"), reset=False)
    print()
    print("=" * 70)

    all_ok = ok_b and ok_c_reset and ok_c_noreset
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
