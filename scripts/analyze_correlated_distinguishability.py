"""Exact distinguishability of the correlated-noise classes (Phase 0 TV=0 test).

Every class differs from the i.i.d. background by exactly ONE correlated
mechanism, so two classes have identical syndrome distributions (TV = 0,
fundamentally indistinguishable) **iff** their correlated error produces the
same deterministic syndrome signature. We compute that signature exactly (fire
the correlated Pauli once, no background, same-seed raw-syndrome difference) and
group classes by it -- no Monte-Carlo / no estimation error.

Caveat: the signature (hence the collision structure) depends on the *injection
convention* -- where in the round the stray correlated error is applied. With
the data<->ancilla ZZ injected right after that pair's CNOT, the ancilla-Z can
propagate through the ancilla's remaining CNOTs and complete a stabilizer, so
some pairs come out silent. This is a real geometric effect but is tied to the
modeling choice; see the per-group output.

Run: python scripts/analyze_correlated_distinguishability.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend_stim.surface_code import RotatedSurfaceCode          # noqa: E402
from src.backend_stim.correlated_noise import enumerate_pairs          # noqa: E402

SEED = 7


def det_injection(base):
    """Deterministic single-firing of a class's correlated Pauli at round 1."""
    w = base["when"]
    if w == "post_cx":
        return [{"round": 1, "pos": "post_cx", "tick": base["tick"],
                 "control": base["q1"], "target": base["q2"], "pauli": base["paulis"]}]
    if w == "pre":
        return [{"round": 1, "pos": "pre", "coord": base["q1"], "pauli": base["paulis"][0]},
                {"round": 1, "pos": "pre", "coord": base["q2"], "pauli": base["paulis"][1]}]
    return [{"round": 1, "pos": "pre_measure",
             "q1": base["q1"], "q2": base["q2"], "pauli": base["paulis"]}]


def analyze(d: int, R: int | None = None):
    code = RotatedSurfaceCode(d)
    R = R or d + 1
    n_anc = len(code.anc_coords)
    pairs = enumerate_pairs(code)

    def raw(inj):
        c = code.build_circuit(rounds=R, injections=inj)  # no background noise
        return c.compile_sampler(seed=SEED).sample(1)[0][:n_anc * R].reshape(R, n_anc).astype(np.uint8)

    base = raw([])
    groups = defaultdict(list)
    for p in pairs:
        sig = (base ^ raw(det_injection(p["_base"]))).tobytes()
        groups[sig].append(p)

    zero = np.zeros((R, n_anc), np.uint8).tobytes()
    n_silent = len(groups.get(zero, []))
    colliding = {s: g for s, g in groups.items() if len(g) > 1}

    print(f"\n=== d={d} (rounds={R}) ===")
    print(f"classes: {len(pairs)} pairs + None")
    print(f"distinct exact signatures: {len(groups)}  "
          f"(< {len(pairs)} => fundamentally indistinguishable cases exist)")
    print(f"silent pairs (signature == 0 == indistinguishable from None): {n_silent}")
    print(f"collision groups (>=2 members, incl. silent): {len(colliding)}")
    for s, g in sorted(colliding.items(), key=lambda kv: -len(kv[1])):
        tag = " [SILENT == None]" if s == zero else ""
        members = ", ".join(f"{x['kind']}{x['q1']}-{x['q2']}" for x in g)
        print(f"  ({len(g)}){tag}: {members}")
    return len(groups), len(pairs)


def main():
    for d in (3, 5):
        analyze(d)


if __name__ == "__main__":
    main()
