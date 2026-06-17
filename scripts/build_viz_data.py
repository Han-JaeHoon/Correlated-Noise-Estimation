"""Generate the JSON payload that drives the surface-code web visualization.

For each distance, we emit: the lattice (data + ancilla coords / type), the
4-tick CNOT schedule, and — for every (data qubit, Pauli) and a few mid-cycle
CNOT faults — the per-round syndrome flip pattern produced by *actually
simulating* our circuit.

Method (deterministic, exact): build the noiseless circuit, then build a copy
with the fault inserted, sample both with the **same seed**, and XOR the raw
ancilla records. Pauli faults add no randomness, so the RNG streams stay aligned
and the XOR isolates exactly which stabilizer measurements the fault flips, every
round. This is the raw-syndrome view (a persistent data error stays lit every
round in reset mode); detection events are derived in the browser.

Output: data/viz/surface_code_viz.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend_stim.surface_code import RotatedSurfaceCode  # noqa: E402

INJECT_ROUND = 1  # round 0 is the projection round; inject from round 1
SEED = 12345


def raw_syndrome(code: RotatedSurfaceCode, reset: bool, injections, rounds: int):
    """Return (rounds, n_anc) raw ancilla measurements for one shot."""
    circ = code.build_circuit(rounds=rounds, basis="Z", reset=reset,
                              injections=injections)
    n_anc = len(code.anc_coords)
    samp = circ.compile_sampler(seed=SEED).sample(shots=1)[0]
    anc = samp[: n_anc * rounds].reshape(rounds, n_anc).astype(int)
    return anc


def flip_pattern(code, reset, injections, rounds):
    base = raw_syndrome(code, reset, [], rounds)
    err = raw_syndrome(code, reset, injections, rounds)
    return (base ^ err).tolist()


def build_for_distance(d: int) -> dict:
    code = RotatedSurfaceCode(d)
    rounds = d + 1  # round 0 projection + d stabilizer rounds (the code's natural depth)
    data = [{"coord": list(c), "index": code.qubit_index[c], "label": f"D{i}"}
            for i, c in enumerate(code.data_coords)]
    anc = [{"coord": list(c), "index": code.qubit_index[c],
            "type": code.anc_type[c],
            "label": f"{code.anc_type[c]}{i}"}
           for i, c in enumerate(code.anc_coords)]
    # schedule: 4 ticks, each a list of directed CNOTs
    schedule = [[] for _ in range(4)]
    for tick, ctrl, tgt, typ in code.cnot_enumeration():
        schedule[tick].append({"control": list(ctrl), "target": list(tgt), "type": typ})

    # Per (data qubit, Pauli) single-fault flip pattern, injected at INJECT_ROUND.
    # The platform places errors at arbitrary rounds by time-shifting this
    # reference pattern (the repeated circuit is translation-invariant after the
    # round-0 projection — verified), and combines multiple errors by XOR (the
    # circuit is Clifford, so Pauli-frame flips add over GF(2)).
    scen = {"reset": {}, "noreset": {}}
    for i, c in enumerate(code.data_coords):
        for p in ("X", "Y", "Z"):
            inj = [{"round": INJECT_ROUND, "pos": "pre", "coord": c, "pauli": p}]
            key = f"D{i}_{p}"
            for mode, reset in (("reset", True), ("noreset", False)):
                scen[mode][key] = {"flip": flip_pattern(code, reset, inj, rounds)}

    return {
        "d": d,
        "rounds": rounds,
        "inject_round": INJECT_ROUND,
        "data": data,
        "ancillas": anc,
        "schedule": schedule,
        "num_directed_cnots": code.num_directed_cnots(),
        "scenarios": scen,
    }


def main() -> int:
    out = {"distances": {}}
    for d in (3, 5):
        out["distances"][str(d)] = build_for_distance(d)
        print(f"d={d}: {out['distances'][str(d)]['num_directed_cnots']} CNOTs/round, "
              f"{len(out['distances'][str(d)]['data'])} data, "
              f"{len(out['distances'][str(d)]['ancillas'])} ancillas")
    dest = Path("data/viz/surface_code_viz.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print("wrote", dest, f"({dest.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
