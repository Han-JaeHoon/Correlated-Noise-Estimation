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

ROUNDS = 4
INJECT_ROUND = 1  # round 0 is the projection round; inject from round 1
SEED = 12345


def raw_syndrome(code: RotatedSurfaceCode, reset: bool, injections):
    """Return (rounds, n_anc) raw ancilla measurements for one shot."""
    circ = code.build_circuit(rounds=ROUNDS, basis="Z", reset=reset,
                              injections=injections)
    n_anc = len(code.anc_coords)
    samp = circ.compile_sampler(seed=SEED).sample(shots=1)[0]
    anc = samp[: n_anc * ROUNDS].reshape(ROUNDS, n_anc).astype(int)
    return anc


def flip_pattern(code, reset, injections):
    base = raw_syndrome(code, reset, [])
    err = raw_syndrome(code, reset, injections)
    return (base ^ err).tolist()


def membership_syndrome(code, coord, pauli):
    """Which ancillas a single data Pauli anticommutes with (clean static view).

    Z data error -> X stabilizers on that qubit; X data error -> Z stabilizers;
    Y -> both.
    """
    lit = []
    for k, a in enumerate(code.anc_coords):
        touches = any(dc == coord for _, dc in code.schedule[a])
        if not touches:
            continue
        t = code.anc_type[a]
        anti = (pauli == "Y") or (pauli == "Z" and t == "X") or (pauli == "X" and t == "Z")
        if anti:
            lit.append(k)
    return lit


def build_for_distance(d: int) -> dict:
    code = RotatedSurfaceCode(d)
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

    scen = {"reset": {}, "noreset": {}}
    for i, c in enumerate(code.data_coords):
        for p in ("X", "Y", "Z"):
            inj = [{"round": INJECT_ROUND, "pos": "pre", "coord": c, "pauli": p}]
            key = f"D{i}_{p}"
            for mode, reset in (("reset", True), ("noreset", False)):
                scen[mode][key] = {
                    "flip": flip_pattern(code, reset, inj),
                    "membership": membership_syndrome(code, c, p),
                }

    # a few mid-cycle CNOT faults (hook-error demonstrations)
    cnot_faults = []
    enum = code.cnot_enumeration()
    # pick a handful of interior CNOTs and inject a 2-qubit Pauli after them
    picks = enum[: min(6, len(enum))]
    for tick, ctrl, tgt, typ in picks:
        for pauli in ("XZ", "ZZ"):
            inj = [{"round": INJECT_ROUND, "pos": "post_cx", "tick": tick,
                    "control": ctrl, "target": tgt, "pauli": pauli}]
            cnot_faults.append({
                "tick": tick, "control": list(ctrl), "target": list(tgt),
                "type": typ, "pauli": pauli,
                "flip_reset": flip_pattern(code, True, inj),
                "flip_noreset": flip_pattern(code, False, inj),
            })

    return {
        "d": d,
        "rounds": ROUNDS,
        "inject_round": INJECT_ROUND,
        "data": data,
        "ancillas": anc,
        "schedule": schedule,
        "num_directed_cnots": code.num_directed_cnots(),
        "scenarios": scen,
        "cnot_faults": cnot_faults,
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
