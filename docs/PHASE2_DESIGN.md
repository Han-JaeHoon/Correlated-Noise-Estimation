# Phase 2 — Design (Step 0: research + empirical grounding)

> 한국어: [`PHASE2_DESIGN_kor.md`](PHASE2_DESIGN_kor.md)

Status: **design fixed, no backend code yet.** This document is the output of
roadmap item **2-0**. It pins down the realistic-circuit design before any
`src/backend_stim/` code is written, grounded in (a) prior work on
constant-depth surface-code scheduling and (b) direct inspection of the circuits
Stim generates. Companion roadmap: [`PHASE2_PLAN.md`](PHASE2_PLAN.md).

---

## 1. Prior work — the constant-depth schedule and hook errors

The naive Phase 0 simulator measured stabilizers **one at a time** (gate depth
∝ number of stabilizers ∝ `d²`). That is not how surface codes are run. The
standard is a **constant-depth** cycle: every ancilla measures its plaquette in
the same **4 CNOT time-steps**, all ancillas in parallel, regardless of `d`.

Two constraints fix the CNOT order (Tomita & Svore 2014; Fowler et al. 2012):

1. **No data-qubit collision** — within one time-step, a data qubit may be the
   partner of at most one CNOT.
2. **Hook-error orientation** — a single fault on an ancilla *mid-cycle* can
   propagate to **two** data qubits (a "hook" error). The CNOT order must orient
   these correlated pairs **along** the code's high-distance axis so they do not
   shorten the effective distance. The textbook solution schedules X-stabilizer
   CNOTs in an **N (Z) shape** and Z-stabilizer CNOTs in the transposed shape, so
   X-hooks and Z-hooks point along orthogonal (harmless) directions.

We do **not** re-derive this schedule. Stim's `Circuit.generated(...)` already
emits the correct hook-error-aware constant-depth schedule for any `d`, and that
is the canonical reference we build on (Gidney 2021, *Stim*).

Note for our problem: hook errors are normally a *liability* to minimize. Here
they are part of the **signal** — a mid-cycle fault after a specific CNOT
produces a structured two-qubit residual whose syndrome footprint is exactly
what we are trying to read back to identify the CNOT. The parallel schedule is
therefore not just "more realistic"; it changes the fault→syndrome map relative
to Phase 0, so Phase 0's atom table / ambiguity groups must be **recomputed**,
not assumed.

References:
- Tomita & Svore, "Low-distance surface codes under realistic quantum noise", PRA **90**, 062320 (2014). https://arxiv.org/abs/1404.3747
- Gidney, "Stim: a fast stabilizer circuit simulator", Quantum **5**, 497 (2021). https://arxiv.org/abs/2103.02202
- Background on hook errors / scheduling: Dennis–Kitaev–Landahl–Preskill (2002); Fowler et al., PRA **86**, 032324 (2012).

---

## 2. Empirical grounding — what Stim actually generates

Measured directly from `stim==1.16.0`,
`Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=R)`:

### 2.1 Per-round structure (verified d=3)

```
R   <all qubits>                      # reset to |0...0>
H   <X-ancillas>
CX  <tick 1>   ┐  6 pairs (d=3)
CX  <tick 2>   │  constant-depth
CX  <tick 3>   │  4-step parallel schedule
CX  <tick 4>   ┘
H   <X-ancillas>
MR  <all ancillas>                    # measure + reset
```

### 2.2 Layout scales as expected (measured d = 3, 5, 7)

| d | data (`d²`) | ancilla (`d²−1`) | X-anc | Z-anc | directed CX / round | ticks |
|---|---|---|---|---|---|---|
| 3 | 9  | 8  | 4  | 4  | **24** | 4 |
| 5 | 25 | 24 | 12 | 12 | **80** | 4 |
| 7 | 49 | 48 | 24 | 24 | **168** | 4 |

- **Data qubits** sit at odd–odd coordinates; **ancillas** at face centers, split
  X/Z in a checkerboard, `(d²−1)/2` each.
- **CNOT direction is type-dependent** (confirmed): X-stabilizer CNOTs use the
  **ancilla as control** (post-H); Z-stabilizer CNOTs use the **data as control**
  (ancilla is target). Direction matters for where a post-CNOT 2-qubit Pauli
  lands — exactly the Phase 0 "direction-matched injection" assumption, now
  realized on the parallel schedule.
- For `d=3` the count is **24 directed CNOTs/round — identical to Phase 0's 24**,
  so the d=3 class set lines up 1:1 for the sanity check (2-2), even though the
  *timing* (and thus the syndrome map) differs.

### 2.3 Key difference from Phase 0: round 0 is a projection round

Stim's `memory_z` starts from the **physical** state `|0…0>`, not the logical
codeword `|0_L>`. Consequently the **X-stabilizers are random in round 0** (the
state is not yet their +1 eigenstate); the first cycle *projects* into a random
codeword and stabilizers become deterministic from round 1 on. Phase 0 instead
prepared `|0_L>` and had deterministic syndromes from round 0.

**Design consequence:** we treat **round 0 as a projection/reference round** and
analyze rounds ≥ 1 (or use detection events = XOR of consecutive rounds). This is
the same "drop round 0" convention Phase 0 already used in §15/§17, so the
pipelines stay compatible.

### 2.4 Syndrome extraction: raw MR records, not Stim's detectors

`num_measurements = 8·R + d²` for `memory_z` (the `8·R` ancilla MR records plus
`d²` final data measurements). Sampling measurements directly and slicing the
first `(d²−1)·R` records reshapes cleanly to **`(R, d²−1)`** — the Phase 0 raw
syndrome format. (Verified: `(shots, 4, 8)` for d=3, R=4.)

We deliberately **bypass Stim's `DETECTOR`/`OBSERVABLE` layer** for the syndrome
tensor: Stim's detector set is basis-asymmetric (32 detectors for d=3/R=4, with
Z-only boundary rounds), whereas our analyses want the **uniform `(R, d²−1)` raw
ancilla syndrome** every round. We keep the detector/DEM path available only for
optional future decoding work.

---

## 3. Design decisions

### 3.1 Backend strategy — leverage, don't re-derive
`src/backend_stim/circuit.py` wraps `stim.Circuit.generated(...)` as the schedule
backbone and exposes a structured layout: data/ancilla ids, `(x,y)` coords, X/Z
type, and a **canonical per-round CNOT enumeration** `[(tick, control, target), …]`.
This enumeration is the index set of fault-injection sites (24 for d=3, …).

### 3.2 Fault model — match Phase 0 on the new schedule
`src/backend_stim/fault_inject.py`: given a target CNOT (by its index in the §3.1
enumeration), a 2-qubit Pauli (the 15 non-identity `{I,X,Y,Z}²∖{II}`), and a set
of rounds, produce a modified circuit that applies that Pauli to `(control, target)`
**immediately after** that CNOT. Stochastic mode injects `PAULI_CHANNEL_2` with
mass `p/15` on each non-identity Pauli — reproducing Phase 0's "uniform 15-Pauli"
`p_bg`/`p_high` model. Implemented as circuit surgery on the flattened circuit.

### 3.3 Measurement modes — both selectable, **no-reset is the data default**
- **reset (MR)** — Stim's native cycle; ancilla measured and re-initialized.
- **no-reset (M)** — replace `MR` with `M`; ancilla state carries across rounds
  (the "differential" signature Phase 0 mode 2 used).

Both are exposed as a flag. **Data generation defaults to no-reset** (per project
decision); reset is available for comparison.

### 3.4 Syndrome tensor — `(R, d²−1)`, round 0 as reference
Sample MR/M records → reshape `(R, d²−1)`. Provide both raw and detection-event
(`d_t = s_t ⊕ s_{t−1}`) views. Downstream `bag_classifier.py` and the sequential
classifier attach by reshape, exactly as in Phase 0.

### 3.5 Spatial `d·k`-round window (roadmap 2-5)
One spatial "shot" becomes **`d·k` rounds** for `k ∈ {1,2,3}` (no decoding):
d=3 → 3/6/9 rounds, d=5 → 5/10/15. Accuracy is then measured over **`(N, k, d)`**
— directly answering the "few long shots vs many short shots" information
trade-off, and watching the mixture vocabulary grow as `2^{(d²−1)·d·k}`.

---

## 4. Module plan (Step 1+)

```
src/backend_stim/
  circuit.py        # generated-schedule wrapper + layout + per-round CNOT enumeration (2-3)
  fault_inject.py   # deterministic / stochastic Pauli injection after a chosen CNOT (2-1)
  syndrome.py       # measurement sampling → (R, d²−1) raw + detection-event tensors (2-2)
```

Downstream (`bag_classifier.py`, sweep/confusion/sequential scripts) is reused
unchanged behind a reshape adapter.

## 5. Sanity check plan (roadmap 2-2)

1. Build the d=3 single-fault enumeration on the Stim schedule (24 CNOT × 15 Pauli
   = 360 + idle-data atoms), reset & no-reset.
2. Compare the **collision structure** to Phase 0's 387-atom table: silent class,
   per-CNOT Pauli uniqueness, ambiguity groups. Expectation: **structure differs**
   (parallel timing changes propagation) — the goal is to characterize the new
   structure, not to reproduce Phase 0 bit-for-bit.
3. Cross-check that round 0 behaves as the projection round (§2.3).

## 6. Open decisions (small, can be made at Step 1)

- Memory basis: `memory_z` vs running both `z` and `x` (X-faults vs Z-faults
  visibility). Default `memory_z`; add `memory_x` if X-detectability needs it.
- Fault-rounds convention for the deterministic enumeration: inject at round 1
  (first deterministic round) rather than round 0 (projection), to keep the
  reference clean.
- Whether the "class label" for large d stays per-CNOT (24/80/168 classes) or is
  coarsened — deferred until the d=3 sanity check shows the new ambiguity graph.
