# Phase 2 — Realistic Surface Code on Stim (general d)

This branch (`1_realisticSurfaceCode`) migrates the simulation backend from
PennyLane (state-vector, d=3 hardcoded, sequential stabilizer schedule) to
**Stim** (Clifford stabilizer simulator) in order to:

1. Use the **constant-depth parallel stabilizer schedule** that surface codes
   are actually run with (4 CNOT timesteps per round, all ancillas in parallel),
   instead of the toy sequential schedule of Phase 1.
2. **Scale to general d** (d=3, 5, 7, …), which is impossible with state-vector
   simulation (d=5 → 49 qubits → 9 PB of amplitudes).
3. Re-run the Phase 1 analyses (sequential long-sequence, spatial bag-of-shots)
   on the realistic circuit.
4. Extend spatial analysis to **d·k round** windows (k ∈ {1,2,3}, no decoding).

## Why Stim

Our entire system is **Clifford + Pauli faults + computational-basis measurement**,
which is exactly Stim's exact (non-approximate) domain. Stim gives:
- correct-by-construction hook-error-aware schedule for any d
- ~1000× faster sampling than state-vector
- built-in detection-event sampler (matches our `(rounds, 8)` syndrome format)
- detector error model (DEM) export → future decoding / AlphaQubit-style work

## Phase 1 vs Phase 2

| | Phase 0 (frozen) | Phase 1 / Phase 2 (this branch) |
|---|---|---|
| Branches | `0_naiveSurfaceCode/{baseline,sequential,spatial,spatial-prebag}` | `1_realisticSurfaceCode` |
| Backend | PennyLane state-vector | Stim |
| Schedule | sequential per-stabilizer | constant-depth 4-step parallel |
| Distance | d=3 hardcoded | general d |
| Status | validated toy results (fail / GRU 95.3% / LogReg 93.9%) | in progress |

> Naming note: the original docs called the naive line "Phase 1" and the
> realistic line "Phase 2". After the branch reorg these are the
> `0_naiveSurfaceCode/*` (frozen) and `1_realisticSurfaceCode` (active)
> namespaces; "Phase 2" below refers to the work on this branch.

## Confirmed: Stim schedule (d=3, rounds=1)

```
R  <all 17 qubits>
H  <X-ancillas: 2 11 16 25>
CX <tick 1>   ┐
CX <tick 2>   │  4-step parallel schedule
CX <tick 3>   │  (all ancillas simultaneously, one corner each)
CX <tick 4>   ┘
H  <X-ancillas>
MR <8 ancillas>
DETECTOR ...   # 24 detectors for rounds=3 = 3 rounds × 8 stabilizers
```

For d=3 we read the **raw ancilla MR records** (`8·R` of them) and reshape to the
existing **`(R, 8)` syndrome format** — the downstream analysis pipeline
(`bag_classifier.py`, training/sweep/confusion scripts) attaches with only a
reshape. We bypass Stim's `DETECTOR`/`OBSERVABLE` layer for the syndrome tensor
because Stim's detector set is basis-asymmetric (round-0 X-stabs random; see
[`PHASE2_DESIGN.md`](PHASE2_DESIGN.md) §2.3–2.4). The detector/DEM path is kept
only for optional future decoding.

## Roadmap

- [x] **2-0** Inspect Stim layout / CNOT schedule / measurement mapping — **done**.
      Empirically verified d=3/5/7 layout, 4-tick parallel schedule, CNOT
      directions, `(R, d²−1)` reshape, and the round-0 projection difference vs
      Phase 0. Full write-up: [`PHASE2_DESIGN.md`](PHASE2_DESIGN.md).
- [x] **2-3** General-d surface code — **done**.
      First-principles construction in `src/backend_stim/surface_code.py`
      (`RotatedSurfaceCode(d)`), constant 4-tick depth for all d, validated
      bit-for-bit against Stim (ancilla set + schedule) plus distance and
      logical-error-rate checks for d=3/5/7. See
      [`SURFACE_CODE_IMPL.md`](SURFACE_CODE_IMPL.md);
      run `python scripts/validate_surface_code.py`.
- [x] **2-1** Fault injection — **done** (deterministic). `build_circuit(injections=…)`
      applies single-qubit data Paulis (pre-round) and 2-qubit post-CNOT Paulis
      (hook errors). Stochastic `PAULI_CHANNEL_2` p_bg/p_high sampling: still to add.
- [ ] **2-viz** Web visualization of how the code works — **done** (this milestone):
      `viz/surface_code.html` (self-contained), data-faithful to the simulator.
- [ ] **2-2** Reproduce single-fault enumeration in Stim; compare **structure** to
      Phase 0's 387 atoms (**sanity check** — structure expected to differ)
- [ ] **2-4** Generate spatial bags on Stim; re-run `bag_classifier`
- [ ] **2-5** Extend spatial to d·k rounds (k=1,2,3), no decoder; accuracy vs (N, k, d)
- [ ] **2-6** Re-run sequential pipeline on Stim
