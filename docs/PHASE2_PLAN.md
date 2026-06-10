# Phase 2 — Realistic Surface Code on Stim (general d)

This branch (`stim-realistic-general-d`) migrates the simulation backend from
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

| | Phase 1 (frozen) | Phase 2 (this branch) |
|---|---|---|
| Branches | main, sequential-data-analysis, spatial-data-analysis, spatial-data-analysis-phase | stim-realistic-general-d |
| Backend | PennyLane state-vector | Stim |
| Schedule | sequential per-stabilizer | constant-depth 4-step parallel |
| Distance | d=3 hardcoded | general d |
| Status | validated toy results (fail / GRU 95.3% / LogReg 93.9%) | in progress |

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

d=3 produces **24 detectors over 3 rounds = our existing (3, 8) format** — the
downstream analysis pipeline (`bag_classifier.py`, training/sweep/confusion
scripts) attaches with only a reshape.

## Roadmap

- [ ] **2-0** Inspect Stim layout / CNOT schedule / detector mapping  ← done (this doc)
- [ ] **2-1** Fault injection: insert a Pauli after a specific CNOT (`src/backend_stim/fault_inject.py`)
- [ ] **2-2** Reproduce single-fault enumeration in Stim; compare structure to Phase 1's 387 atoms (**sanity check**)
- [ ] **2-3** Parameterize general d (`src/backend_stim/circuit.py`)
- [ ] **2-4** Generate spatial bags on Stim; re-run `bag_classifier`
- [ ] **2-5** Extend spatial to d·k rounds (k=1,2,3), no decoder; accuracy vs (N, k)
- [ ] **2-6** Re-run sequential pipeline on Stim
