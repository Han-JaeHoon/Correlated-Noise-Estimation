# Correlated errors in superconducting surface-code devices — a survey for noise modeling

> 한국어: [`CORRELATED_NOISE_SURVEY_kor.md`](CORRELATED_NOISE_SURVEY_kor.md)

**Purpose.** This project aims to *detect and classify the correlation between
qubits* of a real device from syndrome data. This document surveys what
correlated errors actually occur in superconducting surface-code hardware, so the
noise model we sample data from is physically grounded rather than ad hoc. Focus:
superconducting qubits (Google / IBM), modeling-actionable detail.

## TL;DR

- Real correlated errors split into ~5 mechanisms (table below).
- The **dominant and most damaging** correlated error is **stray `ZZ` during the
  two-qubit (CZ/CNOT) gate**, coupling **data ↔ measure(ancilla)** qubits. This is
  essentially what the project's earlier "inject a 2-qubit Pauli right after a
  CNOT" model already captured.
- Correlated errors are **detectable from syndrome data**: in real experiments
  they are diagnosed as off-diagonal structure in the **detector–detector
  correlation matrix (`pij`)** and represented as **hyperedges** in the detector
  error model (DEM). Single-detector statistics (detection probability) do *not*
  reveal correlations — pairwise/sequence analysis is required. This is exactly
  the signal a correlation classifier would learn.

## Mechanisms

| Mechanism | Coupled pairs | Pauli / channel | Spatial · temporal | Magnitude (numbers) |
|---|---|---|---|---|
| **Stray `ZZ` during CZ/CNOT** (dominant) | **data ↔ ancilla** (gate pair), neighbors | correlated `ZZ` + swap-like | nearest-neighbor, at gate time (single-round) | data–ancilla `p_ZZ ≈ 10⁻³`, data–data `≈ 10⁻⁴` (sim.); leakage + stray ≈ **17%** of Google's budget (meas.) |
| **Always-on residual `ZZ`** | data–data (NNN), data–ancilla | correlated `Z` (dephasing), `p ≈ sin²(J·t)` | always-on, persistent | suppressed to **< 50 kHz** (ideally < 10 kHz) with tunable couplers; IBM example ~10 kHz |
| **Readout crosstalk** | **ancilla–ancilla** (shared / frequency-muxed readout), data–ancilla | correlated **measurement bit-flip** + dephasing | nearest-neighbor, at measurement | readout is the largest local error (weight **5p** in SI1000) |
| **Leakage (to \|2⟩) and its spread** | data → measure qubit; neighbor clusters | non-Pauli, multi-qubit correlated | local cluster, persists over rounds | within Google's ~17%; disabling DQLR drops d=5 Λ by ~35% |
| **Cosmic-ray / quasiparticle bursts** | chip-wide (many qubits at once) | simultaneous `T1` collapse (energy relaxation) | ~30-qubit neighborhood to chip-wide; rare but lasting | **~once per hour**; decay time constant **~400 µs** |

### Notes per mechanism

- **Stray ZZ during gates** is both the largest and the most *harmful* correlated
  error for the surface code, because it couples a data qubit to its measure
  qubit *while the ancilla is mid-circuit* (in superposition), so the error
  propagates and is detectable (it is **not** silent). A representative
  simulation study finds gate-based data–ancilla crosstalk degrades the threshold
  more than any other crosstalk type (0.74% → 0.63%).
- **Always-on ZZ** produces correlated *dephasing* (Z) between qubits sharing a
  coupler; it is continuously present and is the textbook "ZZ crosstalk." It is
  actively engineered down to the 10s-of-kHz range with tunable couplers.
- **Readout crosstalk** is the natural origin of **ancilla–ancilla** correlation:
  frequency-multiplexed / shared-resonator readout means one qubit's readout can
  off-resonantly drive a neighbor's resonator, and an adjacent qubit's state can
  bias a readout. Modeled as correlated measurement bit-flips. Mitigated by
  detuning nearby resonators.
- **Leakage** to non-computational states (\|2⟩) creates multi-qubit correlated
  errors that are *invisible to single-detector metrics*; Google swaps data-qubit
  leakage to measure qubits (DQLR). Expected to grow in relative importance as
  local errors shrink.
- **Cosmic-ray / QP bursts** are rare, catastrophic, spatially-extended
  correlated events (simultaneous energy relaxation across many qubits). They set
  an error floor; they are temporally correlated and out-of-model for standard
  i.i.d. noise.

### Baseline (uncorrelated) reference numbers

From Google's 105-qubit Willow surface-code memory: mean `T₁ ≈ 68 µs`,
`T₂,CPMG ≈ 89 µs`, per-round detection probability ≈ 8%, logical error per cycle
≈ `1.4×10⁻³` at d=7 (NN decoder). The **SI1000** circuit noise model weights
errors non-uniformly: measurement ≈ **5p**, two-qubit gate ≈ **p**, single-qubit
gate / idle ≈ **p/10** (with `p` ~ `10⁻³` for a good device).

## Detectability from syndrome data

- A DEM is an **error hypergraph**: each stochastic mechanism is a hyperedge
  connecting the detectors it triggers. Correlated mechanisms = hyperedges
  spanning detectors that single faults would not link.
- Experiments fit these via the **`pij` method** — a Pearson correlation matrix
  between detectors (space + time indices). Off-diagonal structure reveals
  correlated detector pairs and "missing hyperedge" signatures.
- **Implication for this project:** "which qubit pair is correlated" is exactly
  what `pij` analysis extracts. A correlation classifier trained on syndrome bags
  / sequences is effectively learning a `pij`-style correlation detector. Because
  single-detector statistics miss correlations, pairwise/sequence-level features
  are necessary — which is why the bag-of-shots and sequence pipelines are the
  right tools.

## Implications for this project's noise model

1. **Keep data ↔ ancilla pairs — they are the most important, not the least.**
   The dominant real correlated error is gate-based data–ancilla `ZZ`. Earlier we
   considered dropping these (because a `Z` injected *at measurement* on a Z-type
   ancilla is silent); but the realistic correlation is `ZZ` *during* the gate,
   which propagates and is detectable. → the distance-≤2 adjacency set (44 pairs
   for d=3, including the 24 data–ancilla √2 pairs) is well justified.
2. **The earlier post-CNOT injection was the right mechanism, not a hack.**
   "Inject a 2-qubit Pauli right after a CNOT" = stray `ZZ`/swap during the gate.
   The top-down correlation model should still center on gate-adjacent and
   nearest-neighbor pairs.
3. **Pauli structure:** correlated `Z` on data (dephasing → X-stabilizers); on the
   ancilla, `ZZ` during the gate (propagates, detectable) for data–ancilla pairs,
   and correlated readout bit-flips for ancilla–ancilla pairs.
4. **Magnitudes:** baseline single-qubit `p ≈ 10⁻³` (measurement ~`10⁻²`);
   correlated strength `c ≈ 10⁻³` (data–ancilla) down to `10⁻⁴` (data–data) — i.e.
   `c` is ~0.1–1× the single-qubit rate; sweep this range.
5. **"None" class** = standard i.i.d. circuit noise (all pairwise correlations
   zero).
6. **Optional later:** leakage and cosmic-ray bursts are real but harder to model
   (non-Pauli / temporally-correlated / out-of-model); defer to a later stage.

## Sources

- [Quantum error correction below the surface code threshold — Google "Willow" (arXiv 2408.13687 / Nature s41586-024-08449-y)](https://arxiv.org/html/2408.13687v1) — leakage + stray `ZZ` ≈ 17% of error budget, swap-like errors, burst rate ~once/hour with ~400 µs decay over ~30 qubits, `T₁`/`T₂`, detection probabilities.
- [Surface Code Error Correction with Crosstalk Noise (arXiv 2503.04642)](https://arxiv.org/html/2503.04642) — data–ancilla vs data–data, always-on vs gate-based `ZZ`; `p_ZZ` 10⁻³/10⁻⁴; `J = 10 kHz`; Pauli-twirl `sin²(Jt)`; threshold 0.74% → 0.63%.
- [Scalable Method for Eliminating Residual `ZZ` Interaction (arXiv 2111.13292)](https://arxiv.org/pdf/2111.13292) — residual `ZZ` suppression (< 10–50 kHz).
- [Cosmic-ray-induced correlated errors in superconducting qubit array (arXiv 2402.04245 / Nature Commun. s41467-025-59778-z)](https://arxiv.org/abs/2402.04245) — muon/γ quasiparticle bursts, simultaneous multi-qubit relaxation.
- [Resolving catastrophic error bursts from cosmic rays (Nature Physics s41567-021-01432-8)](https://www.nature.com/articles/s41567-021-01432-8) — chip-wide correlated bursts.
- [Learning to Decode the Surface Code … (arXiv 2310.05900)](https://arxiv.org/pdf/2310.05900) and the `pij` / DEM-hyperedge methodology; **SI1000** noise model (measurement weight 5p).

> Attribution note: Google's ~17% correlated budget, burst statistics, and
> `T₁`/`T₂` are *measured*. The `J = 10 kHz` and `p_ZZ` 10⁻³/10⁻⁴ figures are
> *representative simulation parameters* (IBM-like) from the crosstalk study, not
> a single measured device value — use them as order-of-magnitude guidance.
