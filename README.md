# Correlated-Noise-Estimation — Spatial Analysis Branch

A research codebase for analyzing whether the **location of a faulty CNOT gate** acting as a dominant noise source in a `d=3` rotated surface code can be **identified from syndrome measurement sequences alone**, and for designing a learning model on top of those findings.

(See [`README_kor.md`](README_kor.md) for the Korean version.)

> **Repository layout — three branches**
>
> | Branch | Scope |
> |---|---|
> | **`main`** | Problem statement + simulator + baseline single-round 216-case sweep (§§1–11) + foundational 387-atom deterministic enumeration (`data/analysis/fault_enumeration/`) |
> | **`sequential-data-analysis`** | Time-cumulative line: long-sequence syndrome streams, R1/R2/R3b decoders, §15 separability, §16 Task #6 classifier (R2/GRU 95.3 %, R3b/GRU 91.2 %), §17/§18 enumeration-pattern analyses |
> | **`spatial-data-analysis`** | Statistical-bag line: 387 atoms as mixture components, §12 pairwise-TV ambiguity (24/24 distinguishable), §13 ML training dataset, set-classifier learning (next) |
>
> §§1–11 describe the baseline analysis (inherited from `main`). §12 (Spatial mixture ambiguity) and §13 (ML dataset) are the spatial branch's contributions.

---

## 1. Research Goal

### 1.1 Big picture
- `d=3` rotated surface code (17 qubits = 9 data + 8 ancilla)
- Repeated stabilizer-measurement circuit
- **Assumption**: exactly one of the 24 directed CNOTs is the fault source, and a fault recurs at that same location every round (temporally correlated)
- **Target**: from the syndrome bit sequence alone, identify which CNOT is the faulty one (24-class classification)

### 1.2 Nuisance variables we ignore
- The specific rounds where the fault fires
- The Pauli type of the fault (one of XX, XY, ..., ZZ)

### 1.3 Motivation
At toy `d=3` a lookup table works, but the table cost explodes with code distance. The end goal is a single trained model that can be queried at inference time.

---

## 2. Analysis Workflow So Far (as of this writing)

| Step | Question | Result |
|---|---|---|
| (a) | XZ fault only, can `n_rounds=1` separate the 24 CNOTs? | 8 unique / 24, 11 silent |
| (b) | What about `n_rounds=2`? | **24 unique (full identification)** — threshold |
| (c) | `n_rounds≥3`? | Same. Extra rounds just repeat the same info |
| (d) | What about other Pauli pairs? | XY/XZ/YX/YY/ZY = 24 unique; **ZX/YZ = partial; XX = 17; ZZ = 11** |
| (e) | Multi-round XZ at [0,1] vs single round? | Round-0-visible CNOTs are distinguishable; silent ones are not |
| (f) | **What if the Pauli type is unknown? Cross-Pauli ambiguity?** | **216 cases → 123 unique, 38 ambiguous, 0/24 uniquely identifiable** |
| (g) | Does (f) break with more rounds or different mode? | **Invariant** — same under n_rounds 2/3/4 and reset/no-reset |
| (h) | User's key question: do different (CNOT, Pauli) pairs share syndromes? | **72 pairs (cross-CNOT cross-Pauli)** — fundamentally ambiguous |

### 2.1 Headline conclusion

> **With single-round faults and mid-measurement syndromes only, and with the Pauli type unknown, 24-class CNOT identification is information-theoretically impossible.** The 72 collisions are invariant under round count and measurement mode.

### 2.2 Precise statement (after user correction)
- "Same syndrome at round 0 → identical forever" is **wrong** (round 1 can split them).
- Correct statement: **"Same syndrome string through round R = identical (data + ancilla) state at the end of round R → identical syndromes for all subsequent rounds."**

---

## 3. Directory Layout

```text
.
├── README.md                           # this document
├── README_kor.md                       # Korean version
├── requirements.txt                    # pennylane, numpy, matplotlib, stim
├── src/                                # forward-simulator modules
├── scripts/                            # data generation + analysis scripts
├── exercise/                           # early prototyping notebooks
├── notebooks/                          # debug notebooks
└── data/
    ├── mid_measure_reset/              # forward-sim data (mode 1)
    ├── mid_measure_no_reset/           # forward-sim data (mode 2)
    ├── no_mid_measure_final_sample/    # forward-sim data (mode 3)
    ├── no_mid_measure_final_probs/     # forward-sim data (mode 4)
    ├── analysis/
    │   ├── 1_per_pauli_degeneracy/         # §6.1 — baseline single-Pauli
    │   ├── 2_pauli_sweep_summary/          # §6.2
    │   ├── 3_cross_pauli_conflict/         # §6.3
    │   ├── 4_cross_pauli_per_pauli_view/   # §6.4
    │   ├── 5_cross_pauli_pair_collisions/  # §6.5 — 72 pairs
    │   ├── fault_enumeration/              # foundation: 387 atoms
    │   └── spatial_mixture/                # §12 — pairwise TV per regime
    └── ml_dataset/                         # §13 — per-shot training pools
        └── noreset_d3_phigh0.1_pbg0.01_n1000/
```

---

## 4. `src/` — Forward Simulator Modules

| File | Role |
|---|---|
| `config.py` | Default experiment settings (n_rounds, shots, fault location, 4 measurement situations) |
| `surface_code_layout.py` | Data/ancilla qubit indexing, Z/X stabilizer definitions, `enumerate_stabilizer_cnots()` |
| `logical_state.py` | `|0_L⟩` preparation circuit |
| `fault_schedule.py` | `(control, target, round)` fault schedule builder, round-subset enumerator, occurrence-vector helper |
| `stabilizer_circuit.py` | Z/X stabilizer rounds, **direction-matched Pauli injection right after a CNOT** |
| `simulator.py` | `make_repeated_stabilizer_qnode(...)` — builds a QNode for each of the 4 measurement modes |
| `postprocess.py` | Output reshape, detection events (`syndrome[t+1] XOR syndrome[t]`), bit flattening |
| `dataset_generators.py` | Dataset row builders + duplicate-pattern finder |
| `io_utils.py` | CSV/JSON save helpers |

### Surface code layout (summary)
- Data qubits: `0..8`
- Z ancilla: `9, 10, 11, 12` (data → ancilla CNOT)
- X ancilla: `13, 14, 15, 16` (H → ancilla → data CNOT → H)
- 24 directed CNOTs per round
- Stabilizer order per round: `[Z0, Z1, Z2, Z3, X0, X1, X2, X3]`

### Four measurement modes
1. **mid_measure + reset** — standard surface-code cycle
2. **mid_measure + no reset** — ancilla collapses but is not re-initialized (closer to detection-event signature)
3. **no mid-measure, final sample** — ancilla measured only at the very end
4. **no mid-measure, final probs** — final probability distribution over 8 ancilla wires (256 outcomes)

---

## 5. `scripts/` — Forward Data-Generation Scripts

| File | Role |
|---|---|
| `debug_run.py` | Quick smoke test (single fault config, prints output) |
| `generate_fixed_cnot_patterns.py` | Fix one CNOT, sweep all round occurrence patterns (2^N cases) |
| `generate_all_cnot_single_fault.py` | Fix one round, sweep all 24 CNOT locations |
| `generate_all_cnot_round_patterns.py` | Cartesian product of the above (24 × 2^N) |
| `inspect_dataset.py` | CSV inspection |

### Shared CLI flags
- `--n-rounds`, `--control`, `--target`, `--error-types`, `--mode`, `--output-root`

### `data/<mode>/` CSV — one row per simulation
Key columns: `sample_id, mode, n_rounds, control, target, error_rounds, occurrence_vector, error_types, syndrome_bits, detection_bits, final_bits, probabilities, fault_schedule_json, raw_output_json`.

---

## 6. `scripts/` — Analysis Scripts (core of this study)

Each script saves into `data/analysis/{number}_{name}/` automatically.

### 6.1 `analyze_round1_degeneracy.py` → `1_per_pauli_degeneracy/`
- For **one fixed Pauli pair**, collect syndromes from 24 CNOTs and visualize the within-Pauli degeneracy classes.
- X-axis: 24 CNOTs in execution order. Y-axis: stabilizer ancilla × n_rounds (Z/X split, per-round separator).
- Top stripe: degeneracy class id / size.
- CLI: `--n-rounds`, `--error-types XZ`, `--error-rounds 0` (or `0,1`), `--no-reset`.

### 6.2 `sweep_pauli_pairs.py` → `2_pauli_sweep_summary/`
- Sweep all 9 Pauli pairs × 24 CNOTs and report **unique syndromes, silent CNOTs, max class size** per Pauli pair.
- Bar chart.
- CLI: `--n-rounds 2`, `--error-rounds 0`, `--no-reset`.

### 6.3 `analyze_cross_pauli_degeneracy.py` → `3_cross_pauli_conflict/`
- Collect all 216 = 24 × 9 cases.
- For each syndrome, count the **number of distinct CNOTs producing it** (= conflict size).
- 24×9 heatmap (cell = conflict size; 1 = identifiable, ≥2 = ambiguous).
- Print the conflict graph (which CNOTs collide with which others).
- List members of the top 5 largest ambiguous classes.

### 6.4 `visualize_cross_pauli_per_pauli.py` → `4_cross_pauli_per_pauli_view/`
- For each of the 9 Pauli pairs, produce a heatmap in the same format as the `1_per_pauli_degeneracy/` figures.
- **Difference**: the top-stripe class id is a **global id** computed across all 216 cases.
- If the same global id appears in two different Pauli figures → those two (CNOT, Pauli) cases share the syndrome (cross-Pauli match).
- Outputs per condition: 9 PNGs (`eXX.png` ~ `eZZ.png`) + `matches_by_global_id.csv` (sorted by global id).
- One subfolder per condition (e.g. `nrounds2_r0_noreset/`).

### 6.5 `analyze_cross_pauli_cross_cnot_pairs.py` → `5_cross_pauli_pair_collisions/`
- Enumerate all unordered pairs of cases sharing a syndrome and bucket them:
  - `same_cnot_diff_pauli` — same CNOT under different Paulis (harmless).
  - `diff_cnot_same_pauli` — within-Pauli degeneracy (resolvable if Pauli is known).
  - **`diff_cnot_diff_pauli`** — **fundamental ambiguity** (the user's question).
- 24×24 CNOT-pair matrix (cell = number of (α, β) Pauli combinations that collide).
- CSV stores only the `diff_cnot_diff_pauli` pairs.

---

## 7. `data/analysis/` — Output Folder Details

### 7.1 `1_per_pauli_degeneracy/`
Filename pattern: `nrounds{N}_e{Paulis}_r{rounds_str}_{mode}_(degeneracy.png | syndromes.csv)`.

| File | Description |
|---|---|
| `nrounds1_eXZ_r0_reset_*` | XZ fault at round 0, 1 round, reset → 8 unique / 24 |
| `nrounds2_eXZ_r0_reset_*` | XZ at [0], 2 rounds, reset → 24 unique |
| `nrounds2_eXZ_r0_noreset_*` | Same condition, no reset |
| `nrounds2_eXZ_r0-1_reset_*` | XZ at rounds [0,1], 2 rounds, reset |
| `nrounds2_eXZ_r0-1_noreset_*` | Same condition, no reset |

### 7.2 `2_pauli_sweep_summary/`
| File | Description |
|---|---|
| `pauli_sweep_nrounds2_r0_reset.{png,csv}` | Comparison of 9 Pauli pairs at n=2 (XX = 17 unique, ZZ = 11 unique, etc.) |

### 7.3 `3_cross_pauli_conflict/`
Filename: `cross_pauli_nrounds{N}_r{rounds}_{mode}.(png|csv)`.

| File | Description |
|---|---|
| `cross_pauli_nrounds2_r0_reset.*` | All 216 cases combined. 24×9 heatmap; cell = # colliding CNOTs. **All-zero silent class with 8 CNOTs, 38 ambiguous syndromes** |
| `cross_pauli_nrounds2_r0_noreset.*` | Identical result (mode-invariant) |
| `cross_pauli_nrounds3_r0_*` | Identical at n=3 (round-invariant) |

### 7.4 `4_cross_pauli_per_pauli_view/`
| Subfolder | Contents |
|---|---|
| `nrounds2_r0_noreset/` | `eXX.png` ~ `eZZ.png` (9 figures) + `matches_by_global_id.csv` |

### 7.5 `5_cross_pauli_pair_collisions/`
Filename: `cross_pauli_pairs_nrounds{N}_r{rounds}_{mode}.(png|csv)`.

| File | Description |
|---|---|
| `cross_pauli_pairs_nrounds2_r0_noreset.*` | **72 cross-Pauli cross-CNOT pairs**. 24×24 matrix |
| `cross_pauli_pairs_nrounds3_r0_noreset.*` | Same 72 pairs at n=3 (CSV diff: identical) |
| `cross_pauli_pairs_nrounds4_r0_noreset.*` | Same 72 pairs at n=4 |

Verified by CSV diff:
- n_rounds 2 vs 3: identical
- n_rounds 2 vs 4: identical

---

## 8. Headline Findings (at a glance)

```
┌─────────────────────────────────────────────────────────────────┐
│ XZ fault only, n_rounds=1: 8 unique / 24 (11 silent)            │
│ XZ fault only, n_rounds=2: 24 unique (full identification)      │
│ XZ fault only, n_rounds=3+: same as n_rounds=2 (no new info)    │
│                                                                  │
│ Across 9 Pauli pairs (n_rounds=2):                              │
│   XY, XZ, YX, YY, ZY     -> 24 unique                           │
│   YZ -> 21,  ZX -> 20    -> mild ambiguity                      │
│   XX -> 17 (4 silent),  ZZ -> 11 (4 silent)  -> worst           │
│                                                                  │
│ Cross-Pauli (216 cases, Pauli unknown):                         │
│   123 unique syndromes                                          │
│   38 ambiguous syndromes                                        │
│   0/24 CNOTs cleanly identifiable                               │
│   8-way silent class (XX silent 4 + ZZ silent 4)                │
│                                                                  │
│ Cross-Pauli cross-CNOT pair collisions:                         │
│   72 pairs (invariant under n_rounds >= 2 and reset mode)       │
│   Worst pair: CNOT00 <-> CNOT01 (5 (alpha,beta) combos collide) │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Roadmap

Strategies for handling the 72 fundamental collisions:

### 9.1 Extending the analysis (largely reuses existing code)
- **(M1) Multi-round same-CNOT fault** — set `error_rounds=[0,1,2,...]` so the fault recurs every round. Matches the user's original framing ("a specific CNOT is *mostly* the bad one"). Test whether the accumulated information breaks the 72 collisions.
- **(M2) Adding final data measurement** — combine the 9-bit (or 256-probability) `no_mid_measure_final_*` data with the syndrome. A different observable may break the collisions.
- **(M3) Move the fault round** — try `error_rounds=[N-1]` (last round). Identifiability may drop when the fault has no time to propagate.

### 9.2 Learning-model design (after the analysis above)
The model design depends on how we handle the conflict graph:

- **(L1) Soft classification** — 24-dim softmax output. On ambiguous syndromes the "ground truth" is a distribution over the conflated CNOTs. Evaluate with top-k accuracy / KL divergence.
- **(L2) Coarse labels** — merge CNOTs in each connected component of the conflict graph (24 → ~10 groups). Reduce resolution to make the problem solvable.
- **(L3) Marginalized Bayesian** — assume a prior over Pauli pairs (e.g., depolarizing channel). Compute `P(CNOT | syndrome)`. Even within a collision pair, distributions may differ enough.
- **(L4) Multi-task** — jointly estimate (CNOT, Pauli) and inspect the trade-off between Pauli-accuracy and CNOT-accuracy.

### 9.3 Candidate architectures
- Logistic regression (baseline; check if the deterministic map is even cleanly invertible)
- MLP (2–3 hidden layers)
- GNN — exploit the data/ancilla connectivity graph (valuable for `d ≥ 5`)
- Transformer — feed the round sequence directly as tokens

### 9.4 Evaluation design
- Random split (low value on deterministic data)
- **Pauli pair generalization** — train on XZ, test on ZX / YY
- Noise robustness — add syndrome bit-flip noise and measure accuracy
- Compare model performance vs the information-theoretic ceiling

---

## 10. Setup & Usage

### 10.1 Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 10.2 Forward-data generation examples
```bash
# One CNOT fixed, sweep all round patterns
python scripts/generate_fixed_cnot_patterns.py \
    --n-rounds 4 --control 0 --target 9 --error-types XZ --mode mid_reset

# 24 CNOTs x all round patterns
python scripts/generate_all_cnot_round_patterns.py \
    --n-rounds 4 --error-types XZ --mode all
```

### 10.3 Analysis-script examples
```bash
# 1. Single Pauli, within-Pauli degeneracy
python scripts/analyze_round1_degeneracy.py \
    --n-rounds 2 --error-types XZ --error-rounds 0

# 2. Compare 9 Pauli pairs
python scripts/sweep_pauli_pairs.py --n-rounds 2 --error-rounds 0

# 3. Cross-Pauli conflict (all 216 cases)
python scripts/analyze_cross_pauli_degeneracy.py --n-rounds 2 --error-rounds 0

# 4. Per-Pauli view with global ids
python scripts/visualize_cross_pauli_per_pauli.py --n-rounds 2 --error-rounds 0

# 5. Cross-Pauli cross-CNOT pair collisions
python scripts/analyze_cross_pauli_cross_cnot_pairs.py --n-rounds 2 --error-rounds 0
```

Default measurement mode is **no-reset** (cheaper experimentally, same information content). Use `--reset` to switch.

---

## 11. Stated Assumptions

Every conclusion in this study depends on the following assumptions:

1. **Single fault source**: exactly one of the 24 CNOTs is faulty at any time.
2. **Deterministic Pauli fault**: not a stochastic Pauli channel — once a fault fires, a fixed (X, Y, Z) pair is applied deterministically.
3. **Post-CNOT location**: the Pauli error is injected immediately after the CNOT gate (`stabilizer_circuit.py:inject_faults_after_cnot`).
4. **CNOT directionality**: `CNOT(c→t)` and `CNOT(t→c)` are treated as distinct locations.
5. **No measurement / reset noise**: ancilla measurement and reset are themselves perfect.

If any of these are relaxed, the analysis results (especially the 72-pair collision count) must be re-evaluated.

---

## 12. Spatial mixture ambiguity (this branch)

A different way of asking "can a faulty CNOT be identified?". Instead of one **long** syndrome stream (the sequential branch's approach), draw many **short** independent d-round shots; each shot fires the dominant fault at most a few times under a low-rate Bernoulli model, with random Pauli draws. The 387 deterministic atoms (§ "fault_enumeration foundation") become the **components of a mixture distribution** `P_k(s)` over d-round syndromes for each candidate dominant location k.

### 12.1 Model

For one shot of d rounds, location k dominant:

```
For each round r ∈ {0..d-1}:
    a fault fires at k with probability p_high
    if fired, a Pauli is drawn uniformly from k's atomic pool
        (15 for CNOT, 3 for data idle)

For each background location ℓ ≠ k and each round r:
    a fault fires with probability p_bg, uniform Pauli

shot syndrome = XOR of round-shifted atomic syndromes from all fired events
```

Implemented in [`src/spatial_mixture.py`](src/spatial_mixture.py) (`load_atoms`, `precompute_shifted_atoms`, `sample_shots`).

### 12.2 Two backends

| Backend | When used | What it computes |
|---|---|---|
| **exact** | `p_bg = 0` | Enumerates all 4 096 = 1 + 3·15 + 3·15² + 15³ (round-subset × Pauli-tuple) combinations at the dominant location. Each combination has an analytical probability, accumulated into a Counter. Distinct syndromes per location: 32 – 1024. Probability sum verified = 1.000000. **No Monte-Carlo noise.** |
| **mc** | `p_bg > 0` | Samples N_shots from the full bag-of-shots model. Bootstrap split-half noise floor for grouping threshold. |

Script: [`scripts/analyze_spatial_mixture_ambiguity.py`](scripts/analyze_spatial_mixture_ambiguity.py). Outputs land in [`data/analysis/spatial_mixture/<tag>/`](data/analysis/spatial_mixture/).

### 12.3 Headline finding — 24 / 24 distinguishable

**All 24 CNOTs produce mathematically distinct mixture distributions** under every regime tested. Pairwise total-variation distances:

| Regime (`mode, d, p_high, p_bg`) | min TV | max TV | median | # ambiguity groups |
|---|---|---|---|---|
| `reset, 3, 0.1, 0.0` (exact) | **0.104** | 0.271 | 0.243 | **24** |
| `reset, 3, 0.05, 0.0` (exact) | 0.053 | 0.143 | 0.127 | **24** |
| `reset, 3, 0.01, 0.0` (exact) | 0.011 | 0.030 | 0.026 | **24** |
| `reset, 3, 0.001, 0.0` (exact) | 0.0011 | 0.0030 | 0.0027 | **24** |
| `reset, 3, 0.05, 0.005` (MC, N=500k) | 0.070 | 0.135 | 0.124 | **24** |

Two structural observations:

- **`max TV ≈ 1 − (1 − p_high)^d`** = the non-vacuum probability mass — the natural ceiling because two locations with disjoint non-vacuum supports differ only on that mass.
- **`min TV` occurs exactly at the long-sequence R1 ambiguity-group pairs**: `{6, 7}`, `{22, 23}` (smallest, TV ≈ 0.104 at p_high = 0.1), then `{14, 15, 17}`, `{18, 19, 20}` (TV ≈ 0.153). The bag-of-shots scenario **breaks** the long-sequence structural ambiguity, but the hardest pairs are exactly the same.

### 12.4 What "distinguishable" means here

`TV(P_a, P_b) > 0` ⇔ the two distributions are **literally different** ⇒ the optimal Bayes test attains arbitrary accuracy given enough samples (Neyman–Pearson). One sample gives error rate `½(1 − TV)`; N samples give `≤ exp(−N · TV² / 2)`.

Concretely, at `p_high = 0.1`:

| TV value | Example pair | 1-shot error | N for 95 % binary |
|---|---|---|---|
| 0.104 (min) | cnot06 ↔ cnot07 | 44.8 % | ≈ 540 |
| 0.243 (median) | typical | 37.9 % | ≈ 100 |
| 0.271 (max) | cnot02 ↔ cnot11 | 36.5 % | ≈ 80 |

So "distinguishable" is an **information-theoretic** statement: no two location-conditional distributions are identical. Practical sample efficiency is a separate question that depends on `min TV` and the classifier.

---

## 13. ML training dataset (this branch)

Per-shot dataset for 24-class CNOT identification, intended to be **bagged at training time** (sample N shots per bag, label = which CNOT). Built by [`scripts/build_ml_dataset.py`](scripts/build_ml_dataset.py) from the §12 sampler.

### 13.1 First regime

| Setting | Value |
|---|---|
| mode | `noreset` |
| d (rounds per shot) | 3 |
| p_high | 0.1 |
| p_bg | 0.01 |
| N_per_class | 1 000 |
| Total shots | 24 000 |
| Split (per class) | train 800 / val 100 / test 100 |

Location: [`data/ml_dataset/noreset_d3_phigh0.1_pbg0.01_n1000/`](data/ml_dataset/noreset_d3_phigh0.1_pbg0.01_n1000/).

### 13.2 Files

| File | Shape / type | Contents |
|---|---|---|
| `shots.npy` | `(24 000, 3, 8) int8` | raw syndrome bits |
| `labels.npy` | `(24 000,) int8` | CNOT index 0..23 |
| `split.npz` | `train_idx, val_idx, test_idx` | stratified |
| `cnot_keys.json` | 24 strings | label → CNOT name |
| `config.json` | dict | generation parameters + atom source |
| `sanity.json` | dict | per-class vacuum frac, mean bits, totals |

### 13.3 Sanity (recorded in `sanity.json`)

- Class balance: 1 000 ± 0 shots / class.
- Vacuum fraction (all-zero syndrome): **0.32** vs analytical lower bound `(1 − p_high)^d · (1 − p_bg)^(32 d) = 0.28`. The 0.04 gap is XOR-cancellation among multi-fault events.
- Per-class vacuum frac across CNOTs: 0.295 – 0.348 (≈ 3σ MC range at N = 1 000).
- Mean event bits per shot: 2.1 – 2.7.

### 13.4 Theoretical N guidance for a single bag

From §12.4, the hardest pair at this regime has `TV ≈ 0.104`. Approximate `N* ≈ 8 ln(20) / TV² ≈ 720` shots/bag for ~95 % accuracy on the worst pair; `N* ≈ 100` already gives a useful signal on the median pair. Sweep `N ∈ {10, 30, 100, 300}` first; regenerate with larger `N_per_class` (e.g. 5 000) before going past `N = 300` per bag.

### 13.5 Regeneration

```bash
python scripts/build_ml_dataset.py \
    --mode noreset --d 3 --p-high 0.1 --p-bg 0.01 \
    --n-per-class 1000 --seed 42
```

Reproducible — `seed` is recorded in `config.json` and feeds both the per-class sampling and the stratified split.

### 13.6 Status & next

- ✅ Per-shot pool ready for one regime.
- ⏳ Bag DataLoader (PyTorch `Dataset` with on-the-fly N-shot sampling per label).
- ⏳ Deep Sets / Set Transformer baseline, then accuracy-vs-N curve to compare against §12.4 theoretical N*.
- ⏳ Additional regimes when needed: `p_high ∈ {0.01, 0.05}`, `reset` mode counterpart, `d = 5`.
