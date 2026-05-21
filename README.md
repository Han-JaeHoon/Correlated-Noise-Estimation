# Correlated-Noise-Estimation

A research codebase for analyzing whether the **location of a faulty CNOT gate** acting as a dominant noise source in a `d=3` rotated surface code can be **identified from syndrome measurement sequences alone**, and for designing a learning model on top of those findings.

(See [`README_kor.md`](README_kor.md) for the Korean version.)

> **Picking this up from another machine / Claude session?** Start with [`HANDOFF.md`](HANDOFF.md) — a 30-second context loader pointing to the current branch, latest result, and recommended next step.

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
    └── analysis/                       # analysis results (output of this study)
        ├── 1_per_pauli_degeneracy/
        ├── 2_pauli_sweep_summary/
        ├── 3_cross_pauli_conflict/
        ├── 4_cross_pauli_per_pauli_view/
        └── 5_cross_pauli_pair_collisions/
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

## 12. Long-sequence frame (branch `long-sequence-analysis`)

Sections 1–11 analyze a **single-shot** frame: one fault event, one syndrome string, identify the (CNOT, Pauli) pair. That frame hit a hard wall — 72 (CNOT, Pauli) collisions that are invariant under round count and measurement mode. The branch `long-sequence-analysis` reframes the problem to bypass that wall using **sequence-level statistics**.

### 12.1 Problem reframing (R1 scenario)

| Aspect | Single-shot (main) | R1 (this branch) |
|---|---|---|
| Fault occurrence | Single deterministic event at chosen rounds | **Every round, every CNOT** rolls Bernoulli |
| Per-CNOT rate | n/a | Background `p_bg` on all 24 CNOTs, **elevated `p_high`** on one designated CNOT |
| Pauli per event | Fixed pair | **Uniform draw from 15 non-identity 2-qubit Paulis** (II excluded) |
| Observable | One syndrome string | `(T, 8)` syndrome stream |
| Target | (CNOT, Pauli) joint | **Dominant CNOT only** (Pauli marginalized) |
| Decoder | n/a | Pluggable — R1 uses `IdentityDecoder` (no-op) |

The hope: even when single-round syndromes collide, the **distributions over T-round sequences** induced by different dominant CNOTs may be distinguishable, breaking the 72-pair ceiling.

### 12.2 New modules

| File | Role |
|---|---|
| `src/decoder.py` | `Decoder` ABC + `Correction` dataclass + `IdentityDecoder`. Extension point for window decoders (R3b lookup decoder etc.) |
| `src/stochastic_faults.py` | `BackgroundElevatedSampler(p_bg, p_high, faulty_cnot_id, rng)` — per-round per-CNOT Bernoulli + uniform 15-Pauli sampling |
| `src/sequence_runner.py` | `run_sequence(T, sampler, decoder, ...)` — single-QNode fast path for `IdentityDecoder`; non-Identity decoder path raises `NotImplementedError` until window state hand-off is added |
| `src/sequence_dataset.py` | `generate_r1_dataset(...)` + `save_r1_dataset(...)` — N samples of `(T, 8)` syndromes with seeds and oracle fault logs |

### 12.3 New scripts

| Script | Role |
|---|---|
| `scripts/generate_r1_sequences.py` | CLI for R1 dataset generation. Output: `data/r1_sequences/{baseline\|cnotXX}/T{T}_pbg{p}_phigh{p}_seed{s}/{reset\|no_reset}/` |
| `scripts/sanity_check_r1_infra.py` | Regression test — see §12.4 |

### 12.4 Sanity check (passing)

The new infrastructure reproduces main-branch single-shot results bit-for-bit on a controllable case.

- **Test B** — sampler dict structure: For all 216 = 24 CNOT × 9 Pauli pair cases, `BackgroundElevatedSampler(p_bg=0, p_high=1, faulty_cnot_id=k)` emits fault dicts whose `(round, control, target, error_wires, error_types)` match `make_fixed_cnot_fault_schedule`'s output (extra fields `cnot_id`, `pauli_pair` allowed).
- **Test C** — end-to-end syndrome equality: For all 24 CNOTs with Pauli=XZ at every round of a T=2 sequence, a forced-deterministic sampler driven through `run_sequence` produces `(T, 8)` syndromes bit-identical to the main-branch path (`make_fixed_cnot_fault_schedule` + `make_repeated_stabilizer_qnode`). Verified under both `reset` and `no_reset` modes. **24/24 PASS** for each mode.

This validates the wiring `sampler → fault_schedule format → QNode → reshape`. The stochastic-sampling itself is delegated to numpy and not unit-tested separately.

### 12.5 Capability summary

What the R1 infrastructure can produce today, with no further coding:

- Datasets of shape `(N, T, 8) uint8` with deterministic seeds + per-sample oracle fault logs
- Knobs: `faulty_cnot_id ∈ {None, 0..23}`, `p_bg, p_high ∈ [0,1]`, `T ∈ ℤ⁺`, `reset` / `no_reset`
- Scenarios: R1 main experiment, baseline (no elevated CNOT), reproduction of single-shot results via forced sampler

Open extension points: non-Identity decoder path (window state hand-off in `sequence_runner.py:72`), final data-qubit measurement, alternative Pauli pools, Stim backend, code distance > 3.

### 12.6 Next step — device viability map

`p_bg` and `p_high` are properties of the physical device (current superconducting two-qubit gate errors ≈ 10⁻³ – 10⁻²; degraded CNOTs reach 10⁻² – 10⁻¹), not parameters we tune. `T` is an experimental design choice bounded by coherence time. The next step is therefore not "pick parameters" but **characterize the regime of viability**:

> On a grid over `(p_bg, p_high, T)`, compute the theoretical maximum dominant-CNOT identification accuracy (Bayes-optimal classifier on the per-round syndrome distribution, with Pauli marginalized). The output is a *viability map* answering:
> 1. For which device parameter regimes is the identification task feasible?
> 2. How long must one observe (T) to achieve a target accuracy?
> 3. Which of the 72 original collision pairs are broken at the sequence level, and which survive as fundamental limits?

This ceiling is also the ceiling for any trained classifier, so it doubles as an evaluation benchmark for the model phase that follows.

---

## 13. R1 ceiling result (Phase 2 — Task #3)

### 13.1 What we computed

For each candidate dominant CNOT k, the per-round detection event distribution

$$P_k(s) = \left(\bigast_{i \neq k} D_i^{(p_{bg})}\right) \ast D_k^{(p_{high})}, \quad s \in \mathbb{F}_2^8$$

was computed as an XOR-convolution of 24 independent per-CNOT contribution distributions, where each $D_i^{(p)}$ assigns mass $1-p$ to 0 and mass $p/|\text{Pauli pool}|$ to each lookup syndrome. Under reset mode, detection events are i.i.d. across rounds, so the T-round Bayes-optimal classifier is $\hat{k} = \arg\max_k \sum_t \log P_k(d_t)$. Accuracy was estimated by Monte Carlo on a (p_bg, p_high, T) grid.

Sub-task scripts:
- `src/ceiling.py` — lookup builder, XOR-convolution, Bayes classifier
- `scripts/compute_r1_ceiling.py` — (p_bg, p_high, T) sweep + visualization
- `scripts/analyze_ceiling_groups.py` — structural ambiguity group analysis + 15-Pauli vs 9-Pauli comparison

### 13.2 Headline finding — 4 structural ambiguity groups

Two dominant hypotheses $k, k'$ produce identical per-round distributions iff their **single-fault syndrome multisets** $\{\sigma_{k,\alpha}\}_\alpha$ and $\{\sigma_{k',\alpha}\}_\alpha$ are equal as multisets on $\mathbb{F}_2^8$. Direct check on the lookup yields:

| Group | Members | Stabilizer | Multiset (15-Pauli) |
|---|---|---|---|
| $G_1$ | CNOT 6, 7 | Z2 stab (data → ancilla) | $\{0^3, 4^4, 64^4, 68^4\}$ |
| $G_2$ | CNOT 14, 15, 17 | X1 stab interior (ancilla → data) | $\{0^7, 32^8\}$ |
| $G_3$ | CNOT 18, 19, 20 | X2 stab interior | $\{0^7, 64^8\}$ |
| $G_4$ | CNOT 22, 23 | X3 stab | $\{0^7, 128^8\}$ |

Total identifiable groups = 24 − (2+3+3+2) + 4 = **18**. Asymptotic per-class accuracy ceiling = $18/24 = 0.750$. Group-level identification reaches 1.0 in the limit.

### 13.3 Verification — Monte Carlo at large T

At $(p_{bg}, p_{high}) = (0.01, 0.3)$:

| T | per-class acc | group acc |
|---|---|---|
| 10 | 0.42 | 0.56 |
| 100 | 0.747 | 0.996 |
| 300 | 0.749 | 1.000 |
| 1000 | 0.751 | 1.000 |

→ per-class saturates exactly at 18/24, group accuracy reaches 1.0. Within an ambiguity group the Bayes-optimal classifier predicts one fixed representative; the other members receive 0% accuracy.

### 13.4 Per-Pauli vs multiset — what is and isn't claimed

Two CNOTs in the same group **do not produce identical syndromes for every Pauli**. For CNOT 6 vs 7, only 7 out of 15 Paulis give the same syndrome; the other 8 differ. The two are indistinguishable only because there exists a permutation $\pi$ of the 15-Pauli pool with $\sigma_{6,\alpha} = \sigma_{7,\pi(\alpha)}$. Under the uniform random Pauli assumption, marginalizing yields identical distributions, but if the Pauli were known per fault (as in the main-branch single-shot frame), the two would be distinguishable.

### 13.5 Structural reason

A fault injected after CNOT in stabilizer $S$ during round $t$ can only affect ancilla measurements of stabilizers that **(a) run later than $S$ in the round order $[Z_0, Z_1, Z_2, Z_3, X_0, X_1, X_2, X_3]$, and (b) share a data qubit with the fault location**.

The four ambiguity groups all consist of CNOTs whose data qubits have **identical "downstream stabilizer membership"**. For example, within $X_2$ stab (qubits {3,4,6,7}), only qubit 7 is also in the downstream $X_3$ stab. So CNOTs 18 (target=3), 19 (target=4), 20 (target=6) share the same downstream footprint and collapse, while CNOT 21 (target=7) is set apart. Identical reasoning applies to $G_1, G_2, G_4$.

→ The ceiling reflects **stabilizer measurement schedule × code topology**, not Pauli randomness. Changing the Pauli pool can partially shrink the ambiguity (see next), but cannot eliminate the X-stab interior groups.

### 13.6 15-Pauli vs 9-Pauli comparison

If we restrict to the 9 fully-two-qubit Paulis (X/Y/Z × X/Y/Z, excluding single-leg IX, XI, etc.):

| Pauli pool | Ambiguity groups | per-class ceiling |
|---|---|---|
| 15-Pauli (incl. single-leg) | $\{G_1, G_2, G_3, G_4\}$ | $18/24 = 0.750$ |
| 9-Pauli (no single-leg) | $\{G_2, G_3, G_4\}$ | $19/24 = 0.792$ |

$G_1$ (CNOT 6, 7) is **pool-dependent**: the 6 single-leg Paulis play a "noise-balancing" role that equalizes their multisets in the 15-Pauli model. Remove them, and the multisets become $\{0^2, 4^3, 64^1, 68^3\}$ vs $\{0^1, 4^2, 64^2, 68^4\}$ — distinct. The X-stab interior groups $G_2, G_3, G_4$ survive both pools (structural).

See [data/analysis/6_sequence_ceiling/ceiling_compare_curves.png](data/analysis/6_sequence_ceiling/ceiling_compare_curves.png) for the per-class accuracy curves under both models.

### 13.7 Could decoding break the structural groups?

We conjecture **yes** — adding a window decoder (the R3b scenario) likely breaks the X-stab interior groups too. The intuition:

- Without a decoder (R1), the dominant CNOT only affects the *per-round detection event distribution*. Two CNOTs producing identical distributions are forever indistinguishable.
- With a decoder, the decoder applies a correction $C(s)$ based on the observed syndrome $s$. When syndromes are ambiguous between $k$ and $k'$, the decoder applies the **same** correction to both — but the **actual residual data-qubit error differs** (because the true fault was at a different physical qubit). This residual cascades into **subsequent** rounds' syndromes in distinguishable ways.

Concretely for $G_3 = \{18, 19, 20\}$: a fault at CNOT 18 leaves residual on data qubit 3; at CNOT 19, on qubit 4; at CNOT 20, on qubit 6. Single-round syndromes coincide, but subsequent rounds' Z-stab and X-stab measurements involving those different qubits would produce statistically distinct streams once the post-correction residual interacts with new faults.

→ This makes the R3b extension scientifically valuable, not just an "alternative scenario". Promoting Task #7 from "future work" to "near-term experimental priority" is warranted.

### 13.8 Practical implications

1. **Redefine the target as 18-group (or 19-group with 9-Pauli) identification.** That's the information-theoretic resolution of the R1 problem.
2. **Group accuracy is the operationally meaningful metric.** Per-class accuracy is capped at 0.75 by construction and conflates real classifier quality with structural ambiguity.
3. **Viability region is still meaningful.** Within (p_bg, p_high, T), group accuracy ranges from 1/18 (random) to 1.0. The transition region around `(p_bg=1e-2, p_high=1e-1, T≈100)` is the sweet spot for evaluating learned classifiers (Task #6).
4. **Defer detailed parameter selection (Task #2)** until R3b ceiling is known: if R3b breaks the X-stab groups, the (p_bg, p_high, T) region of interest shifts.

---

## 14. R3b ceiling result (branch `decoder-add-analysis`)

Section 13 left the §13.7 conjecture open: would adding a single-round window decoder break the X-stab interior ambiguity groups? Branch `decoder-add-analysis` builds the R3b infrastructure end-to-end and answers it. The summary up front: **the qualitative half of the conjecture is confirmed (per-round marginals diverge between group members under R3b), but the quantitative claim is falsified (the resulting marginal-Bayes accuracy is markedly worse than R1's, not better).**

### 14.1 Infrastructure added

| File | Role |
|---|---|
| `src/pauli_frame.py` | 17-qubit symplectic Pauli frame with in-place CNOT/H/reset updates |
| `src/round_propagation.py` | Symbolic round propagator that mirrors `stabilizer_round` gate-for-gate; produces both the post-round frame and the predicted ancilla outcomes |
| `src/decoder.py` `LookupDecoder` | window-size-1 decoder built from the 24×15 single-fault syndrome and residual lookups; lex-min `(k, α)` tie-break; multi-fault rounds → identity |
| `src/sequence_runner.py` window path | per-round 1-round QNode call with `initial_error_list` seeded from the carried Pauli frame; reproduces fast-path syndromes when run with IdentityDecoder (regression-tested) |
| `src/fast_simulator.py` `run_sequence_symbolic` | drop-in for `run_sequence` that uses the propagator only — ~100× faster, bit-identical (verified 18/18 cases) |
| `src/ceiling_r3b.py` | MC marginal-Bayes ceiling estimator |
| `scripts/sanity_check_round_propagation.py` | PL vs symbolic single-round lookup (360/360), data-Pauli preservation (27/27), 2-round end-to-end (360/360) |
| `scripts/sanity_check_fast_simulator.py` | PL vs symbolic full-sequence (9 + 9 cases) + benchmark |
| `scripts/sanity_check_r3b_runner.py` | fast vs window IdentityDecoder equality; LookupDecoder smoke; LookupDecoder ≠ IdentityDecoder |
| `scripts/compute_r3b_ceiling.py` | `(p_bg, p_high, T)` grid sweep with both R1 (analytic) and R3b (MC) classifiers |
| `scripts/analyze_r3b_ceiling.py` | per-class accuracy by group, within-group spread, pairwise JS divergence between R3b marginals |

The R3b decoder's spec is pinned in [docs/r3b_design.md](docs/r3b_design.md).

### 14.2 What we computed

For each candidate dominant CNOT k ∈ {0..23}, we ran `n_train` symbolic sequences of length T with `LookupDecoder` engaged, dropped round 0 (decoder-agnostic since the frame is always 0 there), and estimated the per-round marginal P(d | k) by histogram (Laplace-smoothed). The marginal Bayes classifier

\[\hat{k} = \arg\max_k \sum_{t \ge 1} \log P(d_t | k)\]

was then applied to `n_test` independently sampled test sequences for each true k*. The resulting confusion matrix gives per-class accuracy, group accuracy (using the R1 ambiguity groups), and per-pair JS divergence between marginals.

### 14.3 Headline numbers

At `(p_bg, p_high) = (0.01, 0.1)`, `n_train = n_test = 50`–`200`, seed 0 (data: `data/analysis/7_r3b_ceiling/`):

8 cells, n_train=n_test=50–200, seeds 0–1 (data: `data/analysis/7_r3b_ceiling/`):

| (p_bg, p_high) | T | R1 per-class | R3b per-class | R1 group | R3b group |
|---|---|---|---|---|---|
| (0.001, 0.01) | 30  | 0.102 | 0.087 | 0.178 | 0.118 |
| (0.001, 0.01) | 100 | 0.190 | 0.087 | 0.255 | 0.133 |
| (0.001, 0.01) | 300 | 0.327 | 0.095 | 0.418 | 0.125 |
| (0.005, 0.05) | 30  | 0.225 | 0.113 | 0.290 | 0.162 |
| (0.005, 0.05) | 100 | 0.421 | 0.104 | 0.544 | 0.140 |
| (0.01,  0.1)  | 10  | 0.182 | 0.112 | 0.243 | 0.149 |
| (0.01,  0.1)  | 30  | 0.309 | 0.093 | 0.395 | 0.120 |
| (0.01,  0.1)  | 50  | 0.406 | 0.101 | 0.527 | 0.130 |

**R1 grows smoothly with T and with the (p_bg, p_high) signal-to-background ratio**; **R3b plateaus around 0.08–0.16 regardless of T or p**. At the longest T tested per cell, R1 already exceeds R3b by 3–4×. The 3-cell column at fixed (p_bg=0.001, p_high=0.01) makes the plateau especially clear — R1 climbs 0.10 → 0.19 → 0.33 as T goes 30 → 100 → 300, while R3b stays at 0.087 → 0.087 → 0.095. Group accuracy tells the same story.

### 14.4 Qualitative confirmation: marginals do diverge

For the R1 ambiguity groups under R3b at T=50, the pairwise Jensen-Shannon divergence of per-round marginals between group members is non-zero:

| Group | mean pairwise JS | max pairwise JS |
|---|---|---|
| {6, 7} | 0.065 | 0.065 |
| {14, 15, 17} | 0.062 | 0.068 |
| {18, 19, 20} | 0.067 | 0.075 |
| {22, 23} | 0.066 | 0.066 |

Under R1 these divergences are exactly 0 by construction (the multisets coincide). R3b does break the per-round-marginal equivalence, which is the §13.7 mechanism. So the *directional* claim is correct: the decoder leaks group-distinguishing information into subsequent rounds' marginals.

### 14.5 Why R3b loses overall: wrong-correction noise injection

Despite the marginals diverging, the R3b classifier underperforms R1 by a wide margin. The cause is the lex-min single-fault decoder itself:

- Every non-zero single-fault syndrome admits **~9 distinct (k, α) preimages** in the 24×15 lookup (cross-CNOT syndrome collisions are pervasive).
- The decoder commits to the lex-min preimage every time it sees that syndrome, regardless of the true (k, α). On nearly every round, the picked residual differs from the true residual.
- Applying that wrong correction injects a near-random Pauli residual into the data-qubit frame entering the next round.
- The dominant-CNOT signal — which R1 was patiently accumulating as a slow systematic bias on top of background noise — is effectively scrambled by these random corrections.

The within-group standard deviation of R3b's per-class accuracy is uniformly low (≤ 0.05 at T=50 across all four groups), in contrast to R1's high spread (driven by the lex-min favoring one specific member of each ambiguous syndrome). R3b makes all group members **equally bad** rather than rescuing the disfavored ones.

### 14.6 Verdict on the §13.7 hypothesis

| Claim | Status |
|---|---|
| "The decoder, by applying the same correction to two ambiguous histories, lets the *true* residuals propagate differently into subsequent rounds." | ✅ Confirmed (JS > 0). |
| "Therefore the R3b classifier can distinguish ambiguity-group members and exceed the 0.75 R1 ceiling." | ❌ Falsified for this particular decoder. R3b sits at ≈ 0.10. |

The gap is not a small constant — it's an order of magnitude. We attribute the gap to the decoder's lex-min commitment policy, not to the window-size choice or the propagator's correctness (both validated independently).

### 14.7 What would actually break the X-stab interior groups

Things to try, in roughly decreasing order of expected payoff:

1. **Posterior-aware decoder.** Instead of lex-min, use the `(p_bg, p_high)` prior to compute a soft posterior over `(k, α)` and either Bayes-average the correction or sample from the posterior. The mean residual becomes a less aggressive perturbation of the frame.
2. **Multi-round window.** With window > 1, the decoder sees several syndromes before committing, which can disambiguate single-fault hypotheses without an arbitrary tie-break.
3. **Full-sequence ML decoder.** Treat the syndrome stream itself as the observation and train the dominant-CNOT classifier directly, with the decoder marginalized over its choices. This is the R3 scenario in its strongest form and is the natural next experimental target.
4. **Restrict the Pauli pool.** As in R1 §13.6, dropping the 6 single-leg Paulis (15-Pauli → 9-Pauli) collapses one R1 group (G₁ = {6,7}); a similar restriction may sharpen R3b's marginals for the same reason. Cheap experiment.

### 14.8 What this means for the project

- Task #7 ("R3b decoder") is **complete** as a hypothesis test for the window-1 lex-min decoder.
- The headline ceiling for sequence-level CNOT identification under physically realistic background noise remains the **R1 ceiling, 18/24 group accuracy** (or 19/24 with 9-Pauli).
- Tasks #5 (dataset) and #6 (classifier) are unblocked. The classifier evaluation benchmark is still R1's per-round marginal Bayes accuracy; R3b is a cautionary example rather than a target.
- The path to actually breaking the structural groups is some combination of a smarter decoder (item 14.7-1 or -2) or a full-sequence classifier (-3). All three are tractable in the same infrastructure built here.


### 14.9 Variant: HammingNearestDecoder (multi-fault → nearest-lookup)

The bare `LookupDecoder` defaults to identity correction on multi-fault rounds (detection events not in the 24×15 single-fault lookup). At p_bg = 0.01 the multi-fault rate per round is ≈ 0.24, so this branch is hit often. To isolate whether the multi-fault-identity policy is what drags R3b below R1, `src/decoder.py:HammingNearestDecoder` replaces it with: on a non-zero d not in the lookup, apply the correction of the lookup syndrome that is closest to d in Hamming distance (lex-min tie-break on the lookup syndrome). On d in the lookup or d == 0, behavior is identical to `LookupDecoder`.

At `(p_bg, p_high) = (0.01, 0.1)`, `n_train = n_test = 100`, `seed = 42`:

| T | R1 (analytic) | R3b LookupDecoder | R3b HammingNearestDecoder |
|---|---|---|---|
| 10 | 0.180 | 0.115 | 0.122 |
| 30 | 0.312 | 0.107 | 0.149 |
| 50 | 0.416 | 0.095 | 0.186 |

(per-class accuracy; group accuracy follows the same ordering with margins ≈ 1.3× higher.)

Two trends across T are striking:

1. **R1 is monotonically increasing** with T (expected — more samples to estimate the marginal log-likelihood).
2. **LookupDecoder is monotonically *decreasing*** with T (0.115 → 0.107 → 0.095). Longer sequences accumulate more wrong corrections, and the *plateau* of §14.3 turns out to be the average of a slow decline. The identity-fallback on multi-fault rounds means most rounds either over-correct (single-fault hypothesis) or do nothing — no policy actually carries useful signal forward.
3. **HammingNearestDecoder is monotonically *increasing*** with T (0.122 → 0.149 → 0.186), at roughly half R1's slope. Replacing the identity fallback with a nearest-Hamming guess turns previously-wasted rounds into partial signal carriers.

Both R3b variants remain well below R1 across all T. The conclusion of §14.6 stands: any single-fault-hypothesis decoder we tried is dominated by R1's marginal-Bayes lower bound. But the multi-fault-fallback policy is *more impactful than the lex-min tie-break*: switching from identity to nearest-Hamming nearly doubles R3b's slope.

`scripts/compare_decoders.py` produces curves across T at one (p_bg, p_high) cell; outputs land in `data/analysis/7_r3b_ceiling/decoder_compare/decoder_compare_curves.png`.



---

## 15. Sequence-level separability under R3b (branch `decoder-add-analysis`)

§14 closed with a marginal-Bayes accuracy plateau (~0.10) for R3b at T → 300. That measurement uses only L=1 per-round marginals and so is a *lower bound* on what is achievable with the full sequence. §15 directly tests the open question §14 left behind:

> **Do R3b syndrome-sequence distributions become uniquely identifiable per dominant CNOT as T grows, or are some (k, k′) pairs structurally indistinguishable at the sequence level?**

The test is classifier-free. It measures the distributions themselves, not the accuracy of any particular learning algorithm.

### 15.1 Setup

- Generate `N = 2000` R3b sequences for every dominant CNOT `k ∈ {0..23}` at `T = 200` rounds, `(p_bg, p_high) = (0.01, 0.1)`, reset mode, `LookupDecoder`. Symbolic simulator (`src/fast_simulator.py`); total wall-clock ≈ 5 min.
- Drop round 0 (decoder-agnostic since the frame is empty there).
- Code: [scripts/test_seq_separability.py](scripts/test_seq_separability.py), [scripts/analyze_seq_separability_extra.py](scripts/analyze_seq_separability_extra.py), [scripts/analyze_d_scaling.py](scripts/analyze_d_scaling.py).
- Outputs: [data/analysis/8_seq_separability/](data/analysis/8_seq_separability/).

### 15.2 Measure A — pairwise L-round marginal JS divergence

For each pair (k, k′) compute the Jensen–Shannon divergence of empirical L-round marginal histograms (L ∈ {1, 2}, sliding window). Self-baseline JS estimated by splitting the N samples for each k into halves and measuring JS(half₁ ‖ half₂) gives the per-class sampling-noise floor; subtracting it gives a signal-only estimate.

| L | self-baseline (noise floor) | within-group net mean | between-group net mean | within/between ratio |
|---|---|---|---|---|
| 1 | 0.00177 | 0.00523 (range 0.00120–0.00851) | 0.01006 | ≈ 0.52 |
| 2 | 0.03003 | 0.01611 (range 0.00699–0.02201) | 0.03213 | ≈ 0.50 |

**Reading.** Within-group JS at L=1 is ~3× the sampling-noise floor — clearly above zero. Between-group JS is ~2× within-group, i.e. ambiguity-group pairs carry about half the per-round distributional separation that easy pairs do, but the separation is real and finite. At L=2 the absolute numbers grow ~3× from L=1, indicating longer windows expose more distinguishing structure; the within/between ratio stays around 0.5.

The headline plot is [data/analysis/8_seq_separability/js_curves_T200_N2000_main.png](data/analysis/8_seq_separability/js_curves_T200_N2000_main.png): every individual within-group pair sits strictly below the between-group mean at both L=1 and L=2, but every one is above the between-group min. The (6, 7) pair is the most marginal.

### 15.3 Measure B — cumulative log-likelihood ratio per pair

For each R1 ambiguity-group pair (k₁, k₂) compute, on sequences sampled from true k = k₁ and true k = k₂ separately, the cumulative L=1 log-likelihood ratio

$$\Lambda_t = \sum_{s=1}^{t} \log \hat{P}(d_s \mid k_1) - \log \hat{P}(d_s \mid k_2).$$

Under R1 this would have drift exactly 0 for ambiguity-group members and no T-scaling discrimination is possible. Under R3b, every pair shows positive drift conditional on its true k, and the two empirical distributions of Λ_t separate steadily.

At T = 200, summarising the empirical (mean ± std) of Λ_T for each pair (data: [cumulative_logLR_T200_N2000_main.csv](data/analysis/8_seq_separability/cumulative_logLR_T200_N2000_main.csv)):

| Group | Pair (k₁, k₂) | E[Λ_T \| k₁] | E[Λ_T \| k₂] | Gap | Cohen's d |
|---|---|---|---|---|---|
| G1 | (6, 7) | +1.47 ± 4.80 | −1.51 ± 5.26 | 2.98 | 0.59 |
| G2 | (14, 15) | +6.50 ± 9.61 | −7.05 ± 11.52 | 13.55 | 1.28 |
| G2 | (14, 17) | +6.22 ± 8.65 | −6.59 ± 11.33 | 12.81 | 1.28 |
| G2 | (15, 17) | +7.23 ± 10.25 | −7.25 ± 11.47 | 14.48 | 1.33 |
| G3 | (18, 19) | +4.84 ± 9.14 | −5.30 ± 8.82 | 10.14 | 1.13 |
| G3 | (18, 20) | +4.01 ± 8.11 | −4.49 ± 8.52 | 8.50 | 1.02 |
| G3 | (19, 20) | +3.82 ± 7.23 | −3.54 ± 7.06 | 7.36 | 1.03 |
| G4 | (22, 23) | +4.02 ± 7.16 | −3.75 ± 6.87 | 7.78 | 1.11 |

Cohen's *d* is the gap divided by the pooled standard deviation; *d* ≈ 1 means the two distributions overlap moderately, *d* ≥ 2 means they barely overlap. Three of the four R1 ambiguity groups (G2, G3, G4) already show *d* > 1 at T = 200 from the L=1 marginal alone; G1 (CNOT 6 vs 7) is the hardest with *d* ≈ 0.59.

The trajectory plot [cumulative_logLR_T200_N2000_main.png](data/analysis/8_seq_separability/cumulative_logLR_T200_N2000_main.png) shows the same content visually: blue band (true k₁) drifts up, red band (true k₂) drifts down, and the overlap shrinks over t.

### 15.4 Scaling — d(t) ∝ √t and projected T for clear separation

Fitting d(t) = a · √t to each pair's trajectory (t ≥ 30 to skip the burn-in; OLS, no intercept) gives the per-round-information rate. RMSE values are 0.04–0.15, well within the Gaussian-approximation noise — the √t scaling holds to within statistical error for every pair.

| Pair | d(T=200) | sqrt-fit a | T required for d = 2 (≈ 95% separation) |
|---|---|---|---|
| (6, 7) | 0.59 | 0.043 | **≈ 2,170** |
| (14, 15) | 1.28 | 0.102 | ≈ 385 |
| (14, 17) | 1.28 | 0.102 | ≈ 390 |
| (15, 17) | 1.33 | 0.100 | ≈ 400 |
| (18, 19) | 1.13 | 0.087 | ≈ 525 |
| (18, 20) | 1.02 | 0.078 | ≈ 650 |
| (19, 20) | 1.03 | 0.077 | ≈ 675 |
| (22, 23) | 1.11 | 0.082 | ≈ 590 |

Plots: [d_scaling_T200_N2000_main.png](data/analysis/8_seq_separability/d_scaling_T200_N2000_main.png) (measured d(t) + fit), [d_projection_T200_N2000_main.png](data/analysis/8_seq_separability/d_projection_T200_N2000_main.png) (√t extrapolation).

### 15.5 Verdict — direct answer

> **Yes — under R3b, syndrome sequences become uniquely identifiable per dominant CNOT as T grows, including within every R1 ambiguity group.**

Concretely:
1. Per-round L=1 distributional information rate is strictly positive for every ambiguity-group pair (within-group net JS > 0 after noise-floor subtraction).
2. The cumulative log-LR Cohen's *d* grows as √t with no observed saturation up to T = 200.
3. Projecting the √t fits, every ambiguity-group pair reaches *d* = 2 (≈ 95% separation under Gaussian approximation, i.e. ≥ 95% Bayes accuracy for a 2-class test on this pair) at finite T. The hardest pair (CNOT 6 vs 7) requires T ≈ 2,170 rounds; the rest 380–700.
4. This is achieved using L=1 marginals only. Wider-window decoders (or full-sequence likelihoods) would lower these thresholds further — the §14 plateau of the 24-class marginal-Bayes classifier (~0.10) is a property of the *classifier*, not of the underlying distinguishability.

### 15.6 Why this reframes §14

§14 reported R3b marginal-Bayes 24-class accuracy plateauing around 0.10 even at T = 300. §15 shows that plateau is *not* a structural ceiling on identification — the underlying distributions are distinguishable. The plateau is a finite-T artifact of attempting 24-way discrimination simultaneously: per-pair separation is ~0.005–0.011 nat per round at L=1, and 24-way discrimination needs log(24) ≈ 3.18 nat of pairwise information to be reliable. At T = 300 the worst pair has only 300 × 0.005 ≈ 1.5 nat — below threshold for that pair, hence the plateau. Larger T (or larger L, or a non-marginal classifier) closes this gap.

The relevant follow-up question is therefore no longer "is R3b indistinguishable?" but "at what T does a learned classifier reach the asymptote?" — a Task #6 question with a clear theoretical target now in place.
