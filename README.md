# Correlated-Noise-Estimation — `1_realisticSurfaceCode` (Phase 2, active)

A research codebase for analyzing whether the **location of a faulty CNOT gate**
acting as a dominant noise source in a rotated surface code can be **identified
from syndrome measurement sequences alone**, and for designing a learning model
on top of those findings.

(See [`README_kor.md`](README_kor.md) for the Korean version.)

> **This is the Phase 2 active branch.** It migrates the backend to **Stim**, the
> **constant-depth parallel** stabilizer schedule, and **general `d`**, then
> re-runs the Phase 0 analyses on the realistic circuit and extends the spatial
> analysis to `d·k`-round windows.
>
> - **Design + roadmap**: [`docs/PHASE2_DESIGN.md`](docs/PHASE2_DESIGN.md) (Step 0 research/grounding), [`docs/PHASE2_PLAN.md`](docs/PHASE2_PLAN.md) (roadmap 2-0…2-6).
> - **Phase 0 (frozen, validated toy results)** — the naive PennyLane `d=3` line
>   lives in the `0_naiveSurfaceCode/*` branches; the cross-phase index is on
>   `main`. This branch keeps only the realistic-circuit work, plus the one Phase 0
>   artifact reused downstream: the 387-atom table (for the 2-2 comparison).

---

## Phase 2 progress (realistic Stim circuit)

Roadmap and full detail in [`docs/PHASE2_PLAN.md`](docs/PHASE2_PLAN.md). Status at a glance:

| Milestone | Status |
|---|---|
| 2-0 Stim schedule research + design doc | ✅ done |
| 2-3 general-d surface code (constant 4-tick depth) | ✅ done, validated d=3/5/7 |
| 2-1 fault injection (deterministic) | ✅ done (stochastic p_bg/p_high pending) |
| 2-viz self-contained web visualization | ✅ done |
| **2-2 single-fault syndrome enumeration** | 🔶 **enumeration done** (below); Phase 0 structural comparison pending |
| 2-dec MWPM decoder (PyMatching) baseline | ✅ done — [`docs/DECODING_SURVEY.md`](docs/DECODING_SURVEY.md) |
| 2-4 / 2-5 / 2-6 spatial bags · d·k windows · sequential re-run | ⬜ not started |

### 2-2 · Single-fault syndrome enumeration (`scripts/build_stim_fault_enumeration.py`)

For **one stabilizer-measurement round** on the realistic circuit, every elementary
fault is enumerated and its syndrome recorded. Because a physical `|0…0⟩` reset is
**not** an X-stabilizer eigenstate, a fault-free **projection round** (round 0,
`|0…0⟩ → |0_L⟩`) precedes the single fault round (round 1); the syndrome is the
round-1 change (detection-event pattern), so it reads all-zero with no fault. This
is the same projection convention the web viz already uses (`round 0 is the
projection round`).

Fault cases:
- **data single-Pauli** — X / Y / Z on each data qubit (pre-round);
- **CNOT two-Pauli** — each of the 15 non-identity 2-qubit Paulis right after each
  directed CNOT (post-CNOT, propagates through the rest of the round = hook error).

Syndrome extraction reuses the validated differential method (sample noiseless vs
fault-injected with the same seed, XOR the raw ancilla records).

| d | ancillas | cases | unique syndromes | silent (all-zero) |
|---|---|---|---|---|
| 3 | 8 | 387 (27 data + 360 CNOT) | 36 | 89 |
| 5 | 24 | 1275 (75 data + 1200 CNOT) | 168 | 261 |

Outputs per distance in `data/analysis/stim_fault_enumeration/d{3,5}/`:
`enumeration.csv` (one row per fault case), `syndromes.npz` (syndrome matrix
`N×(d²−1)` int8 + metadata), `summary.txt` (counts + silent list). Regenerate with
`python scripts/build_stim_fault_enumeration.py` (fixed seed → reproducible).

Example Stim circuits (timeline-svg, generated for reference) live in
`viz/circuits/` — clean d=3/d=5 rounds and a D0-`Z` fault-injected d=3 circuit.

> **Remaining for 2-2:** compare this table's *structure* (collision / silent
> classes) against Phase 0's 387-atom table — the parallel schedule changes the
> fault→syndrome map, so the comparison is the actual sanity check.

---

## 1. Research goal

- **Code**: rotated surface code, general distance `d` (d=3 → 17 qubits = 9 data +
  8 ancilla; d=5 → 49; …), constant-depth parallel stabilizer schedule.
- **Assumption**: exactly one of the directed CNOTs is the dominant fault source,
  recurring at that same location (temporally correlated).
- **Target**: from the syndrome measurements alone, identify which CNOT is faulty.
- **Nuisance variables ignored**: the rounds where the fault fires, and the Pauli
  type of the fault.
- **Motivation**: at toy `d` a lookup table works, but its cost explodes with
  distance — the end goal is a single trained model queried at inference time.

---

## 2. Directory layout

```text
.
├── README.md / README_kor.md
├── requirements.txt                       # numpy, stim (+ matplotlib)
├── docs/                                   # Phase 2 design + roadmap (en/kor)
│   ├── PHASE2_DESIGN*.md
│   ├── PHASE2_PLAN*.md
│   └── SURFACE_CODE_IMPL*.md
├── src/
│   ├── backend_stim/surface_code.py        # general-d Stim surface code
│   └── bag_classifier.py                   # bag-of-shots classifiers (for 2-4 reuse)
├── scripts/
│   ├── build_stim_fault_enumeration.py     # 2-2 single-fault syndrome table
│   ├── validate_surface_code.py            # layout / schedule / distance validation
│   ├── build_viz_data.py                   # simulate → viz JSON
│   └── build_viz_html.py                   # JSON → self-contained HTML
├── viz/
│   ├── surface_code.html                   # self-contained web visualization
│   ├── surface_code.template.html
│   ├── preview_d3.png / preview_d5.png
│   └── circuits/                           # reference Stim circuit diagrams (svg)
└── data/
    ├── viz/surface_code_viz.json
    └── analysis/
        ├── stim_fault_enumeration/d{3,5}/  # 2-2 outputs (this branch)
        └── fault_enumeration/              # Phase 0 387-atom table (kept for 2-2 comparison)
```

---

## 3. `src/` modules

| File | Role |
|---|---|
| `backend_stim/surface_code.py` | First-principles general-`d` rotated surface code (`RotatedSurfaceCode(d)`): layout, constant 4-tick hook-safe schedule, `build_circuit(...)` with deterministic fault injection (pre-round data Paulis + post-CNOT 2-qubit Paulis). See [`docs/SURFACE_CODE_IMPL.md`](docs/SURFACE_CODE_IMPL.md). |
| `bag_classifier.py` | Bag-of-shots classifier models (Empirical Bayes / LogReg / MLP / Deep Sets), carried over for the planned 2-4 re-run on Stim-generated bags. |

## 4. `scripts/`

| File | Role |
|---|---|
| `build_stim_fault_enumeration.py` | 2-2 single-fault syndrome enumeration (see the Phase 2 progress section). |
| `validate_surface_code.py` | Validates layout / 4-tick schedule / distance / logical-error-rate against Stim's generator (d=3/5/7). |
| `build_viz_data.py` | Simulates faults and writes `data/viz/surface_code_viz.json`. |
| `build_viz_html.py` | Inlines the JSON into the self-contained `viz/surface_code.html`. |

---

## 5. Setup & usage

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# validate the general-d surface code (d=3/5/7)
python scripts/validate_surface_code.py

# single-fault syndrome enumeration (d=3, d=5) → data/analysis/stim_fault_enumeration/
python scripts/build_stim_fault_enumeration.py

# rebuild the web visualization, then open viz/surface_code.html
python scripts/build_viz_data.py && python scripts/build_viz_html.py
```

---

## 6. Stated assumptions

Every conclusion depends on:

1. **Single fault source** — exactly one CNOT (or one data location) faults at a time.
2. **Deterministic Pauli fault** — a fixed Pauli is applied when a fault fires (the
   stochastic background+elevated model is roadmap 2-1 / 2-4).
3. **Post-CNOT location** — the 2-qubit Pauli is injected immediately after the CNOT
   (`build_circuit(injections=…, pos="post_cx")`), so it propagates through the rest
   of the round (hook error).
4. **CNOT directionality** — `CNOT(c→t)` and `CNOT(t→c)` are distinct fault sites.
5. **No measurement / reset noise** — ancilla measurement and reset are perfect.
6. **Round 0 is a fault-free projection round** — establishes the `|0_L⟩` reference;
   faults are injected from round 1 (see 2-2 above and
   [`docs/PHASE2_DESIGN.md`](docs/PHASE2_DESIGN.md)).

---

## Phase 0 reference (other branches)

The naive PennyLane `d=3` line — sequential per-stabilizer schedule, the 216-case
single-round sweep, the **72 cross-Pauli cross-CNOT collisions**, the long-sequence
decoders (GRU 95.3 %), and the spatial bag-of-shots study (LogReg 93.9 % at N=300) —
is preserved on the `0_naiveSurfaceCode/{baseline,sequential,spatial,spatial-prebag}`
branches. The cross-phase index lives on `main`.
