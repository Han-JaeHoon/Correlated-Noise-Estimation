# Surface-code decoding — survey + MWPM baseline implementation

> 한국어: [`DECODING_SURVEY_kor.md`](DECODING_SURVEY_kor.md)

**Purpose.** Before generating correlated-noise data, set up the standard
surface-code decoding infrastructure: survey the common decoders, implement the
most representative one (MWPM), and wire it to the existing `RotatedSurfaceCode`.

## 1. The decoding problem

A noisy QEC cycle produces **detection events** (detectors that flipped). The
decoder maps detection events → a correction, and we score it by whether the
**logical observable** ends up flipped. Stim packages the noise as a **detector
error model (DEM)**: an error hypergraph whose hyperedges connect the detectors
each elementary fault triggers. Every decoder below consumes this DEM (or its
graph form).

## 2. Common decoders (what's used in practice)

| Decoder | Idea | Accuracy | Speed | When |
|---|---|---|---|---|
| **MWPM** (matching) | Errors that flip ≤2 detectors → graph edges; find min-weight matching of the flipped detectors | **High — reference** | Moderate | **Standard baseline** for surface code threshold studies |
| **Union-Find** | Grow/merge clusters until each spans a correction | Near-MWPM | **Fast** (≈ near-linear) | large `d`, real-time decoding |
| **BP + OSD** | Belief propagation + ordered-statistics post-processing | High, handles non-graphlike | Slow | general / **correlated** (qLDPC) codes where MWPM's graph assumption breaks |
| **Neural** (e.g. AlphaQubit) | Learned from data | Highest reported | Expensive | research frontier; needs training data |

**Why MWPM is the representative choice.** For the surface code, errors
decompose into mechanisms that flip at most two detectors, so the DEM is
**graphlike** and matching is exact-ish and well-understood. MWPM is the
accuracy-oriented reference every other decoder is benchmarked against
(Dennis–Kitaev–Landahl–Preskill 2002; Fowler et al. 2012), and **PyMatching 2**
(Higgott & Gidney 2023) is its de-facto fast implementation, built to consume
Stim DEMs directly. So MWPM is both the standard and the lowest-friction fit for
our Stim-based code.

## 3. Implementation (`src/backend_stim/decoding.py`)

`MatchingDecoder` wraps PyMatching, driven entirely by the DEM that
`RotatedSurfaceCode.build_circuit` already emits — no extra modeling:

```
circuit ──detector_error_model(decompose_errors=True)──▶ pymatching.Matching
        ──decode detection events──▶ predicted logical flips
```

```python
from src.backend_stim import RotatedSurfaceCode, MatchingDecoder, logical_error_rate

code = RotatedSurfaceCode(5)

# build a decoder under a simple circuit-level noise model (gate + measurement)
dec = MatchingDecoder.from_code(code, after_cnot_depolarize=1e-3, measure_flip=1e-3)
ler = dec.logical_error_rate(shots=100_000)          # Monte-Carlo logical error rate

# or the one-liner
ler = logical_error_rate(code, p=1e-3, shots=100_000)

# decode your own detection events (shots × num_detectors) -> (shots × num_observables)
pred = dec.decode_batch(detection_events)
```

Key property: the decoder is **noise-model agnostic** — it decodes whatever DEM
the circuit compiles to. The noise knobs live on the circuit
(`after_cnot_depolarize`, `measure_flip`, later a correlated channel); the same
`MatchingDecoder` handles all of them as long as the DEM stays graphlike.

## 4. Validation (`scripts/validate_decoder.py`)

| Check | Result |
|---|---|
| Noiseless → zero logical errors (d=3,5,7) | ✅ LER = 0 |
| Below threshold (p=1e-3): LER decreases with d | ✅ 3.5e-4 → 7.0e-5 → 1.0e-5 (d=3→5→7) |
| (p×d) sweep brackets a threshold | ✅ curves cross near p* ≈ 0.7–1% |

`(p × d)` logical error rate (depolarizing CNOT + measurement flip at rate p):

| p | d=3 | d=5 | d=7 |
|---|---|---|---|
| 0.003 | 0.003 | 0.001 | 0.000 |
| 0.006 | 0.011 | 0.010 | 0.006 |
| 0.010 | 0.028 | 0.036 | 0.037 |
| 0.015 | 0.058 | 0.098 | 0.132 |
| 0.020 | 0.096 | 0.177 | 0.260 |

Below ~0.7% larger `d` helps (error suppression); above it larger `d` hurts —
the textbook threshold crossing, which simultaneously confirms the decoder is
correctly bound to our distance-`d` code.

## 5. Connection to the correlated-noise work

- The DEM/`pij` picture the correlated-noise survey
  ([`CORRELATED_NOISE_SURVEY.md`](CORRELATED_NOISE_SURVEY.md)) describes is the
  **same object** this decoder consumes: correlated mechanisms are extra DEM
  hyperedges. So once a correlated channel is added to `build_circuit`, this
  decoder runs on it unchanged (PyMatching decomposes hyperedges to its graph).
- For genuinely correlation-aware decoding, **BP+OSD** is the natural upgrade
  (it does not assume a graphlike DEM). MWPM stays the baseline; BP+OSD is a
  documented next option, not implemented here.
- The decoder is infrastructure: the correlated-pair *classification* task
  operates on syndromes directly, but having a standard decoder gives a logical
  error-rate benchmark and the DEM tooling the classifier analysis will reuse.

## Sources

- [Review on the decoding algorithms for surface codes (arXiv 2307.14989)](https://arxiv.org/html/2307.14989v4) — MWPM vs Union-Find vs BP+OSD vs neural.
- [PyMatching 2 (Higgott & Gidney)](https://pymatching.readthedocs.io/en/stable/) — `Matching.from_detector_error_model`, `decode_batch`.
- Dennis, Kitaev, Landahl, Preskill (2002); Fowler, Mariantoni, Martinis, Cleland, PRA 86, 032324 (2012) — surface code + MWPM.
