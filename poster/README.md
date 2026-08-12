# Poster — Surface-Code Syndrome Utilization for Fault Identification

A1 **portrait** academic poster for the Niels Bohr Quantum Summer School 2026
student poster session, built around a single research line: identifying the
dominant faulty CNOT location in a `d=3` rotated surface code from repeated
syndrome sequences alone.

## Narrative / sections

1. **Motivation** — recycle QEC syndrome data into device knowledge (closed loop)
2. **Case Study & Problem** — `d=3` rotated code layout + problem statement
3. **Single-Shot Ambiguity** — one shot is not enough (216→123 signatures,
   72 cross-CNOT collision pairs; single-shot identification impossible)
4. **Temporal Information Breaks the Ambiguity** — temporal `(T×8)` vs spatial
   bag-of-`N` representations; accuracy climbs from random to `≈0.9` as data grows
5. **Learned Decoders** — matched model per regime (temporal GRU, spatial LogReg)
   + a prominent best-accuracy summary
6. **Status & Outlook** (full-width footer)

## Files

| File | Role |
|---|---|
| `poster.tex` | Self-contained `tikzposter` source (A1 portrait) |
| `figures/fig_collisions.png` | Cross-Pauli cross-CNOT collision map (72 collisions) — §3 |
| `figures/fig_acc_vs_N.png` | Bag-of-shots accuracy vs bag size `N` — §4 |

All figures are **real analysis outputs** from the project (no fabricated data):

- `fig_collisions` — from `data/analysis/5_cross_pauli_pair_collisions/`
  (this branch / `main`).
- `fig_acc_vs_N` — from `0_naiveSurfaceCode/spatial`
  (`data/analysis/spatial_mixture/plots/full_accuracy_vs_N.png`).

The two TikZ schematics (data-shape in §4, model architecture in §5) are drawn
from the documented model descriptions; they are structural diagrams, not data.
Result numbers (216→123 signatures, 72 collision pairs, GRU ≈95%, LogReg ≈94% at
N=300) are the values documented in the repository READMEs and source figures.
The `≈95%` / `≈94%` accuracies are labelled **preliminary, d=3**. Ongoing /
probabilistic / general-`d` work is phrased as **in progress**, not confirmed.

## Compiling

- **Overleaf**: upload `poster.tex` + the `figures/` folder, set compiler to
  **pdfLaTeX**, compile. No shell-escape or external assets required.
- **Local**: `pdflatex poster.tex` (needs `tikzposter`, `lmodern`, `booktabs`).

## Logos

The header shows a logo panel on each side. Drop the real logos in and they
appear automatically (no code change needed):

- `figures/logo_nbi.png` — top-left (Niels Bohr Quantum Summer School)
- `figures/logo_kaist.png` — top-right (KAIST)

Until a file is present, a labelled placeholder box is shown. Logos are placed
**directly on the dark navy header** (no white panel): the NBI logo is
self-contained, and the KAIST wordmark is **white**, so it reads on the dark
background — a white panel would hide it. Use transparent-background PNGs.
Current sizes: `logo_nbi.png` at height 3 cm, `logo_kaist.png` at height 1.7 cm
(tweak the `height=` values in `\leftlogo` / `\rightlogo` if needed).
