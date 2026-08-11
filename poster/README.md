# Poster — Surface-Code Syndrome Utilization for Fault Identification

A1 **portrait** academic poster for the Niels Bohr Quantum Summer School 2026
student poster session, built around a single research line: identifying the
dominant faulty CNOT location in a `d=3` rotated surface code from repeated
syndrome sequences alone.

## Narrative / sections

1. **Motivation** — recycle QEC syndrome data into device knowledge (closed loop)
2. **Case Study & Problem** — `d=3` rotated code layout + problem statement
3. **Single-Shot Ambiguity** — one shot is not enough (72 cross-CNOT collisions)
4. **More Data Breaks the Ambiguity** — temporal `(T×8)` vs spatial bag-of-`N`
   representations; accuracy climbs from random to `≈0.9` as data grows
5. **Learned Decoders: Models & Performance** — matched model per regime
   (GRU / Deep Sets) + accuracy-vs-`T` performance
6. **Status & Outlook** (full-width footer)

## Files

| File | Role |
|---|---|
| `poster.tex` | Self-contained `tikzposter` source (A1 portrait) |
| `figures/fig_collisions.png` | Cross-Pauli cross-CNOT collision map (72 collisions) — §3 |
| `figures/fig_acc_vs_N.png` | Bag-of-shots accuracy vs bag size `N` — §4 |
| `figures/fig_acc_vs_T.png` | Sequence-classifier accuracy vs `T` (R1/R2/R3B) — §5 |

All figures are **real analysis outputs** from the project (no fabricated data):

- `fig_collisions` — from `data/analysis/5_cross_pauli_pair_collisions/`
  (this branch / `main`).
- `fig_acc_vs_N` — from `0_naiveSurfaceCode/spatial`
  (`data/analysis/spatial_mixture/plots/full_accuracy_vs_N.png`).
- `fig_acc_vs_T` — from `0_naiveSurfaceCode/sequential`
  (`data/analysis/9_classifier/accuracy_vs_T.png`).

The two TikZ schematics (data-shape in §4, model architecture in §5) are drawn
from the documented model descriptions; they are structural diagrams, not data.
Result numbers (216→72 collisions, 8/24 single-round, GRU ≈95%, LogReg ≈94% at
N=300) are the values documented in the repository READMEs and the source
figures. Ongoing / probabilistic / general-`d` work is phrased as **in
progress**, not confirmed.

## Compiling

- **Overleaf**: upload `poster.tex` + the `figures/` folder, set compiler to
  **pdfLaTeX**, compile. No shell-escape or external assets required.
- **Local**: `pdflatex poster.tex` (needs `tikzposter`, `lmodern`, `booktabs`).

## Placeholders to fill in

Header carries placeholders — replace before printing:

- `[co-authors / advisor placeholder]`
- `[Institution / Group placeholder]`
