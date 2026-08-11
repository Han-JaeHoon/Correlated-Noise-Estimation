# Poster — Surface-Code Syndrome Utilization for Fault Identification

A1 **portrait** academic poster for the Niels Bohr Quantum Summer School 2026
student poster session, built around a single research line: identifying the
dominant faulty CNOT location in a `d=3` rotated surface code from repeated
syndrome sequences alone.

## Files

| File | Role |
|---|---|
| `poster.tex` | Self-contained `tikzposter` source (A1 portrait) |
| `figures/fig_round1_degeneracy.png` | Single-round XZ degeneracy (8 / 24 unique) |
| `figures/fig_round2_degeneracy.png` | Two-round XZ degeneracy (24 / 24 unique) |
| `figures/fig_collisions.png` | Cross-Pauli cross-CNOT collision map (72 collisions) |

All three figures are **real analysis outputs** copied from `data/analysis/`
(no fabricated data). The result numbers quoted in the poster (8/24, 24/24, 216
cases → 72 collisions, and the preliminary learned-decoder figures) are the
values documented in the repository `README.md`.

## Compiling

- **Overleaf**: upload `poster.tex` and the `figures/` folder, set the compiler
  to **pdfLaTeX**, and compile. No shell-escape or external assets required.
- **Local**: `pdflatex poster.tex` (needs `tikzposter`, `lmodern`, `booktabs`).

## Layout

Two columns + a full-width footer:

- **Left** — 1. Motivation (pipeline), 2. Case Study & Problem, 3. Beyond Single-Shot
- **Right** — 4. Temporal Information Threshold, 5. Cross-Pauli Ambiguity Wall
- **Footer** — 6. Status & Outlook + contact

## Placeholders to fill in

The header carries placeholders — replace before printing:

- `[co-authors / advisor placeholder]`
- `[Institution / Group placeholder]`

## Notes

- Ongoing / probabilistic / general-`d` work is phrased as **in progress**, not
  as confirmed results, per the current state of the project.
- Poster body text is English; this README documents the build.
