#!/bin/bash
# Run analysis + git commit+push after sweep completes.
# Usage: bash scripts/post_sweep_analyze.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Running analyze_classifier_sweep.py ==="
correst_env/bin/python scripts/analyze_classifier_sweep.py \
  --scenarios r1 r2 r3b \
  --models rnn gru transformer \
  --T 128 512

echo ""
echo "=== Git status ==="
git status --short

echo ""
echo "=== Staging analysis outputs ==="
git add -f \
  data/analysis/9_classifier/training_curves_loss.png \
  data/analysis/9_classifier/training_curves_acc.png \
  data/analysis/9_classifier/accuracy_vs_T.png \
  data/analysis/9_classifier/per_class_bars.png \
  data/analysis/9_classifier/confusion_grid.png \
  data/analysis/9_classifier/sweep_summary.csv \
  data/analysis/9_classifier/r1/ \
  data/analysis/9_classifier/r2/ \
  data/analysis/9_classifier/r3b/ \
  scripts/analyze_classifier_sweep.py \
  2>/dev/null || true

echo "=== Committing ==="
git commit -m "260522 : Task #6 classifier sweep results — T∈{128,512}, 3 models × 3 scenarios

18-cell sweep (RNN/GRU/Transformer × R1/R2/R3b × T∈{128,512}).
Includes training curves (loss+acc per epoch), accuracy-vs-T summary,
per-class bar charts, and confusion matrices.

Also: bug fix (MPS label comparison), r2 scenario support, updated
analyze_classifier_sweep.py with epoch-level training curve plots.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo "=== Pushing to decoder-add-analysis ==="
git push origin decoder-add-analysis

echo ""
echo "Done. Results in data/analysis/9_classifier/"
