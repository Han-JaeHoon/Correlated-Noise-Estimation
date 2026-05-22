"""
Orchestrate the (model, scenario, T) grid for the Task #6 classifier.

Calls `scripts/train_seq_classifier.py` for each cell and aggregates the
result NIs (best val acc, test overall+group acc, per-class) into a single
sweep CSV under `data/analysis/9_classifier/sweep_summary.csv`.

Example:
    python scripts/sweep_classifiers.py \\
        --models gru transformer --scenarios r1 r3b --T 100 300 1000

Cells that already have a metrics.json under the canonical output path are
skipped unless --force is set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


def cell_dir(out_root: Path, scenario: str, model: str, T: int) -> Path:
    return out_root / scenario / f"{model}_T{T}"


def is_done(out_root: Path, scenario: str, model: str, T: int) -> bool:
    return (cell_dir(out_root, scenario, model, T) / "metrics.json").exists()


def run_cell(args, scenario: str, model: str, T: int) -> int:
    """Invoke train_seq_classifier.py as a subprocess. Return its exit code."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_seq_classifier.py"),
        "--model", model, "--scenario", scenario, "--T", str(T),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--device", args.device,
        "--seed", str(args.seed),
        "--data-dir", str(args.data_dir),
        "--out-root", str(args.out_root),
        "--input-kind", args.input_kind,
    ]
    if model == "transformer":
        cmd += ["--d-model", str(args.transformer_d_model),
                "--n-layers", str(args.transformer_n_layers),
                "--n-heads", str(args.transformer_n_heads)]
    print(f"\n=========== {scenario} / {model} / T={T} ===========")
    print(" ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"-- cell {scenario}/{model}/T{T} rc={rc} in {time.time()-t0:.1f}s")
    return rc


def collect_summary(out_root: Path, cells, out_csv: Path):
    rows = []
    for scenario, model, T in cells:
        m_path = cell_dir(out_root, scenario, model, T) / "metrics.json"
        if not m_path.exists():
            continue
        with open(m_path) as f:
            m = json.load(f)
        rows.append({
            "scenario": scenario, "model": model, "T": T,
            "n_params": m.get("n_params"),
            "best_val_acc": m.get("best_val_acc"),
            "best_epoch": m.get("best_epoch"),
            "test_overall_acc": m.get("test_overall_acc"),
            "test_group_acc": m.get("test_group_acc"),
            "train_time_s": m.get("train_time_s"),
        })
    if not rows:
        print("[summary] no cells to summarize")
        return
    cols = list(rows[0].keys())
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"[saved] {out_csv} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["rnn", "gru", "transformer"],
                    choices=["rnn", "gru", "transformer"])
    ap.add_argument("--scenarios", nargs="+", default=["r1", "r2", "r3b"],
                    choices=["r1", "r2", "r3b"])
    ap.add_argument("--T", nargs="+", type=int, default=[100, 300, 1000])

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-kind", default="raw", choices=["raw", "byte"])

    # Transformer-only knobs that tend to need slightly bigger models
    ap.add_argument("--transformer-d-model", type=int, default=64)
    ap.add_argument("--transformer-n-layers", type=int, default=2)
    ap.add_argument("--transformer-n-heads", type=int, default=4)

    ap.add_argument("--data-dir", type=Path,
                    default=DATA_DIR / "classifier_dataset")
    ap.add_argument("--out-root", type=Path,
                    default=DATA_DIR / "analysis" / "9_classifier")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells that already have metrics.json")
    args = ap.parse_args()

    cells = list(product(args.scenarios, args.models, args.T))
    print(f"[sweep] {len(cells)} cells:")
    for c in cells:
        print(f"  {c}")

    for scenario, model, T in cells:
        if (not args.force) and is_done(args.out_root, scenario, model, T):
            print(f"[skip] {scenario}/{model}/T{T} — metrics.json exists")
            continue
        rc = run_cell(args, scenario, model, T)
        if rc != 0:
            print(f"[warn] cell {scenario}/{model}/T{T} failed (rc={rc}), continuing")

    summary_csv = args.out_root / "sweep_summary.csv"
    args.out_root.mkdir(parents=True, exist_ok=True)
    collect_summary(args.out_root, cells, summary_csv)


if __name__ == "__main__":
    main()
