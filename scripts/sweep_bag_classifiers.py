"""
Full sweep: model × N grid for bag-of-shots classifiers.

Grid:
  models : empirical_bayes, logreg, mlp, deepsets
  N      : 1, 3, 10, 30, 100, 300

Runs each cell, collects metrics.json, writes sweep_summary.csv.

Usage:
    python scripts/sweep_bag_classifiers.py
    python scripts/sweep_bag_classifiers.py --models logreg mlp --N-list 10 100
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "set_classifier"

DEFAULT_MODELS = ["empirical_bayes", "logreg", "mlp", "deepsets"]
DEFAULT_N = [1, 3, 10, 30, 100, 300]


def run_cell(model, N, epochs=80):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_bag_classifier.py"),
        "--model", model,
        "--N", str(N),
        "--epochs", str(epochs),
    ]
    if model == "empirical_bayes":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_bag_classifier.py"),
            "--model", "empirical_bayes",
            "--N", str(N),
        ]
    print(f"\n{'='*50}")
    print(f"  model={model}  N={N}")
    print(f"{'='*50}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        return None
    return elapsed


def load_metric(model, N):
    if model == "empirical_bayes":
        p = OUT_ROOT / "empirical_bayes" / "metrics.json"
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        r = d["results"].get(str(N))
        if r is None:
            return None
        return {
            "model": model, "N": N,
            "overall_acc": r["accuracy"],
            "best_val_acc": "-",
            "ambiguity_group_acc": "-",
        }
    else:
        p = OUT_ROOT / f"{model}_N{N}" / "metrics.json"
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        return {
            "model": model, "N": N,
            "overall_acc": d.get("overall_acc"),
            "best_val_acc": d.get("best_val_acc"),
            "ambiguity_group_acc": d.get("ambiguity_group_acc"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--N-list", nargs="+", type=int, default=DEFAULT_N)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip training, just collect existing metrics")
    args = parser.parse_args()

    if not args.summary_only:
        for model in args.models:
            for N in args.N_list:
                # empirical_bayes: one run covers all N
                if model == "empirical_bayes" and N != args.N_list[0]:
                    continue
                if model == "empirical_bayes":
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "train_bag_classifier.py"),
                        "--model", "empirical_bayes",
                        "--N-sweep",
                    ]
                    print(f"\n{'='*50}")
                    print(f"  model=empirical_bayes  (all N)")
                    print(f"{'='*50}")
                    subprocess.run(cmd)
                else:
                    run_cell(model, N, epochs=args.epochs)

    # Collect summary
    print("\n\n=== SWEEP SUMMARY ===")
    rows = []
    header = f"{'model':<18} {'N':>4}  {'overall_acc':>11}  {'best_val':>8}"
    print(header)
    print("-" * len(header))

    for model in args.models:
        for N in args.N_list:
            m = load_metric(model, N)
            if m is None:
                print(f"  {model:<16} N={N:>4}  [no data]")
                continue
            acc = m["overall_acc"]
            val = m["best_val_acc"]
            print(f"  {model:<16} N={N:>4}  acc={acc:.4f}  val={val}")
            rows.append(m)

    # Write CSV
    import csv
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "N", "overall_acc", "best_val_acc", "ambiguity_group_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
