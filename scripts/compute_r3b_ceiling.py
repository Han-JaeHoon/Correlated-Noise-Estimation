"""
Compute the R3b LookupDecoder marginal-Bayes ceiling on a small (p_bg,
p_high, T) grid and compare to the R1 analytic ceiling on the same grid.

Output:
    data/analysis/7_r3b_ceiling/
        r3b_grid.csv                 — one row per (p_bg, p_high, T): both
                                       R1 (analytic) and R3b (MC) per-class
                                       and group-level accuracies.
        r3b_marginals_pbg{...}_phigh{...}_T{...}.npz
                                     — per-cell raw marginals + confusion
                                       (for replotting / debugging).
        r3b_compare_curve.png        — per-class accuracy vs T at one
                                       (p_bg, p_high) line, R1 vs R3b.
        r3b_group_compare.png        — group-level accuracy vs T.
        r3b_confusion_diffs.csv      — R3b confusion - R1 confusion per
                                       ambiguity-group cell (signed).

Default grid is intentionally small so a full sweep finishes in well
under one hour at p_bg=0.01, p_high=0.1, T∈{10, 30, 50}, n_test=50,
n_train=80.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.ceiling import (
    bayes_classifier_confusion,
    compute_all_distributions,
    compute_single_round_lookup,
)
from src.ceiling_r3b import (
    compute_ceiling,
    group_accuracy,
    per_class_accuracy,
)
from src.config import DATA_DIR


def r1_analytic_run(p_bg: float, p_high: float, T: int, n_test: int, seed: int,
                    contributions: np.ndarray) -> np.ndarray:
    """R1 ceiling via analytic XOR-convolution + MC Bayes classifier."""
    dists = compute_all_distributions(p_bg, p_high, contributions)
    rng = np.random.default_rng(seed + 1_000_000)
    return bayes_classifier_confusion(dists=dists, T=T, n_test=n_test, rng=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=DATA_DIR / "analysis" / "7_r3b_ceiling")
    ap.add_argument("--p-bg", type=float, nargs="+", default=[0.01])
    ap.add_argument("--p-high", type=float, nargs="+", default=[0.1])
    ap.add_argument("--T", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test-r1", type=int, default=2000,
                    help="MC samples for the R1 analytic ceiling estimate "
                         "(cheap — no PennyLane circuit involved)")
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if len(args.p_bg) != len(args.p_high):
        raise SystemExit("--p-bg and --p-high must have the same length")

    print(f"[r3b-ceiling] R1 analytic vs R3b MC marginal-Bayes")
    print(f"  out: {out_root}")
    print(f"  grid: p=(bg,high) ∈ {list(zip(args.p_bg, args.p_high))}, T ∈ {args.T}")
    print(f"  R3b sim: n_train={args.n_train}, n_test={args.n_test}, seed={args.seed}")
    print()

    print("[r3b-ceiling] Building 24x15 lookup (PL)... ", end="", flush=True)
    t0 = time.time()
    contributions = compute_single_round_lookup(reset=True)
    print(f"{time.time()-t0:.1f}s")

    rows = []
    for p_bg, p_high in zip(args.p_bg, args.p_high):
        for T in args.T:
            tag = f"pbg{p_bg}_phigh{p_high}_T{T}"
            print(f"=== ({p_bg}, {p_high}, T={T}) ===")

            # R1
            t0 = time.time()
            conf_r1 = r1_analytic_run(p_bg, p_high, T, args.n_test_r1, args.seed,
                                      contributions=contributions)
            r1_overall = float(np.diag(conf_r1).sum() / conf_r1.sum())
            r1_pc = per_class_accuracy(conf_r1)
            r1_g_overall, r1_per_g, groups = group_accuracy(conf_r1)
            print(f"  R1 analytic: per-class mean={r1_pc.mean():.4f}, "
                  f"overall={r1_overall:.4f}, group={r1_g_overall:.4f}  "
                  f"({time.time()-t0:.1f}s)")

            # R3b
            t0 = time.time()
            run = compute_ceiling(
                p_bg=p_bg, p_high=p_high,
                T_train=T, T_test=T,
                n_train=args.n_train, n_test=args.n_test,
                decoder_kind="lookup", seed=args.seed,
                drop_first_round=True,
            )
            r3b_overall = float(np.diag(run.confusion).sum() / run.confusion.sum())
            r3b_pc = per_class_accuracy(run.confusion)
            r3b_g_overall, r3b_per_g, _ = group_accuracy(run.confusion)
            print(f"  R3b MC      : per-class mean={r3b_pc.mean():.4f}, "
                  f"overall={r3b_overall:.4f}, group={r3b_g_overall:.4f}  "
                  f"({time.time()-t0:.1f}s)")

            np.savez_compressed(
                out_root / f"r3b_marginals_{tag}.npz",
                marginals=run.marginals,
                confusion_r3b=run.confusion,
                confusion_r1=conf_r1,
            )

            row = {
                "p_bg": p_bg, "p_high": p_high, "T": T,
                "r1_overall": r1_overall, "r3b_overall": r3b_overall,
                "r1_per_class_mean": float(r1_pc.mean()),
                "r3b_per_class_mean": float(r3b_pc.mean()),
                "r1_group": r1_g_overall, "r3b_group": r3b_g_overall,
            }
            # per-class accuracy difference for each ambiguity group
            ambig_groups = [g for g in groups if len(g) > 1]
            for g in ambig_groups:
                tag_g = "_".join(str(k) for k in g)
                row[f"r1_pc_group_{tag_g}_mean"] = float(np.mean([r1_pc[k] for k in g]))
                row[f"r3b_pc_group_{tag_g}_mean"] = float(np.mean([r3b_pc[k] for k in g]))
            rows.append(row)

            # Write CSV header on first row to keep things tidy
    csv_path = out_root / "r3b_grid.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print()
        print(f"[r3b-ceiling] wrote {csv_path}")

    # Plots — only if at least 2 T values for one (pbg, phigh) line
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    # Per (pbg, phigh) line: per-class mean acc vs T, R1 vs R3b
    for p_bg, p_high in zip(args.p_bg, args.p_high):
        relevant = [r for r in rows if r["p_bg"] == p_bg and r["p_high"] == p_high]
        if len(relevant) < 2:
            continue
        relevant.sort(key=lambda r: r["T"])
        Ts = [r["T"] for r in relevant]
        plt.figure(figsize=(7, 4.5))
        plt.plot(Ts, [r["r1_per_class_mean"] for r in relevant], "o-",
                 label="R1 (analytic)")
        plt.plot(Ts, [r["r3b_per_class_mean"] for r in relevant], "s-",
                 label="R3b (MC, LookupDecoder)")
        plt.axhline(18/24, ls="--", color="gray", alpha=0.6,
                    label="R1 15-Pauli ceiling 18/24")
        plt.xlabel("T (rounds)")
        plt.ylabel("Per-class accuracy (mean over 24 CNOTs)")
        plt.title(f"R1 vs R3b ceiling  (p_bg={p_bg}, p_high={p_high})")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = out_root / f"r3b_compare_curve_pbg{p_bg}_phigh{p_high}.png"
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"[r3b-ceiling] wrote {path}")

        # group-level
        plt.figure(figsize=(7, 4.5))
        plt.plot(Ts, [r["r1_group"] for r in relevant], "o-", label="R1 (analytic)")
        plt.plot(Ts, [r["r3b_group"] for r in relevant], "s-", label="R3b (MC)")
        plt.axhline(1.0, ls="--", color="gray", alpha=0.6, label="ceiling 1.0")
        plt.xlabel("T (rounds)")
        plt.ylabel("Group accuracy")
        plt.title(f"R1 vs R3b group accuracy  (p_bg={p_bg}, p_high={p_high})")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = out_root / f"r3b_group_compare_pbg{p_bg}_phigh{p_high}.png"
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"[r3b-ceiling] wrote {path}")

    (out_root / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == "__main__":
    main()
