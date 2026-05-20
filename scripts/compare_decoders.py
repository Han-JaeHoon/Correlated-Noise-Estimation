"""
R3b ceiling comparison across decoder variants: IdentityDecoder (= R1
baseline), LookupDecoder (multi-fault → identity), HammingNearestDecoder
(multi-fault → nearest-Hamming lookup).

Reuses src/ceiling_r3b.compute_ceiling. Output goes to
data/analysis/7_r3b_ceiling/decoder_compare/.
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


def r1_analytic_acc(p_bg, p_high, T, n_test, seed, contributions):
    dists = compute_all_distributions(p_bg, p_high, contributions)
    rng = np.random.default_rng(seed + 1_000_000)
    conf = bayes_classifier_confusion(dists=dists, T=T, n_test=n_test, rng=rng)
    pc = per_class_accuracy(conf)
    g_overall, _, _ = group_accuracy(conf)
    return float(np.diag(conf).sum() / conf.sum()), float(pc.mean()), g_overall, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=DATA_DIR / "analysis" / "7_r3b_ceiling" / "decoder_compare")
    ap.add_argument("--p-bg", type=float, default=0.01)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--T", type=int, nargs="+", default=[10, 30, 50, 100])
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[compare-decoders] p_bg={args.p_bg}, p_high={args.p_high}")
    print(f"  T ∈ {args.T}, n_train={args.n_train}, n_test={args.n_test}, seed={args.seed}")
    print(f"  out: {out_root}")
    print()

    contributions = compute_single_round_lookup(reset=True)

    rows = []
    for T in args.T:
        print(f"=== T={T} ===")

        t0 = time.time()
        r1_overall, r1_pc, r1_g, conf_r1 = r1_analytic_acc(
            args.p_bg, args.p_high, T, args.n_test * 4, args.seed,
            contributions=contributions,
        )
        print(f"  R1 analytic: per-class={r1_pc:.4f} group={r1_g:.4f}  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        run_lk = compute_ceiling(
            p_bg=args.p_bg, p_high=args.p_high,
            T_train=T, T_test=T,
            n_train=args.n_train, n_test=args.n_test,
            decoder_kind="lookup", seed=args.seed,
        )
        lk_pc = float(per_class_accuracy(run_lk.confusion).mean())
        lk_g, _, _ = group_accuracy(run_lk.confusion)
        print(f"  R3b lookup : per-class={lk_pc:.4f} group={lk_g:.4f}  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        run_hm = compute_ceiling(
            p_bg=args.p_bg, p_high=args.p_high,
            T_train=T, T_test=T,
            n_train=args.n_train, n_test=args.n_test,
            decoder_kind="hamming", seed=args.seed,
        )
        hm_pc = float(per_class_accuracy(run_hm.confusion).mean())
        hm_g, _, _ = group_accuracy(run_hm.confusion)
        print(f"  R3b hamming: per-class={hm_pc:.4f} group={hm_g:.4f}  ({time.time()-t0:.1f}s)")

        rows.append({
            "T": T, "r1_pc": r1_pc, "lookup_pc": lk_pc, "hamming_pc": hm_pc,
            "r1_group": r1_g, "lookup_group": lk_g, "hamming_group": hm_g,
        })

        np.savez_compressed(
            out_root / f"decoders_T{T}.npz",
            r1=conf_r1,
            lookup=run_lk.confusion,
            hamming=run_hm.confusion,
            lookup_marginals=run_lk.marginals,
            hamming_marginals=run_hm.marginals,
        )

    csv_path = out_root / "decoder_compare.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in rows[0].keys()) + "\n")
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    Ts = [r["T"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(Ts, [r["r1_pc"] for r in rows], "o-", label="R1 (analytic)")
    ax.plot(Ts, [r["lookup_pc"] for r in rows], "s-", label="R3b LookupDecoder")
    ax.plot(Ts, [r["hamming_pc"] for r in rows], "^-", label="R3b HammingNearestDecoder")
    ax.axhline(18/24, ls="--", color="gray", alpha=0.6, label="R1 15-Pauli ceiling 18/24")
    ax.set_xlabel("T (rounds)")
    ax.set_ylabel("Per-class accuracy (mean over 24)")
    ax.set_title(f"R1 vs R3b decoder variants  (p_bg={args.p_bg}, p_high={args.p_high})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ts, [r["r1_group"] for r in rows], "o-", label="R1 (analytic)")
    ax.plot(Ts, [r["lookup_group"] for r in rows], "s-", label="R3b LookupDecoder")
    ax.plot(Ts, [r["hamming_group"] for r in rows], "^-", label="R3b HammingNearestDecoder")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.6)
    ax.set_xlabel("T (rounds)")
    ax.set_ylabel("Group accuracy")
    ax.set_title("Group accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = out_root / "decoder_compare_curves.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"wrote {out_png}")

    (out_root / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == "__main__":
    main()
