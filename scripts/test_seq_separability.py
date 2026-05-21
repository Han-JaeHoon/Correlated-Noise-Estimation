"""
Directly test whether R3b syndrome-sequence distributions of different
dominant CNOTs are distinguishable, WITHOUT going through a classifier.

Method
------
For each candidate dominant CNOT k in 0..23, sample N independent T-round
sequences under the R3b LookupDecoder. For each window size L in
{1, 2, 3}, build the empirical L-round marginal P(d_t, d_{t+1}, ..., d_{t+L-1} | k)
by sliding L-grams over the (T - 1) post-round-0 syndromes (round 0 is
decoder-agnostic since the frame always starts empty). For every pair
(k, k') compute the Jensen–Shannon divergence between the two empirical
L-marginals.

Three outputs answer "T 늘면 sequence가 unique 해지는가?":

  1. JS(L) per (k, k') — if this grows with L within R1 ambiguity-group
     pairs, longer windows carry additional distinguishing information,
     so a long-enough sequence becomes unique per k.
  2. Within-group vs between-group mean / max / min JS — at each L, are
     ambiguity-group pairs (the suspect indistinguishable ones) catching
     up with the easy between-group pairs?
  3. Heatmaps at L=1 vs L=L_max — visual collapse of group structure.

Outputs land in `data/analysis/8_seq_separability/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.ceiling_r3b import N_CNOT, simulate_class_sequences
from src.config import DATA_DIR
from src.decoder import LookupDecoder


R1_AMBIG_GROUPS = [
    [6, 7],
    [14, 15, 17],
    [18, 19, 20],
    [22, 23],
]


# --- empirical-distribution helpers ----------------------------------------


def pack_windows(syndromes: np.ndarray, L: int, drop_first: bool = True) -> np.ndarray:
    """
    Slide L-round windows over a (N, T, 8) syndrome stream and return a 1D
    int64 array of packed L-grams (8L bits each).

    drop_first=True skips round 0 because the data-qubit Pauli frame is
    always 0 at that point under R3b, so its syndrome distribution is
    decoder-agnostic and identical for R1 / R3b. The R3b-specific behavior
    only starts from round 1.
    """
    N, T, S = syndromes.shape
    if S != 8:
        raise ValueError(f"expected 8 stabilizers, got {S}")
    start = 1 if drop_first else 0
    bit_weights = (1 << np.arange(S)).astype(np.int64)
    packed_rounds = (syndromes.astype(np.int64) * bit_weights).sum(axis=2)  # (N, T)

    n_windows = T - L + 1 - start
    if n_windows <= 0:
        return np.empty(0, dtype=np.int64)

    out = np.zeros((N, n_windows), dtype=np.int64)
    for li in range(L):
        out += packed_rounds[:, start + li : start + li + n_windows] << (li * 8)
    return out.ravel()


def empirical_distribution(packed: np.ndarray) -> tuple:
    """Return (keys, probs) such that probs sums to 1 over the observed support."""
    keys, counts = np.unique(packed, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return keys, probs


def js_divergence_sparse(k1: np.ndarray, p1: np.ndarray, k2: np.ndarray, p2: np.ndarray) -> float:
    """
    JS divergence between two sparse discrete distributions. 0·log(0)=0
    handled by skipping zeros, so missing keys contribute only via the
    mixture m = 0.5 (p+q).

    Bounded in [0, log 2].
    """
    d1 = dict(zip(k1.tolist(), p1.tolist()))
    d2 = dict(zip(k2.tolist(), p2.tolist()))
    union = set(d1) | set(d2)
    js = 0.0
    for k in union:
        a = d1.get(k, 0.0)
        b = d2.get(k, 0.0)
        m = 0.5 * (a + b)
        if a > 0.0:
            js += 0.5 * a * (np.log(a) - np.log(m))
        if b > 0.0:
            js += 0.5 * b * (np.log(b) - np.log(m))
    return float(js)


def pairwise_js_matrix(class_dists: list) -> np.ndarray:
    K = len(class_dists)
    M = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            d = js_divergence_sparse(*class_dists[i], *class_dists[j])
            M[i, j] = M[j, i] = d
    return M


# --- summary stats ---------------------------------------------------------


def collect_pair_lists():
    within = []
    for g in R1_AMBIG_GROUPS:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                within.append((g[i], g[j]))
    within_set = set(within)
    all_pairs = [(i, j) for i in range(N_CNOT) for j in range(i + 1, N_CNOT)]
    between = [p for p in all_pairs if p not in within_set]
    return within, between, all_pairs


def summarize(js_matrices: dict, L_list: list) -> list:
    within, between, all_pairs = collect_pair_lists()
    rows = []
    for L in L_list:
        M = js_matrices[L]
        wg = np.array([M[i, j] for i, j in within])
        bg = np.array([M[i, j] for i, j in between])
        rows.append({
            "L": L,
            "within_n": len(within),
            "within_min": float(wg.min()),
            "within_mean": float(wg.mean()),
            "within_max": float(wg.max()),
            "between_n": len(between),
            "between_min": float(bg.min()),
            "between_mean": float(bg.mean()),
            "between_max": float(bg.max()),
            "ratio_between_over_within": float(bg.mean() / max(wg.mean(), 1e-30)),
        })
    return rows


# --- generation ------------------------------------------------------------


def generate_all_classes(
    p_bg: float,
    p_high: float,
    T: int,
    n_samples: int,
    seed: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Returns: syndromes (24, n_samples, T, 8) uint8.
    """
    decoder = LookupDecoder(reset=True)
    rng_master = np.random.default_rng(seed)
    out = np.zeros((N_CNOT, n_samples, T, 8), dtype=np.uint8)
    t0 = time.time()
    for k in range(N_CNOT):
        sub_seed = int(rng_master.integers(0, 2**63 - 1))
        rng_k = np.random.default_rng(sub_seed)
        out[k] = simulate_class_sequences(
            k_dom=k,
            n_samples=n_samples,
            T=T,
            p_bg=p_bg,
            p_high=p_high,
            decoder_kind="lookup",
            rng=rng_k,
            decoder=decoder,
        )
        if verbose:
            print(f"  [gen] k={k:2d} done ({time.time()-t0:.1f}s)")
    return out


# --- plots -----------------------------------------------------------------


def plot_js_curves(js_matrices: dict, L_list: list, args, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    Ls = list(L_list)
    group_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    # Individual within-group pairs
    for gi, g in enumerate(R1_AMBIG_GROUPS):
        col = group_colors[gi % len(group_colors)]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ki, kj = g[i], g[j]
                vals = [js_matrices[L][ki, kj] for L in Ls]
                ax.plot(Ls, vals, marker="o", color=col, alpha=0.85,
                        label=f"G{gi+1} ({ki},{kj})")

    # Between-group mean / min as black reference lines
    _, between, _ = collect_pair_lists()
    bg_means = [np.mean([js_matrices[L][i, j] for i, j in between]) for L in Ls]
    bg_mins = [np.min([js_matrices[L][i, j] for i, j in between]) for L in Ls]
    ax.plot(Ls, bg_means, marker="s", color="black", lw=2, label="between-group mean")
    ax.plot(Ls, bg_mins, marker="^", color="gray", lw=1.5, ls="--",
            label="between-group min")

    ax.set_xlabel("L (window size, rounds)")
    ax.set_ylabel("Pairwise JS divergence")
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.set_title(
        f"R3b: L-round marginal JS divergence — within R1 ambiguity groups vs between\n"
        f"p_bg={args.p_bg}, p_high={args.p_high}, T={args.T}, N={args.n_samples}, seed={args.seed}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_heatmap(js_matrix: np.ndarray, L: int, args, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7))
    M = js_matrix.copy()
    # log scale, clip floor for visibility
    floor = 1e-6
    Mlog = np.log10(np.maximum(M, floor))
    im = ax.imshow(Mlog, cmap="viridis", vmin=np.log10(floor))
    ax.set_xlabel("k'")
    ax.set_ylabel("k")
    ax.set_title(
        f"log10 pairwise JS divergence at L={L}\n"
        f"p_bg={args.p_bg}, p_high={args.p_high}, T={args.T}, N={args.n_samples}"
    )
    cb = plt.colorbar(im)
    cb.set_label("log10 JS")
    # outline ambiguity groups with red boxes
    for g in R1_AMBIG_GROUPS:
        gmin, gmax = min(g), max(g)
        for k in g:
            ax.add_patch(
                plt.Rectangle((gmin - 0.5, k - 0.5),
                              gmax - gmin + 1, 1,
                              fill=False, edgecolor="red", lw=0.6, alpha=0.7)
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# --- main ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-bg", type=float, default=0.01)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--L-list", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=DATA_DIR / "analysis" / "8_seq_separability")
    ap.add_argument("--tag", type=str, default="",
                    help="optional suffix on output filenames")
    ap.add_argument("--save-raw", action="store_true",
                    help="save full (24, N, T, 8) syndromes (can be large)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    L_list = sorted(set(args.L_list))
    tag = args.tag if not args.tag or args.tag.startswith("_") else "_" + args.tag

    print(f"[args] p_bg={args.p_bg} p_high={args.p_high} T={args.T} "
          f"N={args.n_samples} L={L_list} seed={args.seed}")
    print(f"[out]  {args.out_dir}")

    # 1. generate
    t0 = time.time()
    syndromes = generate_all_classes(
        p_bg=args.p_bg, p_high=args.p_high, T=args.T,
        n_samples=args.n_samples, seed=args.seed,
    )
    gen_seconds = time.time() - t0
    print(f"[gen] done in {gen_seconds:.1f}s")

    if args.save_raw:
        raw_path = (args.out_dir /
                    f"sequences_T{args.T}_N{args.n_samples}_pbg{args.p_bg}_phigh{args.p_high}{tag}.npz")
        np.savez_compressed(raw_path,
                            syndromes=syndromes,
                            p_bg=args.p_bg, p_high=args.p_high,
                            T=args.T, n=args.n_samples, seed=args.seed)
        print(f"[saved] {raw_path} ({raw_path.stat().st_size/1e6:.1f} MB)")

    # 2. L-marginals + pairwise JS for each L
    js_matrices = {}
    for L in L_list:
        print(f"[L={L}] building empirical L-marginals and JS matrix")
        class_dists = []
        for k in range(N_CNOT):
            packed = pack_windows(syndromes[k], L=L, drop_first=True)
            class_dists.append(empirical_distribution(packed))
        M = pairwise_js_matrix(class_dists)
        js_matrices[L] = M
        positive = M[M > 0]
        if positive.size:
            print(f"  JS range: [{positive.min():.5f}, {M.max():.5f}]  mean(>0): {positive.mean():.5f}")

    # 3. save
    js_path = (args.out_dir /
               f"pairwise_js_T{args.T}_N{args.n_samples}{tag}.npz")
    np.savez(js_path,
             **{f"L{L}": js_matrices[L] for L in L_list},
             L_list=np.array(L_list, dtype=np.int64),
             p_bg=args.p_bg, p_high=args.p_high,
             T=args.T, n=args.n_samples, seed=args.seed)
    print(f"[saved] {js_path}")

    # summary
    summary = summarize(js_matrices, L_list)
    summary_path = (args.out_dir /
                    f"summary_T{args.T}_N{args.n_samples}{tag}.csv")
    keys = list(summary[0].keys())
    with open(summary_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in summary:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    print(f"[saved] {summary_path}")

    # also a json with args + summary for record
    args_path = (args.out_dir /
                 f"args_T{args.T}_N{args.n_samples}{tag}.json")
    with open(args_path, "w") as f:
        json.dump({
            "p_bg": args.p_bg, "p_high": args.p_high,
            "T": args.T, "n_samples": args.n_samples,
            "L_list": L_list, "seed": args.seed,
            "gen_seconds": gen_seconds,
            "summary": summary,
        }, f, indent=2)
    print(f"[saved] {args_path}")

    # 4. plots
    curves_path = (args.out_dir /
                   f"js_curves_T{args.T}_N{args.n_samples}{tag}.png")
    plot_js_curves(js_matrices, L_list, args, curves_path)
    print(f"[saved] {curves_path}")

    L_max = L_list[-1]
    heat_path = (args.out_dir /
                 f"js_heatmap_L{L_max}_T{args.T}_N{args.n_samples}{tag}.png")
    plot_heatmap(js_matrices[L_max], L_max, args, heat_path)
    print(f"[saved] {heat_path}")

    if 1 in L_list:
        heat1_path = (args.out_dir /
                      f"js_heatmap_L1_T{args.T}_N{args.n_samples}{tag}.png")
        plot_heatmap(js_matrices[1], 1, args, heat1_path)
        print(f"[saved] {heat1_path}")

    # 5. print final summary
    print()
    print("===== SUMMARY =====")
    for row in summary:
        print(
            f"L={row['L']}:  within mean={row['within_mean']:.5f} "
            f"[{row['within_min']:.5f}, {row['within_max']:.5f}]   "
            f"between mean={row['between_mean']:.5f}   "
            f"ratio(between/within)={row['ratio_between_over_within']:.1f}"
        )
    print()
    print("Interpretation hints:")
    print("  * If within-group mean GROWS with L, longer windows leak more")
    print("    info; sequence-level uniqueness across (k,k') is achievable.")
    print("  * If within-group mean stays ~constant with L, only L=1 info")
    print("    separates the groups; full-sequence likelihood cannot do")
    print("    much better than marginal-Bayes.")


if __name__ == "__main__":
    main()
