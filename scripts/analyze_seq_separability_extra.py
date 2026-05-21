"""
Follow-up analysis on top of test_seq_separability.py.

Re-generates the same (24, N, T, 8) sequences (deterministic from --seed)
and produces robustness-corrected measures that the bare JS computation
in test_seq_separability.py cannot give:

  1. Self-baseline JS:
       Each empirical JS of cross-class (k vs k') is biased upward by the
       finite-sample noise floor. We estimate that floor by splitting the N
       samples for each k into two halves and computing the JS between the
       two halves — for the same k this is purely sampling noise. Net JS
       = cross-JS − mean(self-JS for k, k') is the signal above noise.

  2. L-marginal Bayes classifier with proper held-out test:
       Train empirical L-round marginals on the first half of samples per k,
       classify the second half. Plot per-class and group accuracy at
       L ∈ {1, 2}, on top of the marginal-only Bayes infrastructure
       already in src/ceiling_r3b.py.

  3. Cumulative log-LR distribution evolution:
       For each ambiguity-group pair (k, k') compute the running sum
       Λ_t = Σ_{s≤t} log P̂(d_s | k) - log P̂(d_s | k') on sequences from
       both true k and true k'. Plot mean ± std vs t to show whether the
       two distributions separate as T grows or stay overlapping.
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


# Re-imports from sister script (kept self-contained)


def pack_windows(syndromes: np.ndarray, L: int, drop_first: bool = True) -> np.ndarray:
    N, T, S = syndromes.shape
    start = 1 if drop_first else 0
    bit_weights = (1 << np.arange(S)).astype(np.int64)
    packed_rounds = (syndromes.astype(np.int64) * bit_weights).sum(axis=2)
    n_windows = T - L + 1 - start
    if n_windows <= 0:
        return np.empty(0, dtype=np.int64)
    out = np.zeros((N, n_windows), dtype=np.int64)
    for li in range(L):
        out += packed_rounds[:, start + li : start + li + n_windows] << (li * 8)
    return out.ravel()


def empirical_distribution(packed: np.ndarray) -> tuple:
    keys, counts = np.unique(packed, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return keys, probs


def js_divergence_sparse(k1, p1, k2, p2) -> float:
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


def generate_all_classes(p_bg, p_high, T, n_samples, seed, verbose=False):
    decoder = LookupDecoder(reset=True)
    rng_master = np.random.default_rng(seed)
    out = np.zeros((N_CNOT, n_samples, T, 8), dtype=np.uint8)
    t0 = time.time()
    for k in range(N_CNOT):
        sub_seed = int(rng_master.integers(0, 2**63 - 1))
        rng_k = np.random.default_rng(sub_seed)
        out[k] = simulate_class_sequences(
            k_dom=k, n_samples=n_samples, T=T,
            p_bg=p_bg, p_high=p_high,
            decoder_kind="lookup", rng=rng_k, decoder=decoder,
        )
        if verbose:
            print(f"  [gen] k={k:2d} ({time.time()-t0:.1f}s)")
    return out


# --- self-baseline JS ------------------------------------------------------


def self_baseline_js_per_class(syndromes: np.ndarray, L: int) -> np.ndarray:
    """
    For each class k, split N into halves and compute JS(half1 || half2).
    Returns (N_CNOT,) array — the per-class sampling noise floor.
    """
    K = syndromes.shape[0]
    N = syndromes.shape[1]
    half = N // 2
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        pa = pack_windows(syndromes[k, :half], L=L)
        pb = pack_windows(syndromes[k, half:], L=L)
        ka, qa = empirical_distribution(pa)
        kb, qb = empirical_distribution(pb)
        out[k] = js_divergence_sparse(ka, qa, kb, qb)
    return out


def cross_js_matrix_halfsize(syndromes: np.ndarray, L: int) -> np.ndarray:
    """
    Cross-class JS using only the first half of each class's samples — so
    the cross JS is computed at the same per-class sample budget as the
    self-baseline. This makes the noise-floor subtraction valid.
    """
    K = syndromes.shape[0]
    N = syndromes.shape[1]
    half = N // 2
    class_dists = []
    for k in range(K):
        pa = pack_windows(syndromes[k, :half], L=L)
        class_dists.append(empirical_distribution(pa))
    M = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            M[i, j] = M[j, i] = js_divergence_sparse(*class_dists[i], *class_dists[j])
    return M


# --- L-marginal Bayes classifier with train/test split --------------------


def build_log_marginal_dict(syndromes_train_k: np.ndarray, L: int, alpha: float = 1.0) -> dict:
    """
    Returns {key: log P̂(key | k)} as a Python dict (sparse) + a fallback
    log-prob value for unseen keys. Laplace smoothing with virtual alphabet
    of just-observed support is approximate; for unseen keys we return
    log(alpha / (n + alpha * support_size_seen_so_far)) which is a coarse
    smoothing but adequate for tie-breaking purposes.

    Returns (log_p_dict, log_p_unseen).
    """
    packed = pack_windows(syndromes_train_k, L=L)
    keys, counts = np.unique(packed, return_counts=True)
    n = int(counts.sum())
    K_seen = len(keys)
    denom = n + alpha * (K_seen + 1)  # +1 reserves mass for the "unseen" bucket
    log_p = {}
    for k_int, c in zip(keys.tolist(), counts.tolist()):
        log_p[int(k_int)] = float(np.log((c + alpha) / denom))
    log_p_unseen = float(np.log(alpha / denom))
    return log_p, log_p_unseen


def classify_with_L_marginals(
    syndromes_test: np.ndarray, log_marginals: list, L: int
) -> np.ndarray:
    """
    Marginal Bayes classifier using L-round windows.

    Args:
        syndromes_test: (N_test, T, 8) for one true k.
        log_marginals: list of length N_CNOT; each item = (log_p_dict, log_p_unseen).
        L: window size.
    Returns:
        predictions: (N_test,) int.
    """
    N_test, T, S = syndromes_test.shape
    bit_weights = (1 << np.arange(S)).astype(np.int64)
    packed_rounds = (syndromes_test.astype(np.int64) * bit_weights).sum(axis=2)  # (N_test, T)
    start = 1  # drop round 0
    n_windows = T - L + 1 - start
    if n_windows <= 0:
        return np.zeros(N_test, dtype=np.int64)

    # Build (N_test, n_windows) L-grams as int keys
    windows = np.zeros((N_test, n_windows), dtype=np.int64)
    for li in range(L):
        windows += packed_rounds[:, start + li : start + li + n_windows] << (li * 8)

    # For each candidate class, sum log P̂(window | k)
    K = len(log_marginals)
    scores = np.zeros((K, N_test), dtype=np.float64)
    for k in range(K):
        d, unseen = log_marginals[k]
        # Vectorize-light: per (i, t) lookup
        # numpy doesn't have built-in dict.get vectorized; we loop with .get
        for i in range(N_test):
            s = 0.0
            for w in windows[i]:
                s += d.get(int(w), unseen)
            scores[k, i] = s
    return scores.argmax(axis=0)


# --- cumulative log-LR -----------------------------------------------------


def per_round_log_marginals(syndromes: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Estimate per-round (L=1) log P̂(d | k) for each k from drop_first
    syndromes. Returns (N_CNOT, 256) float64 log-probs.
    """
    K = syndromes.shape[0]
    out = np.zeros((K, 256), dtype=np.float64)
    bit_weights = (1 << np.arange(8)).astype(np.int64)
    for k in range(K):
        packed = (syndromes[k, :, 1:, :].astype(np.int64) * bit_weights).sum(axis=2).ravel()
        bins = np.bincount(packed, minlength=256).astype(np.float64)
        bins += alpha
        bins /= bins.sum()
        out[k] = np.log(bins)
    return out


# --- main ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-bg", type=float, default=0.01)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--L-list", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=DATA_DIR / "analysis" / "8_seq_separability")
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--skip-logLR", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    L_list = sorted(set(args.L_list))
    tag = "_" + args.tag if args.tag else ""

    print(f"[args] p_bg={args.p_bg} p_high={args.p_high} T={args.T} N={args.n_samples} L={L_list} seed={args.seed}")
    t0 = time.time()
    print("[gen] regenerating sequences with same seed for follow-up...")
    syndromes = generate_all_classes(
        p_bg=args.p_bg, p_high=args.p_high, T=args.T,
        n_samples=args.n_samples, seed=args.seed, verbose=False,
    )
    print(f"[gen] done in {time.time()-t0:.1f}s")

    # ----- 1. self-baseline JS -----
    print()
    print("[A] Self-baseline JS (split-half within-class)")
    self_baseline = {}
    cross_half = {}
    for L in L_list:
        self_baseline[L] = self_baseline_js_per_class(syndromes, L=L)
        cross_half[L] = cross_js_matrix_halfsize(syndromes, L=L)
        sb_mean = float(np.mean(self_baseline[L]))
        print(f"  L={L}: mean self-baseline = {sb_mean:.5f} (per-class noise floor)")

    # within / between with noise floor subtracted
    within_pairs = []
    for g in R1_AMBIG_GROUPS:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                within_pairs.append((g[i], g[j]))
    within_set = set(within_pairs)
    all_pairs = [(i, j) for i in range(N_CNOT) for j in range(i + 1, N_CNOT)]
    between_pairs = [p for p in all_pairs if p not in within_set]

    print()
    summary_rows = []
    for L in L_list:
        M = cross_half[L]
        sb_per_class = self_baseline[L]
        # subtract average self-baseline of the two classes involved
        net = M - 0.5 * (sb_per_class[:, None] + sb_per_class[None, :])
        net = np.maximum(net, 0.0)  # clip slight negatives

        wg_raw = np.array([M[i, j] for i, j in within_pairs])
        bg_raw = np.array([M[i, j] for i, j in between_pairs])
        wg_net = np.array([net[i, j] for i, j in within_pairs])
        bg_net = np.array([net[i, j] for i, j in between_pairs])

        row = {
            "L": L,
            "self_baseline_mean": float(sb_per_class.mean()),
            "within_mean_raw": float(wg_raw.mean()),
            "between_mean_raw": float(bg_raw.mean()),
            "within_mean_net": float(wg_net.mean()),
            "within_min_net": float(wg_net.min()),
            "within_max_net": float(wg_net.max()),
            "between_mean_net": float(bg_net.mean()),
            "between_min_net": float(bg_net.min()),
            "between_max_net": float(bg_net.max()),
        }
        summary_rows.append(row)
        print(f"  L={L} (after self-baseline subtraction):")
        print(f"    within  : mean={row['within_mean_net']:.5f}  [{row['within_min_net']:.5f}, {row['within_max_net']:.5f}]")
        print(f"    between : mean={row['between_mean_net']:.5f}  [{row['between_min_net']:.5f}, {row['between_max_net']:.5f}]")
        print(f"    self-bl : mean={row['self_baseline_mean']:.5f}")

    # save NPZ
    out_npz = args.out_dir / f"netjs_T{args.T}_N{args.n_samples}{tag}.npz"
    np.savez(
        out_npz,
        **{f"cross_half_L{L}": cross_half[L] for L in L_list},
        **{f"self_baseline_L{L}": self_baseline[L] for L in L_list},
        L_list=np.array(L_list, dtype=np.int64),
        p_bg=args.p_bg, p_high=args.p_high,
        T=args.T, n=args.n_samples, seed=args.seed,
    )
    print(f"[saved] {out_npz}")

    csv_path = args.out_dir / f"netjs_summary_T{args.T}_N{args.n_samples}{tag}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(summary_rows[0].keys()) + "\n")
        for row in summary_rows:
            f.write(",".join(str(v) for v in row.values()) + "\n")
    print(f"[saved] {csv_path}")

    # ----- 2. L-marginal Bayes classifier with train/test split -----
    if not args.skip_classifier:
        print()
        print("[B] L-marginal Bayes classifier (train on half, test on other half)")
        N = args.n_samples
        half = N // 2
        classifier_results = {}
        for L in L_list:
            # Build training marginals
            print(f"  L={L}: building marginals from training half...")
            log_marginals = [
                build_log_marginal_dict(syndromes[k, :half], L=L, alpha=1.0)
                for k in range(N_CNOT)
            ]
            # Predict on test half
            print(f"  L={L}: classifying test half...")
            confusion = np.zeros((N_CNOT, N_CNOT), dtype=np.int64)
            for k_true in range(N_CNOT):
                preds = classify_with_L_marginals(
                    syndromes[k_true, half:], log_marginals, L=L
                )
                for kp in range(N_CNOT):
                    confusion[k_true, kp] = int((preds == kp).sum())
            per_class = np.diag(confusion).astype(np.float64) / confusion.sum(axis=1).clip(min=1)
            overall = float(np.diag(confusion).sum() / confusion.sum())
            # group accuracy with R1 ambiguity groups
            in_group = set()
            for g in R1_AMBIG_GROUPS:
                in_group.update(g)
            canonical_groups = list(R1_AMBIG_GROUPS) + [[k] for k in range(N_CNOT) if k not in in_group]
            group_of = {}
            for gi, g in enumerate(canonical_groups):
                for k in g:
                    group_of[k] = gi
            group_correct = 0
            group_total = 0
            for k_true in range(N_CNOT):
                rt = confusion[k_true].sum()
                if rt == 0:
                    continue
                for kp in range(N_CNOT):
                    if group_of[kp] == group_of[k_true]:
                        group_correct += int(confusion[k_true, kp])
                group_total += int(rt)
            group_acc = group_correct / max(group_total, 1)
            classifier_results[L] = {
                "confusion": confusion,
                "per_class": per_class,
                "overall": overall,
                "group_acc": group_acc,
            }
            print(f"  L={L}:  overall accuracy = {overall:.4f}   group accuracy = {group_acc:.4f}")

        # Save classifier results
        cls_npz = args.out_dir / f"classifier_T{args.T}_N{args.n_samples}{tag}.npz"
        np.savez(
            cls_npz,
            **{f"confusion_L{L}": classifier_results[L]["confusion"] for L in L_list},
            **{f"per_class_L{L}": classifier_results[L]["per_class"] for L in L_list},
            overall=np.array([classifier_results[L]["overall"] for L in L_list]),
            group_acc=np.array([classifier_results[L]["group_acc"] for L in L_list]),
            L_list=np.array(L_list, dtype=np.int64),
            T=args.T, n=args.n_samples, seed=args.seed,
        )
        print(f"[saved] {cls_npz}")

    # ----- 3. Cumulative log-LR -----
    if not args.skip_logLR:
        print()
        print("[C] Cumulative log-likelihood ratio for ambiguity-group pairs")

        # Use ALL N samples to estimate per-round L=1 log marginals (these
        # are well-estimated; bias is ~256/(N*T-1) ≪ 1)
        log_pks = per_round_log_marginals(syndromes, alpha=1.0)  # (24, 256)

        cum_log_lr_summary = []
        for g in R1_AMBIG_GROUPS:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    k1, k2 = g[i], g[j]
                    delta_logp = log_pks[k1] - log_pks[k2]  # (256,) per-round log-LR weights
                    bit_weights = (1 << np.arange(8)).astype(np.int64)
                    # For sequences from k1
                    packed_k1 = (syndromes[k1, :, 1:, :].astype(np.int64) * bit_weights).sum(axis=2)
                    lr_k1 = delta_logp[packed_k1]  # (n, T-1)
                    cum_k1 = np.cumsum(lr_k1, axis=1)
                    # For sequences from k2
                    packed_k2 = (syndromes[k2, :, 1:, :].astype(np.int64) * bit_weights).sum(axis=2)
                    lr_k2 = delta_logp[packed_k2]
                    cum_k2 = np.cumsum(lr_k2, axis=1)
                    # mean ± std of cum at end
                    mean_T_k1 = float(cum_k1[:, -1].mean())
                    std_T_k1 = float(cum_k1[:, -1].std())
                    mean_T_k2 = float(cum_k2[:, -1].mean())
                    std_T_k2 = float(cum_k2[:, -1].std())
                    cohen = (mean_T_k1 - mean_T_k2) / (0.5 * (std_T_k1 + std_T_k2) + 1e-30)
                    cum_log_lr_summary.append({
                        "k1": k1, "k2": k2,
                        "mean_T_k1": mean_T_k1, "std_T_k1": std_T_k1,
                        "mean_T_k2": mean_T_k2, "std_T_k2": std_T_k2,
                        "gap": mean_T_k1 - mean_T_k2, "cohen_d": cohen,
                    })
                    print(f"  ({k1},{k2}): from k1 mean={mean_T_k1:.2f}±{std_T_k1:.2f}  "
                          f"from k2 mean={mean_T_k2:.2f}±{std_T_k2:.2f}  "
                          f"gap={mean_T_k1-mean_T_k2:.2f}  cohen_d={cohen:.3f}")

        logLR_csv = args.out_dir / f"cumulative_logLR_T{args.T}_N{args.n_samples}{tag}.csv"
        with open(logLR_csv, "w") as f:
            f.write("k1,k2,mean_T_k1,std_T_k1,mean_T_k2,std_T_k2,gap,cohen_d\n")
            for r in cum_log_lr_summary:
                f.write(f"{r['k1']},{r['k2']},{r['mean_T_k1']},{r['std_T_k1']},"
                        f"{r['mean_T_k2']},{r['std_T_k2']},{r['gap']},{r['cohen_d']}\n")
        print(f"[saved] {logLR_csv}")

        # plot evolution for the first ambiguity pair
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, len(R1_AMBIG_GROUPS), figsize=(15, 4), sharey=False)
            for gi, g in enumerate(R1_AMBIG_GROUPS):
                k1, k2 = g[0], g[1]
                ax = axes[gi]
                delta_logp = log_pks[k1] - log_pks[k2]
                bit_weights = (1 << np.arange(8)).astype(np.int64)
                packed_k1 = (syndromes[k1, :, 1:, :].astype(np.int64) * bit_weights).sum(axis=2)
                lr_k1 = delta_logp[packed_k1]
                cum_k1 = np.cumsum(lr_k1, axis=1)
                packed_k2 = (syndromes[k2, :, 1:, :].astype(np.int64) * bit_weights).sum(axis=2)
                lr_k2 = delta_logp[packed_k2]
                cum_k2 = np.cumsum(lr_k2, axis=1)
                ts = np.arange(1, cum_k1.shape[1] + 1)
                m_k1 = cum_k1.mean(axis=0)
                s_k1 = cum_k1.std(axis=0)
                m_k2 = cum_k2.mean(axis=0)
                s_k2 = cum_k2.std(axis=0)
                ax.plot(ts, m_k1, color="C0", label=f"true k={k1}")
                ax.fill_between(ts, m_k1 - s_k1, m_k1 + s_k1, color="C0", alpha=0.2)
                ax.plot(ts, m_k2, color="C3", label=f"true k={k2}")
                ax.fill_between(ts, m_k2 - s_k2, m_k2 + s_k2, color="C3", alpha=0.2)
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xlabel("t (rounds)")
                ax.set_ylabel(f"Σ log P({k1}|d) - log P({k2}|d)")
                ax.set_title(f"Group G{gi+1}: ({k1},{k2})")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            plt.suptitle(f"Cumulative log-LR — does it diverge as T grows?\n"
                         f"p_bg={args.p_bg}, p_high={args.p_high}, N={args.n_samples}")
            plt.tight_layout()
            logLR_plot = args.out_dir / f"cumulative_logLR_T{args.T}_N{args.n_samples}{tag}.png"
            plt.savefig(logLR_plot, dpi=140)
            plt.close()
            print(f"[saved] {logLR_plot}")
        except ImportError:
            pass

    print()
    print("done.")


if __name__ == "__main__":
    main()
