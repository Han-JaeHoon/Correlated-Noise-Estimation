"""
Compute and plot Cohen's d(t) vs t for each R1 ambiguity-group pair, using
R3b sequences. Fit d(t) ~ a * sqrt(t) (Stein's-lemma-consistent scaling)
and project the T_required to reach d=2 (≈95% separation under Gaussian
approximation).

This is the cleanest empirical answer to "do R3b syndrome sequences become
unique per CNOT as T grows?" The two cumulative log-LR distributions for
true k1 and true k2 separate at rate sqrt(T) under marginal-only (L=1)
Bayes; if the per-round KL rate is positive, T -> infinity gives perfect
discrimination.

Reads no external file: regenerates the same sequences as
test_seq_separability.py for deterministic --seed.
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
from src.decoder import LookupDecoder, PhenomDecoder, IdentityDecoder


def build_decoder(name: str):
    if name == "lookup":
        return LookupDecoder(reset=True), "lookup"
    if name == "phenom":
        return PhenomDecoder(reset=True), "phenom"
    if name == "identity":
        return IdentityDecoder(), "identity"
    raise ValueError(f"unknown decoder: {name!r}")


R1_AMBIG_GROUPS = [
    [6, 7],
    [14, 15, 17],
    [18, 19, 20],
    [22, 23],
]


def generate_all_classes(p_bg, p_high, T, n_samples, seed, decoder_name="lookup"):
    decoder, decoder_kind = build_decoder(decoder_name)
    rng_master = np.random.default_rng(seed)
    out = np.zeros((N_CNOT, n_samples, T, 8), dtype=np.uint8)
    t0 = time.time()
    for k in range(N_CNOT):
        sub_seed = int(rng_master.integers(0, 2**63 - 1))
        rng_k = np.random.default_rng(sub_seed)
        out[k] = simulate_class_sequences(
            k_dom=k, n_samples=n_samples, T=T,
            p_bg=p_bg, p_high=p_high,
            decoder_kind=decoder_kind, rng=rng_k, decoder=decoder,
        )
        print(f"  [gen] k={k:2d} ({time.time()-t0:.1f}s)")
    return out


def per_round_log_marginals(syndromes: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(N_CNOT, 256) log-marginal P(d | k), L=1, drop_first."""
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


def cumulative_logLR_trajectory(syndromes_k: np.ndarray, log_pks_k1: np.ndarray, log_pks_k2: np.ndarray):
    """
    For one true class's syndromes, return cumulative log-LR per round.

    Returns: (N, T-1) array.  Time axis = post-round-0 indices 1..T-1.
    """
    bit_weights = (1 << np.arange(8)).astype(np.int64)
    packed = (syndromes_k[:, 1:, :].astype(np.int64) * bit_weights).sum(axis=2)  # (N, T-1)
    delta = log_pks_k1 - log_pks_k2  # (256,)
    lr = delta[packed]  # (N, T-1)
    return np.cumsum(lr, axis=1)


def cohen_d_traj(cum_from_k1: np.ndarray, cum_from_k2: np.ndarray) -> np.ndarray:
    """
    Pooled-Cohen's-d trajectory.
    cum_from_k1, cum_from_k2: each (N, T-1).
    Returns (T-1,) array of Cohen's d at each t.
    """
    m1 = cum_from_k1.mean(axis=0)
    m2 = cum_from_k2.mean(axis=0)
    s1 = cum_from_k1.std(axis=0, ddof=1)
    s2 = cum_from_k2.std(axis=0, ddof=1)
    s_pool = 0.5 * (s1 + s2)
    d = (m1 - m2) / np.maximum(s_pool, 1e-30)
    return d


def fit_sqrt_scaling(t: np.ndarray, d: np.ndarray, t_min: int = 30) -> tuple:
    """Fit d(t) = a*sqrt(t) on t >= t_min via OLS in (sqrt(t), d) space."""
    mask = t >= t_min
    x = np.sqrt(t[mask].astype(np.float64))
    y = d[mask].astype(np.float64)
    # No intercept: y = a x
    a = float((x * y).sum() / (x * x).sum())
    resid = y - a * x
    rmse = float(np.sqrt((resid ** 2).mean()))
    return a, rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-bg", type=float, default=0.01)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-target", type=float, default=2.0,
                    help="Cohen's d to project T_required at (default 2.0 ≈ 95% sep)")
    ap.add_argument("--out-dir", type=Path,
                    default=DATA_DIR / "analysis" / "8_seq_separability")
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--decoder", default="lookup",
                    choices=["lookup", "phenom", "identity"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_" + args.tag if args.tag else ""

    print(f"[args] p_bg={args.p_bg} p_high={args.p_high} T={args.T} N={args.n_samples} seed={args.seed} decoder={args.decoder}")
    print(f"[gen] regenerating ...")
    t0 = time.time()
    syndromes = generate_all_classes(args.p_bg, args.p_high, args.T, args.n_samples, args.seed,
                                      decoder_name=args.decoder)
    print(f"[gen] done in {time.time()-t0:.1f}s")

    # build per-round (L=1) log marginals
    log_pks = per_round_log_marginals(syndromes, alpha=1.0)

    # for each R1 ambiguity-group pair, compute d(t) trajectory
    pairs = []
    for g in R1_AMBIG_GROUPS:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pairs.append((g[i], g[j]))

    ts = np.arange(1, args.T)  # cumulative log-LR computed over rounds 1..T-1
    d_trajs = {}
    fits = {}
    for (k1, k2) in pairs:
        c1 = cumulative_logLR_trajectory(syndromes[k1], log_pks[k1], log_pks[k2])
        c2 = cumulative_logLR_trajectory(syndromes[k2], log_pks[k1], log_pks[k2])
        d = cohen_d_traj(c1, c2)
        d_trajs[(k1, k2)] = d
        a, rmse = fit_sqrt_scaling(ts, d, t_min=30)
        T_required = (args.d_target / a) ** 2 if a > 0 else float("inf")
        fits[(k1, k2)] = {"a": a, "rmse": rmse, "T_required_for_target": T_required}
        print(f"  ({k1},{k2}): d(T={args.T})={d[-1]:.3f}   "
              f"sqrt-fit a={a:.4f}, rmse={rmse:.3f}   "
              f"T_required(d={args.d_target}) = {T_required:.0f}")

    # save
    out_npz = args.out_dir / f"d_scaling_T{args.T}_N{args.n_samples}{tag}.npz"
    np.savez(
        out_npz,
        ts=ts,
        **{f"d_{k1}_{k2}": d_trajs[(k1, k2)] for (k1, k2) in pairs},
        pairs=np.array(pairs),
        a=np.array([fits[p]["a"] for p in pairs]),
        rmse=np.array([fits[p]["rmse"] for p in pairs]),
        T_required=np.array([fits[p]["T_required_for_target"] for p in pairs]),
        d_target=args.d_target,
    )
    print(f"[saved] {out_npz}")

    # CSV summary
    csv_path = args.out_dir / f"d_scaling_summary_T{args.T}_N{args.n_samples}{tag}.csv"
    with open(csv_path, "w") as f:
        f.write("k1,k2,d_at_T,sqrt_fit_a,sqrt_fit_rmse,T_required_for_d_target,d_target\n")
        for (k1, k2) in pairs:
            ft = fits[(k1, k2)]
            f.write(f"{k1},{k2},{d_trajs[(k1,k2)][-1]:.4f},{ft['a']:.5f},"
                    f"{ft['rmse']:.4f},{ft['T_required_for_target']:.1f},{args.d_target}\n")
    print(f"[saved] {csv_path}")

    # plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    group_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    # color by group
    pair_to_color = {}
    for gi, g in enumerate(R1_AMBIG_GROUPS):
        col = group_colors[gi]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pair_to_color[(g[i], g[j])] = col

    for (k1, k2) in pairs:
        d = d_trajs[(k1, k2)]
        col = pair_to_color[(k1, k2)]
        ax.plot(ts, d, color=col, lw=1.5, alpha=0.85, label=f"({k1},{k2})")
        # sqrt fit overlay
        a = fits[(k1, k2)]["a"]
        ax.plot(ts, a * np.sqrt(ts), color=col, lw=0.7, ls="--", alpha=0.6)

    ax.axhline(args.d_target, color="black", lw=1.0, ls=":",
               label=f"d={args.d_target} (95% sep)")
    ax.set_xlabel("t (rounds since round 0 drop)")
    ax.set_ylabel("Cohen's d  (true k1 vs true k2 cumulative log-LR)")
    ax.set_title(
        f"R3b sequence-level separability: Cohen's d(t) per ambiguity-group pair\n"
        f"p_bg={args.p_bg}, p_high={args.p_high}, N={args.n_samples}, seed={args.seed}\n"
        f"dashed lines: sqrt(t) fit"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    plt.tight_layout()
    plot_path = args.out_dir / f"d_scaling_T{args.T}_N{args.n_samples}{tag}.png"
    plt.savefig(plot_path, dpi=140)
    plt.close()
    print(f"[saved] {plot_path}")

    # also plot projection: extrapolate sqrt fit to T_required for each pair
    fig, ax = plt.subplots(figsize=(9, 6))
    for (k1, k2) in pairs:
        a = fits[(k1, k2)]["a"]
        col = pair_to_color[(k1, k2)]
        T_req = fits[(k1, k2)]["T_required_for_target"]
        t_extrap = np.linspace(1, max(T_req * 1.1, args.T * 2), 200)
        ax.plot(t_extrap, a * np.sqrt(t_extrap), color=col, lw=1.5,
                label=f"({k1},{k2}) → T={T_req:.0f}")
    ax.axhline(args.d_target, color="black", lw=1.0, ls=":")
    ax.set_xlabel("t (rounds)")
    ax.set_ylabel("projected Cohen's d (from sqrt fit)")
    ax.set_xscale("log")
    ax.set_title(f"sqrt(t) projection: T required for d={args.d_target} per ambiguity-group pair\n"
                 f"(L=1 marginal-only Bayes log-LR; longer-window decoder would be faster)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best")
    plt.tight_layout()
    proj_path = args.out_dir / f"d_projection_T{args.T}_N{args.n_samples}{tag}.png"
    plt.savefig(proj_path, dpi=140)
    plt.close()
    print(f"[saved] {proj_path}")


if __name__ == "__main__":
    main()
