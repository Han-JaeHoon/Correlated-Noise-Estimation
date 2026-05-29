"""
Generate train / val / test syndrome-sequence datasets for the Task #6
classifier (RNN / GRU / Transformer dominant-CNOT identification).

For each of two scenarios — R1 (IdentityDecoder) and R3b (LookupDecoder) —
sample N_train + N_val + N_test independent sequences of length T_MAX for
every dominant CNOT k in 0..23. Shorter T values used later are taken as
prefixes (so the splits are nested: the same sequence appears as both its
T=100 prefix and its T=300 prefix during evaluation).

Output (under `data/classifier_dataset/`):
    {scenario}_train.npz, {scenario}_val.npz, {scenario}_test.npz
each containing
    syndromes : (N, T_MAX, 8) uint8
    labels    : (N,)         int64    (= true dominant CNOT id 0..23)
    meta      : structured info

Deterministic from --seed.
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
from src.decoder import IdentityDecoder, LookupDecoder, PhenomDecoder


SCENARIOS = {
    "r1": "identity",      # R1: no decoder, sequence_runner Identity fast path
    "r2": "phenom",        # R2: phenomenological — 9×3 single-data-qubit-Pauli lookup
    "r3b": "lookup",       # R3b: window=1 circuit-level LookupDecoder (24×15 fault)
}


def generate_scenario(
    scenario: str,
    p_bg: float,
    p_high: float,
    T_max: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    out_dir: Path,
    verbose: bool = True,
):
    decoder_kind = SCENARIOS[scenario]

    # Re-use the same decoder instance across all 24 classes (LookupDecoder
    # construction is non-trivial)
    if decoder_kind == "identity":
        decoder = IdentityDecoder()
    elif decoder_kind == "lookup":
        decoder = LookupDecoder(reset=True)
    elif decoder_kind == "phenom":
        decoder = PhenomDecoder(reset=True)
    else:
        raise ValueError(decoder_kind)

    n_total = n_train + n_val + n_test
    print(f"[{scenario}] generating {n_total} seq/class × 24 classes × T={T_max}")
    print(f"  (n_train={n_train}, n_val={n_val}, n_test={n_test})")
    print(f"  p_bg={p_bg}, p_high={p_high}, decoder={decoder_kind}, seed={seed}")

    rng_master = np.random.default_rng(seed)

    syndromes_all = np.zeros((N_CNOT, n_total, T_max, 8), dtype=np.uint8)

    t0 = time.time()
    for k in range(N_CNOT):
        sub_seed = int(rng_master.integers(0, 2**63 - 1))
        rng_k = np.random.default_rng(sub_seed)
        syndromes_all[k] = simulate_class_sequences(
            k_dom=k,
            n_samples=n_total,
            T=T_max,
            p_bg=p_bg,
            p_high=p_high,
            decoder_kind=decoder_kind,
            rng=rng_k,
            decoder=decoder,
        )
        if verbose:
            print(f"  k={k:2d} done ({time.time()-t0:.1f}s)")

    print(f"[{scenario}] generation finished in {time.time()-t0:.1f}s")

    # Split: first n_train of each class -> train, then n_val -> val, then n_test -> test
    train_syn = syndromes_all[:, :n_train].reshape(N_CNOT * n_train, T_max, 8)
    train_lab = np.repeat(np.arange(N_CNOT, dtype=np.int64), n_train)

    val_syn = syndromes_all[:, n_train : n_train + n_val].reshape(N_CNOT * n_val, T_max, 8)
    val_lab = np.repeat(np.arange(N_CNOT, dtype=np.int64), n_val)

    test_syn = syndromes_all[:, n_train + n_val :].reshape(N_CNOT * n_test, T_max, 8)
    test_lab = np.repeat(np.arange(N_CNOT, dtype=np.int64), n_test)

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(
        scenario=scenario, decoder=decoder_kind, p_bg=p_bg, p_high=p_high,
        T_max=T_max, n_train=n_train, n_val=n_val, n_test=n_test,
        seed=seed,
    )

    for split, syn, lab in [
        ("train", train_syn, train_lab),
        ("val", val_syn, val_lab),
        ("test", test_syn, test_lab),
    ]:
        path = out_dir / f"{scenario}_{split}.npz"
        np.savez_compressed(path, syndromes=syn, labels=lab,
                            meta=json.dumps(meta))
        size_mb = path.stat().st_size / 1e6
        print(f"  saved {path}  shape={syn.shape}  size={size_mb:.1f}MB")

    # also save plain JSON meta sidecar for easy reading
    with open(out_dir / f"{scenario}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=["r1", "r3b"],
                    choices=list(SCENARIOS.keys()))
    ap.add_argument("--p-bg", type=float, default=0.01)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--T-max", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=DATA_DIR / "classifier_dataset")
    args = ap.parse_args()

    for scenario in args.scenarios:
        generate_scenario(
            scenario=scenario,
            p_bg=args.p_bg, p_high=args.p_high,
            T_max=args.T_max,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            seed=args.seed,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
