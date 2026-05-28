"""
Build per-shot ML training dataset for 24-class CNOT identification.

For each of 24 CNOTs k, sample N_per_class d-round shots from the
bag-of-shots fault model (p_high at k, p_bg at all other 32 locations).
Save as one (N_total, d, n_stab) array of int8 syndromes plus integer
labels, with stratified train/val/test splits.

Bag grouping is deferred to training time -- the shots here are the
atomic unit.

Outputs (under data/ml_dataset/<tag>/):
  shots.npy       int8 (N_total, d, n_stab)
  labels.npy      int8 (N_total,)  values 0..23
  cnot_keys.json  ordered list of CNOT names (label -> name mapping)
  split.npz       {train_idx, val_idx, test_idx} stratified by label
  config.json     full generation config + atom_source
  sanity.json     vacuum rates, mean event bits, split sizes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_mixture import (  # noqa: E402
    load_atoms,
    precompute_shifted_atoms,
    sample_shots,
)


OUT_ROOT = PROJECT_ROOT / "data" / "ml_dataset"


def stratified_split(labels, val_frac, test_frac, rng):
    train, val, test = [], [], []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])
    return (
        np.array(sorted(train), dtype=np.int64),
        np.array(sorted(val), dtype=np.int64),
        np.array(sorted(test), dtype=np.int64),
    )


def sanity_summary(shots, labels, cnot_keys, n_per_class, p_high, p_bg, d):
    out = {
        "shape_shots": list(shots.shape),
        "shape_labels": list(labels.shape),
        "dtype_shots": str(shots.dtype),
        "dtype_labels": str(labels.dtype),
        "n_total": int(len(shots)),
        "n_classes": int(len(np.unique(labels))),
        "per_class": {},
    }
    flat = shots.reshape(len(shots), -1)
    vacuum_mask = (flat == 0).all(axis=1)
    for lbl in range(24):
        mask = labels == lbl
        n = int(mask.sum())
        cls_vac = int(vacuum_mask[mask].sum())
        cls_bits = int(flat[mask].sum())
        out["per_class"][cnot_keys[lbl]] = {
            "n": n,
            "vacuum_frac": cls_vac / n,
            "mean_bits_per_shot": cls_bits / n,
        }
    out["overall_vacuum_frac"] = float(vacuum_mask.mean())
    # lower-bound expectation: no event anywhere
    n_other = 33 - 1
    expected_lb = (1 - p_high) ** d * (1 - p_bg) ** (n_other * d)
    out["expected_vacuum_lower_bound"] = float(expected_lb)
    out["notes"] = (
        "vacuum_frac >= expected_lower_bound; difference comes from "
        "syndrome cancellation when multiple events XOR to zero."
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["reset", "noreset"], default="noreset")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--p-bg", type=float, default=0.01)
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    atoms, d_atomic = load_atoms(args.mode)
    if args.d > d_atomic:
        raise SystemExit(
            f"--d={args.d} exceeds atomic table d_atomic={d_atomic}."
        )

    all_loc_keys = sorted(atoms.keys(), key=lambda x: (not x.startswith("cnot"), x))
    cnot_keys = [k for k in all_loc_keys if k.startswith("cnot")]
    shifted = precompute_shifted_atoms(atoms, args.d)
    rng = np.random.default_rng(args.seed)

    tag = args.tag or (
        f"{args.mode}_d{args.d}_phigh{args.p_high}_pbg{args.p_bg}_n{args.n_per_class}"
    )
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.n_per_class} shots/class x {len(cnot_keys)} CNOTs "
        f"(mode={args.mode}, d={args.d}, p_high={args.p_high}, p_bg={args.p_bg})"
    )
    parts_shots, parts_labels = [], []
    for lbl, dom in enumerate(cnot_keys):
        s = sample_shots(
            shifted, all_loc_keys, dom,
            args.p_high, args.p_bg, args.d, args.n_per_class, rng,
        )
        parts_shots.append(s)
        parts_labels.append(np.full(args.n_per_class, lbl, dtype=np.int8))
        print(f"  [{lbl + 1:2d}/24] {dom}")

    shots = np.concatenate(parts_shots, axis=0)
    labels = np.concatenate(parts_labels, axis=0)

    perm = rng.permutation(len(labels))
    shots = shots[perm]
    labels = labels[perm]

    train_idx, val_idx, test_idx = stratified_split(
        labels, args.val_frac, args.test_frac, rng,
    )

    np.save(out_dir / "shots.npy", shots)
    np.save(out_dir / "labels.npy", labels)
    np.savez(
        out_dir / "split.npz",
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    (out_dir / "cnot_keys.json").write_text(json.dumps(cnot_keys, indent=2))

    config = vars(args).copy()
    config["atom_source"] = str(
        Path("data") / "analysis" / "fault_enumeration" / "fault_enumeration_table.npz"
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    sanity = sanity_summary(
        shots, labels, cnot_keys, args.n_per_class,
        args.p_high, args.p_bg, args.d,
    )
    sanity["split"] = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=2))

    print()
    print(f"shots: {shots.shape} {shots.dtype}")
    print(f"labels: {labels.shape} {labels.dtype}")
    print(
        f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    print(f"overall vacuum frac: {sanity['overall_vacuum_frac']:.4f}")
    print(f"  lower bound (no events anywhere): {sanity['expected_vacuum_lower_bound']:.4f}")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
