"""
Spatial mixture ambiguity analysis.

Bag-of-shots fault model:
  - Dominant location k is one of the 24 CNOTs (optionally also 9 data
    idle slots). For each round in a d-round shot, an event at k fires
    with probability p_high; if it fires, a Pauli is drawn uniformly
    from k's atomic Pauli pool (15 for CNOT, 3 for data idle).
  - At every other location ell, in every round, an event fires with
    probability p_bg with a uniform Pauli draw.
  - The d-round shot syndrome is the XOR of all event contributions,
    each contribution being the round-shifted atomic syndrome loaded
    from data/analysis/fault_enumeration/fault_enumeration_table.npz.

Two computation backends:

  exact (default when p_bg = 0):
      Enumerates every (round-subset, pauli-tuple) combination at the
      dominant location. For d=3, n_paulis_C=15 this is 4,096 atomic
      XOR-sums per location -- the resulting Counter is the exact
      mixture distribution. TV distances are exact (no MC noise).

  mc (default when p_bg > 0):
      Monte Carlo sampler that fires events at all (location, round)
      slots independently per the rate spec. A bootstrap noise floor
      is estimated by splitting one location's samples in half.

Outputs (under data/analysis/spatial_mixture/<tag>/):
  - tv_matrix.npy            pairwise TV between candidate locations
  - location_keys.json       row/column order
  - ambiguity_groups.json    union-find grouping under threshold
  - tv_heatmap.png           visualization
  - summary.txt              human-readable report
  - config.json              reproducibility record
"""

import argparse
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATOM_NPZ = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration" / "fault_enumeration_table.npz"
OUT_ROOT = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture"


# ---------------------------------------------------------------------------
# atom loading + time-translation
# ---------------------------------------------------------------------------

def load_atoms(mode):
    z = np.load(ATOM_NPZ, allow_pickle=True)
    sel = z["mode"] == mode
    err = z["error"][sel]
    syn = z["syndrome"][sel].astype(np.int8)
    d_atomic = syn.shape[1]

    by_loc = {}
    for e, s in zip(err, syn):
        _, loc, pauli = e.split(":")
        by_loc.setdefault(loc, []).append((pauli, s))
    atoms = {loc: (np.stack([s for _, s in items]),
                    [p for p, _ in items])
             for loc, items in by_loc.items()}
    return atoms, d_atomic


def precompute_shifted_atoms(atoms, d):
    """shifted[loc] : (d, n_paulis, d, 8) int8 -- fault-round x pauli x rounds x stab"""
    shifted = {}
    for loc, (arr, _) in atoms.items():
        n_paulis, d_atomic, n_stab = arr.shape
        out = np.zeros((d, n_paulis, d, n_stab), dtype=np.int8)
        for r in range(d):
            end = min(d_atomic, d - r)
            if end > 0:
                out[r, :, r:r + end, :] = arr[:, :end, :]
        shifted[loc] = out
    return shifted


def pack_syndrome(syn):
    return np.packbits(syn.flatten().astype(np.uint8), bitorder="little").tobytes()


# ---------------------------------------------------------------------------
# exact enumeration (p_bg == 0)
# ---------------------------------------------------------------------------

def exact_distribution(shifted_loc, p_high, d):
    """Exact P_k for single-location mixture. Returns {syndrome_bytes: prob}."""
    n_paulis = shifted_loc.shape[1]
    n_stab = shifted_loc.shape[3]
    result = Counter()
    for k in range(d + 1):
        outer = (p_high ** k) * ((1.0 - p_high) ** (d - k))
        if outer == 0.0 and k > 0:
            continue
        per_combo = outer / (n_paulis ** k) if k > 0 else outer
        for round_subset in combinations(range(d), k):
            for pauli_combo in product(range(n_paulis), repeat=k):
                syn = np.zeros((d, n_stab), dtype=np.int8)
                for r, pi in zip(round_subset, pauli_combo):
                    syn ^= shifted_loc[r, pi]
                result[pack_syndrome(syn)] += per_combo
    return result


def tv_exact(p, q):
    """TV between two {key: prob} dicts."""
    s = 0.0
    for k, v in p.items():
        s += abs(v - q.get(k, 0.0))
    for k, v in q.items():
        if k not in p:
            s += v
    return 0.5 * s


# ---------------------------------------------------------------------------
# MC backend (p_bg > 0 or sanity check)
# ---------------------------------------------------------------------------

def sample_shots(shifted, all_loc_keys, dominant, p_high, p_bg, d, n_shots, rng):
    n_stab = next(iter(shifted.values())).shape[-1]
    shots = np.zeros((n_shots, d, n_stab), dtype=np.int8)
    for loc in all_loc_keys:
        sh_arr = shifted[loc]
        n_paulis = sh_arr.shape[1]
        rate = p_high if loc == dominant else p_bg
        if rate <= 0.0:
            continue
        event_mask = rng.random((n_shots, d)) < rate
        n_events = int(event_mask.sum())
        if n_events == 0:
            continue
        shot_idx, round_idx = np.where(event_mask)
        pauli_idx = rng.integers(0, n_paulis, size=n_events)
        atoms_to_add = sh_arr[round_idx, pauli_idx]
        np.bitwise_xor.at(shots, shot_idx, atoms_to_add)
    return shots


def empirical_counter(shots):
    flat = shots.reshape(shots.shape[0], -1).astype(np.uint8)
    packed = np.packbits(flat, axis=1, bitorder="little")
    keys = [row.tobytes() for row in packed]
    c = Counter(keys)
    return {k: v / len(keys) for k, v in c.items()}


def bootstrap_noise_floor(shifted, all_loc_keys, dominant, p_high, p_bg, d, n_shots, rng,
                           n_splits=4):
    """Split-half TV estimates for a single location -> noise floor."""
    floors = []
    for _ in range(n_splits):
        shots = sample_shots(shifted, all_loc_keys, dominant, p_high, p_bg, d,
                              n_shots, rng)
        h = n_shots // 2
        p = empirical_counter(shots[:h])
        q = empirical_counter(shots[h:])
        floors.append(tv_exact(p, q))
    return max(floors), float(np.mean(floors))


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def union_find_groups(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["reset", "noreset"], default="reset")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--p-high", type=float, default=0.1)
    ap.add_argument("--p-bg", type=float, default=0.0)
    ap.add_argument("--backend", choices=["auto", "exact", "mc"], default="auto")
    ap.add_argument("--n-shots", type=int, default=200_000, help="MC backend only")
    ap.add_argument("--include-data-as-dominant", action="store_true")
    ap.add_argument("--threshold", type=float, default=None,
                    help="TV threshold for grouping; default 1e-10 (exact) / bootstrap (mc)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    atoms, d_atomic = load_atoms(args.mode)
    if args.d > d_atomic:
        raise SystemExit(
            f"--d={args.d} exceeds atomic table d_atomic={d_atomic}."
        )

    backend = args.backend
    if backend == "auto":
        backend = "exact" if args.p_bg == 0.0 else "mc"
    if backend == "exact" and args.p_bg != 0.0:
        raise SystemExit("exact backend requires p_bg == 0")

    all_loc_keys = sorted(atoms.keys(), key=lambda x: (not x.startswith("cnot"), x))
    dominant_keys = [k for k in all_loc_keys if k.startswith("cnot")]
    if args.include_data_as_dominant:
        dominant_keys = all_loc_keys

    print(f"Atoms loaded: {len(all_loc_keys)} locations "
          f"(d_atomic={d_atomic}, mode={args.mode}). "
          f"Dominant pool: {len(dominant_keys)}. Backend: {backend}.")

    shifted = precompute_shifted_atoms(atoms, args.d)
    rng = np.random.default_rng(args.seed)

    tag = args.tag or (
        f"{backend}_{args.mode}_d{args.d}_phigh{args.p_high}_pbg{args.p_bg}"
        + (f"_n{args.n_shots}" if backend == "mc" else "")
    )
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    distributions = []
    print()
    if backend == "exact":
        for i, dom in enumerate(dominant_keys):
            dist = exact_distribution(shifted[dom], args.p_high, args.d)
            distributions.append(dist)
            print(f"  [{i + 1:2d}/{len(dominant_keys)}] {dom:>8s}: "
                  f"{len(dist):>6d} distinct syndromes "
                  f"(sum={sum(dist.values()):.6f})")
    else:
        print(f"Sampling {args.n_shots:,} shots per dominant location...")
        for i, dom in enumerate(dominant_keys):
            shots = sample_shots(shifted, all_loc_keys, dom, args.p_high, args.p_bg,
                                  args.d, args.n_shots, rng)
            dist = empirical_counter(shots)
            distributions.append(dist)
            print(f"  [{i + 1:2d}/{len(dominant_keys)}] {dom:>8s}: "
                  f"{len(dist):>6d} distinct syndromes")

    # threshold
    if args.threshold is not None:
        threshold = args.threshold
        floor_info = f"manual={threshold}"
    elif backend == "exact":
        threshold = 1e-10
        floor_info = "exact (numerical zero)"
    else:
        max_floor, mean_floor = bootstrap_noise_floor(
            shifted, all_loc_keys, dominant_keys[0], args.p_high, args.p_bg,
            args.d, args.n_shots, rng)
        threshold = 1.5 * max_floor
        floor_info = f"bootstrap (max={max_floor:.4f}, mean={mean_floor:.4f}) -> 1.5*max"

    n_dom = len(dominant_keys)
    tv = np.zeros((n_dom, n_dom))
    print("\nComputing pairwise TV...")
    for i in range(n_dom):
        for j in range(i + 1, n_dom):
            d_ij = tv_exact(distributions[i], distributions[j])
            tv[i, j] = tv[j, i] = d_ij

    edges = [(i, j) for i in range(n_dom) for j in range(i + 1, n_dom)
             if tv[i, j] < threshold]
    grouping = union_find_groups(n_dom, edges)
    grouping_keys = sorted([[dominant_keys[i] for i in g] for g in grouping],
                            key=lambda g: (-len(g), g[0]))

    np.save(out_dir / "tv_matrix.npy", tv)
    (out_dir / "location_keys.json").write_text(json.dumps(dominant_keys, indent=2))
    (out_dir / "ambiguity_groups.json").write_text(json.dumps({
        "backend": backend,
        "tv_threshold": threshold,
        "threshold_source": floor_info,
        "n_groups": len(grouping_keys),
        "groups": grouping_keys,
    }, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(tv, cmap="viridis", vmin=0.0)
    ax.set_xticks(range(n_dom))
    ax.set_xticklabels(dominant_keys, rotation=90, fontsize=7)
    ax.set_yticks(range(n_dom))
    ax.set_yticklabels(dominant_keys, fontsize=7)
    plt.colorbar(im, label="Total-variation distance")
    ax.set_title(
        f"Pairwise TV (backend={backend}, threshold={threshold:.2e})\n"
        f"mode={args.mode}  d={args.d}  p_high={args.p_high}  p_bg={args.p_bg}\n"
        f"n_groups={len(grouping_keys)}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "tv_heatmap.png", dpi=130)
    plt.close()

    lines = [
        "Spatial mixture ambiguity",
        f"  backend={backend}  mode={args.mode}  d={args.d}",
        f"  p_high={args.p_high}  p_bg={args.p_bg}",
        f"  dominant locations: {n_dom}",
        f"  TV threshold: {threshold:.2e}  ({floor_info})",
        f"  ambiguity groups: {len(grouping_keys)}",
        "",
        "Groups (size, members):",
    ]
    for g in grouping_keys:
        marker = "  AMBIG " if len(g) > 1 else "  ok    "
        lines.append(f"{marker}[{len(g):2d}] " + ", ".join(g))
    txt = "\n".join(lines)
    print("\n" + txt)
    (out_dir / "summary.txt").write_text(txt + "\n")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
