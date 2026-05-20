"""
Analyze the saved R3b ceiling outputs and produce the headline plots that
answer the §13.7 hypothesis.

Reads from `data/analysis/7_r3b_ceiling/`:
  - r3b_grid.csv                         (per-cell R1 vs R3b accuracies)
  - r3b_marginals_pbg*_phigh*_T*.npz     (per-cell raw marginals + confusion)

Writes back into the same directory:
  - r3b_per_class_by_group.png           per-class accuracy bar chart split by
                                         R1 ambiguity group, R1 vs R3b side by side
  - r3b_within_group_spread.csv          for each R1 ambiguity group, the
                                         standard deviation of per-class
                                         accuracy under R1 vs R3b (a non-zero
                                         spread under R3b is direct evidence
                                         that R3b breaks the symmetry)
  - r3b_marginal_divergence.csv          KL divergence between R3b marginals
                                         of ambiguity-group members; expected
                                         to be exactly 0 under R1 and ≥ 0
                                         under R3b — magnitude indicates
                                         how strongly R3b breaks the group
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.ceiling_r3b import group_accuracy, per_class_accuracy
from src.config import DATA_DIR


R1_AMBIG_GROUPS = [
    [6, 7],
    [14, 15, 17],
    [18, 19, 20],
    [22, 23],
]


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) on a discrete distribution."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def analyze_cell(npz_path: Path) -> dict:
    data = np.load(npz_path)
    marginals = data["marginals"]              # (24, 256)
    conf_r3b = data["confusion_r3b"]            # (24, 24)
    conf_r1 = data["confusion_r1"]              # (24, 24)

    pc_r1 = per_class_accuracy(conf_r1)
    pc_r3b = per_class_accuracy(conf_r3b)
    g_overall_r1, _, groups_full = group_accuracy(conf_r1)
    g_overall_r3b, _, _ = group_accuracy(conf_r3b)

    # within-group spread of per-class accuracy
    within_group_std = {}
    for g in R1_AMBIG_GROUPS:
        tag = "_".join(str(k) for k in g)
        within_group_std[tag] = {
            "r1_std": float(np.std([pc_r1[k] for k in g])),
            "r3b_std": float(np.std([pc_r3b[k] for k in g])),
            "r1_mean": float(np.mean([pc_r1[k] for k in g])),
            "r3b_mean": float(np.mean([pc_r3b[k] for k in g])),
        }

    # marginal JS divergence between group members
    marginal_div = {}
    for g in R1_AMBIG_GROUPS:
        tag = "_".join(str(k) for k in g)
        pairs = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = js_divergence(marginals[g[i]], marginals[g[j]])
                pairs.append(d)
        marginal_div[tag] = {
            "mean_pair_js": float(np.mean(pairs)),
            "max_pair_js": float(np.max(pairs)),
            "n_pairs": len(pairs),
        }

    return {
        "filename": npz_path.name,
        "pc_r1": pc_r1.tolist(),
        "pc_r3b": pc_r3b.tolist(),
        "overall_r1": float(np.diag(conf_r1).sum() / conf_r1.sum()),
        "overall_r3b": float(np.diag(conf_r3b).sum() / conf_r3b.sum()),
        "group_r1": g_overall_r1,
        "group_r3b": g_overall_r3b,
        "within_group_std": within_group_std,
        "marginal_divergence": marginal_div,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path,
                    default=DATA_DIR / "analysis" / "7_r3b_ceiling")
    args = ap.parse_args()

    in_dir = args.in_dir
    npz_files = sorted(in_dir.glob("r3b_marginals_*.npz"))
    if not npz_files:
        raise SystemExit(f"no r3b_marginals_*.npz files in {in_dir}")

    print(f"[analyze-r3b] {len(npz_files)} cell file(s) found")
    print()

    results = []
    for p in npz_files:
        r = analyze_cell(p)
        results.append(r)
        print(f"=== {r['filename']} ===")
        print(f"  overall:  R1 {r['overall_r1']:.3f}   R3b {r['overall_r3b']:.3f}")
        print(f"  group:    R1 {r['group_r1']:.3f}   R3b {r['group_r3b']:.3f}")
        print(f"  R1 ambiguity groups: within-group per-class spread (std)")
        for tag, v in r["within_group_std"].items():
            print(f"    {tag}: R1 std={v['r1_std']:.3f} (mean {v['r1_mean']:.3f})  "
                  f"R3b std={v['r3b_std']:.3f} (mean {v['r3b_mean']:.3f})")
        print(f"  R3b marginal JS divergence between group members:")
        for tag, v in r["marginal_divergence"].items():
            print(f"    {tag}: mean pair JS = {v['mean_pair_js']:.4f}  max = {v['max_pair_js']:.4f}")
        print()

    # Write CSVs
    csv_spread = in_dir / "r3b_within_group_spread.csv"
    with open(csv_spread, "w") as f:
        f.write("cell,group,r1_mean,r1_std,r3b_mean,r3b_std\n")
        for r in results:
            for tag, v in r["within_group_std"].items():
                f.write(f"{r['filename']},{tag},{v['r1_mean']},{v['r1_std']},{v['r3b_mean']},{v['r3b_std']}\n")
    print(f"wrote {csv_spread}")

    csv_div = in_dir / "r3b_marginal_divergence.csv"
    with open(csv_div, "w") as f:
        f.write("cell,group,mean_pair_js,max_pair_js,n_pairs\n")
        for r in results:
            for tag, v in r["marginal_divergence"].items():
                f.write(f"{r['filename']},{tag},{v['mean_pair_js']},{v['max_pair_js']},{v['n_pairs']}\n")
    print(f"wrote {csv_div}")

    # Per-class bar plot — use the LARGEST T cell for clarity
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing, skipping plot")
        return

    # Pick cell with largest T (heuristic: max number in filename)
    def get_T(name):
        for chunk in name.replace(".npz", "").split("_"):
            if chunk.startswith("T"):
                try:
                    return int(chunk[1:])
                except ValueError:
                    pass
        return 0

    cell = max(results, key=lambda r: get_T(r["filename"]))
    pc_r1 = np.array(cell["pc_r1"])
    pc_r3b = np.array(cell["pc_r3b"])
    ks = np.arange(24)
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.4
    ax.bar(ks - width / 2, pc_r1, width=width, label="R1 (analytic)", color="#888")
    ax.bar(ks + width / 2, pc_r3b, width=width, label="R3b (MC)", color="#1f77b4")
    # mark ambiguity groups with shaded backgrounds
    colors = ["#fde", "#dfe", "#def", "#fed"]
    for gi, g in enumerate(R1_AMBIG_GROUPS):
        ax.axvspan(min(g) - 0.5, max(g) + 0.5, color=colors[gi % len(colors)], alpha=0.25,
                   zorder=0)
    ax.set_xticks(ks)
    ax.set_xlabel("CNOT index k")
    ax.set_ylabel("Per-class accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class accuracy: R1 vs R3b ({cell['filename']})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = in_dir / "r3b_per_class_by_group.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
