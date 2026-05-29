"""
Analyze the structural ambiguity groups in the R1 dominant-CNOT identification
ceiling, and compare 15-Pauli vs 9-Pauli fault models.

Outputs (under data/analysis/6_sequence_ceiling/):
    ambiguity_groups.json        — group structure for both Pauli pools
    ceiling_compare_grid.csv     — sweep over (p_bg, p_high, T) for both models
    ceiling_compare_curves.png   — per-class accuracy vs T, 15p vs 9p
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.ceiling import (
    PAULI_PAIRS_15,
    bayes_classifier_confusion,
    compute_all_distributions,
    compute_single_round_lookup,
    overall_accuracy,
)
from src.config import DATA_DIR
from src.surface_code_layout import enumerate_stabilizer_cnots


OUT_DIR = DATA_DIR / "analysis" / "6_sequence_ceiling"


def ambiguity_groups(lookup_subset: np.ndarray) -> list[list[int]]:
    """Return groups of CNOT indices sharing identical syndrome multisets."""
    sigs = {}
    for k in range(lookup_subset.shape[0]):
        sig = tuple(sorted(lookup_subset[k].tolist()))
        sigs.setdefault(sig, []).append(k)
    return sorted([g for g in sigs.values() if len(g) > 1])


def label_cnots(cnots):
    return [
        f"CNOT{k:02d} {c['stab_type']}{c['stab_index']} "
        f"{c['control']}->{c['target']}"
        for k, c in enumerate(cnots)
    ]


def compute_ceiling_from_groups(groups, n_cnot: int = 24) -> float:
    """Asymptotic per-class accuracy = (number of identifiable groups) / 24."""
    in_group = set()
    for g in groups:
        in_group.update(g)
    n_singletons = n_cnot - len(in_group)
    n_groups = n_singletons + len(groups)
    return n_groups / n_cnot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lookup_path = OUT_DIR / "lookup_reset.npy"
    if lookup_path.exists():
        print(f"[groups] loading lookup from {lookup_path}")
        lookup_15 = np.load(lookup_path)
    else:
        print("[groups] computing single-round lookup (reset)...")
        lookup_15 = compute_single_round_lookup(reset=True)
        np.save(lookup_path, lookup_15)

    nine_idx = [i for i, p in enumerate(PAULI_PAIRS_15) if "I" not in p]
    lookup_9 = lookup_15[:, nine_idx]

    g15 = ambiguity_groups(lookup_15)
    g9 = ambiguity_groups(lookup_9)
    cnot_labels = label_cnots(enumerate_stabilizer_cnots())

    print()
    print(f"15-Pauli ambiguity groups: {g15}")
    print(f"  asymptotic per-class ceiling = {compute_ceiling_from_groups(g15):.4f}")
    print(f" 9-Pauli ambiguity groups: {g9}")
    print(f"  asymptotic per-class ceiling = {compute_ceiling_from_groups(g9):.4f}")

    info = {
        "n_cnot": 24,
        "pauli_pool_15": {
            "size": 15,
            "members": ["".join(p) for p in PAULI_PAIRS_15],
            "ambiguity_groups": [
                {
                    "members": g,
                    "labels": [cnot_labels[k] for k in g],
                    "multiset_sorted": sorted(lookup_15[g[0]].tolist()),
                }
                for g in g15
            ],
            "asymptotic_per_class_ceiling": compute_ceiling_from_groups(g15),
            "asymptotic_group_ceiling": 1.0,
        },
        "pauli_pool_9": {
            "size": 9,
            "members": ["".join(PAULI_PAIRS_15[i]) for i in nine_idx],
            "ambiguity_groups": [
                {
                    "members": g,
                    "labels": [cnot_labels[k] for k in g],
                    "multiset_sorted": sorted(lookup_9[g[0]].tolist()),
                }
                for g in g9
            ],
            "asymptotic_per_class_ceiling": compute_ceiling_from_groups(g9),
            "asymptotic_group_ceiling": 1.0,
        },
        "note": (
            "Per-class ceiling is the fraction of CNOTs that lie in singleton "
            "groups (uniquely identifiable). Within a group of size g, the "
            "Bayes-optimal predictor picks one representative, so per-class "
            "accuracy averages to (# singletons + # groups) / 24. Group "
            "accuracy reaches 1.0 since group identity is identifiable."
        ),
    }
    (OUT_DIR / "ambiguity_groups.json").write_text(json.dumps(info, indent=2))
    print(f"[groups] wrote ambiguity_groups.json")

    # Sweep for the comparison plot
    configs = [(1e-3, 1e-2), (1e-3, 3e-2), (1e-2, 1e-1), (1e-2, 3e-1)]
    Ts = [10, 30, 100, 300, 1000]
    rng = np.random.default_rng(42)

    rows = []
    print()
    print(
        f'{"p_bg":>6} {"p_high":>8} {"T":>5}  '
        f'{"acc15":>7}  {"acc9":>7}'
    )
    print("-" * 50)
    for p_bg, p_high in configs:
        d15 = compute_all_distributions(p_bg, p_high, lookup_15)
        d9 = compute_all_distributions(p_bg, p_high, lookup_9)
        for T in Ts:
            c15 = bayes_classifier_confusion(d15, T, n_test=300, rng=rng)
            c9 = bayes_classifier_confusion(d9, T, n_test=300, rng=rng)
            a15, a9 = overall_accuracy(c15), overall_accuracy(c9)
            rows.append(
                {
                    "p_bg": p_bg,
                    "p_high": p_high,
                    "T": T,
                    "acc_15p": a15,
                    "acc_9p": a9,
                }
            )
            print(f"{p_bg:>6.0e} {p_high:>8.0e} {T:>5d}  {a15:>7.3f}  {a9:>7.3f}")

    # Save CSV
    import csv as _csv

    with open(OUT_DIR / "ceiling_compare_grid.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[groups] wrote ceiling_compare_grid.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model_key, label in zip(
        axes, ("acc_15p", "acc_9p"), ("15-Pauli", "9-Pauli")
    ):
        for p_bg, p_high in configs:
            sub = [r for r in rows if r["p_bg"] == p_bg and r["p_high"] == p_high]
            sub.sort(key=lambda r: r["T"])
            xs = [r["T"] for r in sub]
            ys = [r[model_key] for r in sub]
            ax.plot(xs, ys, marker="o", label=f"p_bg={p_bg:.0e} p_h={p_high:.0e}")
        ceiling = (
            compute_ceiling_from_groups(g15)
            if model_key == "acc_15p"
            else compute_ceiling_from_groups(g9)
        )
        ax.axhline(ceiling, color="red", ls="--", alpha=0.6, label=f"ceiling={ceiling:.3f}")
        ax.axhline(1 / 24, color="gray", ls=":", alpha=0.5, label="random=1/24")
        ax.set_xscale("log")
        ax.set_xlabel("T (rounds)")
        ax.set_title(f"{label} fault model")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Bayes-optimal per-class accuracy")
    fig.suptitle("R1 ceiling: 15-Pauli vs 9-Pauli fault model")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ceiling_compare_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[groups] wrote ceiling_compare_curves.png")


if __name__ == "__main__":
    main()
