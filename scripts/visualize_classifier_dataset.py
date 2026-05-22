"""
Inspect the classifier dataset built by `build_classifier_dataset.py` and
emit a handful of figures so a human can understand what one sample
actually looks like before training begins.

Outputs (under `data/analysis/9_classifier/dataset_preview/`):

  1. single_samples_T100.png
        For 5 representative classes (one from each R1 ambiguity group +
        one non-ambiguous baseline), show ONE example sequence per class,
        R1 vs R3b side by side. T=100 prefix only for readability.
        Each panel is (T=100, 8) — rows are the 8 stabilizers
        [Z0..Z3, X0..X3], columns are rounds.

  2. mean_event_rate_T200.png
        Same 5 classes. For each (scenario, k), average detection-bit value
        over all 2000 training samples per round per stabilizer; T=200
        prefix. Reveals where each class's signal concentrates without the
        per-sample noise. Mostly 0–0.05 in absolute value.

  3. per_stab_rate_all24.png
        For ALL 24 classes, the per-stabilizer mean event rate (averaged
        over all rounds + all samples). Compact 8×24 heatmap per scenario.
        Lets you see at a glance the "fingerprint" of each class.

  4. fault_counts_summary.csv
        Per-class mean and std of total non-zero detection events across
        the 1000 rounds. Quick sanity check that the elevated CNOT really
        produces more events.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import DATA_DIR


# 5 representative classes — one from each R1 ambiguity group + one baseline
REPRESENTATIVE_CLASSES = [
    (1,  "non-ambig", "G0 (CNOT 1, no group)"),
    (6,  "G1",        "G1 ambig (6, 7) — Z2 stab"),
    (14, "G2",        "G2 ambig (14, 15, 17) — X1 stab interior"),
    (18, "G3",        "G3 ambig (18, 19, 20) — X2 stab interior"),
    (22, "G4",        "G4 ambig (22, 23) — X3 stab"),
]

STAB_LABELS = ["Z0", "Z1", "Z2", "Z3", "X0", "X1", "X2", "X3"]


def load_train_split(scenario: str, data_dir: Path):
    data = np.load(data_dir / f"{scenario}_train.npz", allow_pickle=False)
    return data["syndromes"], data["labels"]


def first_per_class(syndromes, labels, k, count=1):
    """Return the first `count` samples whose label == k."""
    idx = np.where(labels == k)[0][:count]
    return syndromes[idx]


def all_per_class(syndromes, labels, k):
    idx = np.where(labels == k)[0]
    return syndromes[idx]


def plot_single_samples(scenarios_data, out_path, T_show=100):
    """
    Figure 1: 5 classes × 2 scenarios = 10 panels in a (5, 2) grid.
    Each panel is a (8, T_show) heatmap of one example sample.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(REPRESENTATIVE_CLASSES), 2,
        figsize=(11, 1.6 * len(REPRESENTATIVE_CLASSES)),
        sharey=True,
    )

    for row, (k, tag, title) in enumerate(REPRESENTATIVE_CLASSES):
        for col, scenario in enumerate(["r1", "r3b"]):
            ax = axes[row, col]
            syn, lab = scenarios_data[scenario]
            sample = first_per_class(syn, lab, k, count=1)[0]  # (T_max, 8)
            # transpose to (8, T_show) for natural reading
            disp = sample[:T_show, :].T
            im = ax.imshow(disp, aspect="auto", cmap="Greys",
                           vmin=0, vmax=1, interpolation="nearest")
            ax.set_yticks(np.arange(8))
            ax.set_yticklabels(STAB_LABELS, fontsize=7)
            if row == len(REPRESENTATIVE_CLASSES) - 1:
                ax.set_xlabel("round t")
            if col == 0:
                ax.set_ylabel(f"k={k}\n{tag}", fontsize=8)
            # detection event count for header
            n_events = int(disp.sum())
            ax.set_title(
                f"{scenario.upper()}  |  {title}  |  T={T_show} shown ({n_events} events total)",
                fontsize=8,
            )

    fig.suptitle(
        "One sample per (scenario, class) — black dot = detection event in that round on that stabilizer\n"
        "(stabilizers = [Z0..Z3, X0..X3]; rows=stab, cols=round)",
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_mean_event_rate(scenarios_data, out_path, T_show=200):
    """
    Figure 2: class-averaged detection event rate per (round, stabilizer).
    Reveals where each class concentrates its signal.
    """
    import matplotlib.pyplot as plt

    n_rows = len(REPRESENTATIVE_CLASSES)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, 2.2 * n_rows),
        sharey=True,
        gridspec_kw={"hspace": 0.55, "wspace": 0.05},
    )

    # global vmax for fair color scale across panels
    vmaxs = []
    panels = []
    for k, tag, title in REPRESENTATIVE_CLASSES:
        for scenario in ["r1", "r3b"]:
            syn, lab = scenarios_data[scenario]
            class_samples = all_per_class(syn, lab, k)
            rate = class_samples[:, :T_show, :].mean(axis=0).T  # (8, T_show)
            panels.append((k, tag, title, scenario, rate))
            vmaxs.append(rate.max())
    vmax = max(vmaxs)

    for idx, (k, tag, title, scenario, rate) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        im = ax.imshow(rate, aspect="auto", cmap="magma",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(np.arange(8))
        ax.set_yticklabels(STAB_LABELS, fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("round t", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"k={k}\n{tag}", fontsize=9)
        ax.set_title(
            f"{scenario.upper()}  |  {title}  |  mean rate over 2000 samples",
            fontsize=9,
        )

    # Right-side colorbar (outside the axes, not squeezing them)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.10, 0.012, 0.78])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"detection event rate (0 to {vmax:.3f})", fontsize=9)

    fig.suptitle(
        "Class-averaged detection event rate — where each class's signal concentrates\n"
        "(rows = stabilizers; columns = rounds; vmax across all panels for fair compare)",
        y=0.995, fontsize=11,
    )
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_per_stab_rate_all24(scenarios_data, out_path):
    """
    Figure 3: per-class per-stabilizer mean rate, all 24 classes.
    8 stab × 24 class heatmap per scenario.
    """
    import matplotlib.pyplot as plt

    rates = {}
    for scenario in ["r1", "r3b"]:
        syn, lab = scenarios_data[scenario]
        # shape (24, 8)
        mat = np.zeros((24, 8), dtype=np.float64)
        for k in range(24):
            class_samples = all_per_class(syn, lab, k)
            mat[k] = class_samples.mean(axis=(0, 1))   # avg over samples & rounds
        rates[scenario] = mat

    vmax = max(rates["r1"].max(), rates["r3b"].max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, scenario in zip(axes, ["r1", "r3b"]):
        im = ax.imshow(rates[scenario], aspect="auto", cmap="magma",
                       vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(STAB_LABELS)
        ax.set_yticks(np.arange(24))
        ax.set_xlabel("stabilizer")
        ax.set_ylabel("dominant CNOT k")
        ax.set_title(f"{scenario.upper()}: P(stab_i fires | dominant k)\n"
                     f"avg over all rounds & samples")
        # highlight R1 ambiguity-group rows
        for g in [[6, 7], [14, 15, 17], [18, 19, 20], [22, 23]]:
            ax.axhspan(min(g) - 0.5, max(g) + 0.5, color="cyan",
                       alpha=0.07, zorder=0)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                        label=f"mean detection rate (0 to {vmax:.3f})")
    fig.suptitle(
        "All 24 classes: per-stabilizer mean event rate — class fingerprint summary",
        y=1.01,
    )
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")
    return rates


def emit_summary_csv(scenarios_data, out_path):
    """For each (scenario, k), mean+std of total detection events per sample."""
    lines = ["scenario,k,n_samples,T,mean_total_events,std_total_events,mean_per_round"]
    for scenario in ["r1", "r3b"]:
        syn, lab = scenarios_data[scenario]
        for k in range(24):
            cs = all_per_class(syn, lab, k)
            totals = cs.sum(axis=(1, 2))
            T = cs.shape[1]
            lines.append(
                f"{scenario},{k},{cs.shape[0]},{T},"
                f"{totals.mean():.3f},{totals.std():.3f},"
                f"{totals.mean() / T:.5f}"
            )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=DATA_DIR / "classifier_dataset")
    ap.add_argument("--out-dir", type=Path,
                    default=DATA_DIR / "analysis" / "9_classifier" / "dataset_preview")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.data_dir}")
    scenarios_data = {}
    for scenario in ["r1", "r3b"]:
        syn, lab = load_train_split(scenario, args.data_dir)
        scenarios_data[scenario] = (syn, lab)
        print(f"  {scenario}: syndromes {syn.shape} dtype={syn.dtype}  labels {lab.shape}")

    plot_single_samples(scenarios_data, args.out_dir / "single_samples_T100.png", T_show=100)
    plot_mean_event_rate(scenarios_data, args.out_dir / "mean_event_rate_T200.png", T_show=200)
    plot_per_stab_rate_all24(scenarios_data, args.out_dir / "per_stab_rate_all24.png")
    emit_summary_csv(scenarios_data, args.out_dir / "fault_counts_summary.csv")

    print("\n[done] preview figures in:")
    print(f"  {args.out_dir}")


if __name__ == "__main__":
    main()
