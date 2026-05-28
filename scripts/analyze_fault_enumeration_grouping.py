# scripts/analyze_fault_enumeration_grouping.py

"""
Group the 387-case fault-enumeration table by:

    (1) full d-round syndrome equality   — entire (n_rounds, 8) array equal
    (2) last-round syndrome equality     — only syndrome[-1] equal

For each grouping and each measurement mode (reset / noreset), the script
writes a CSV listing the groups (id, size, syndrome signature, members)
and a heatmap PNG showing all 387 errors stacked by group with
group-boundary separators.

Input:
    data/analysis/fault_enumeration/fault_enumeration_table.npz

Output (under data/analysis/fault_enumeration_grouping/):
    full_sequence_groups_{mode}.csv
    full_sequence_groups_{mode}.png
    last_syndrome_groups_{mode}.csv
    last_syndrome_groups_{mode}.png
    group_size_summary.png
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration" / "fault_enumeration_table.npz"
OUT_DIR = PROJECT_ROOT / "data" / "analysis" / "fault_enumeration_grouping"


def group_by_key(errors, keys):
    groups = defaultdict(list)
    for e, k in zip(errors, keys):
        groups[k].append(str(e))
    # sort: largest first, ties broken by lex of key tuple for determinism
    items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [{"key": k, "members": sorted(v)} for k, v in items]


def write_csv(path, groups, n_rounds, n_stab):
    bit_cols = [f"b{i}" for i in range(len(groups[0]["key"]))]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "size", *bit_cols, "members"])
        for gid, g in enumerate(groups):
            w.writerow([gid, len(g["members"]), *g["key"], ";".join(g["members"])])


def render_heatmap(out_path, errors, syndromes, group_of, n_groups, title,
                   n_rounds, n_stab, stab_labels):
    """
    errors:    (N,) labels in display order (already sorted by group)
    syndromes: (N, n_rounds, n_stab) int array in display order
    group_of:  (N,) group id of each row
    """
    N = len(errors)
    flat = syndromes.reshape(N, n_rounds * n_stab)
    # height proportional to N, capped
    height = max(8.0, min(36.0, N * 0.06))
    fig, ax = plt.subplots(figsize=(10, height))

    cmap = mcolors.ListedColormap(["#f7fbff", "#08306b"])
    ax.imshow(flat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    # round separators (vertical)
    for r in range(1, n_rounds):
        ax.axvline(r * n_stab - 0.5, color="orange", lw=1.2)

    # group separators (horizontal)
    boundaries = []
    for i in range(1, N):
        if group_of[i] != group_of[i - 1]:
            boundaries.append(i - 0.5)
    for b in boundaries:
        ax.axhline(b, color="red", lw=0.6, alpha=0.7)

    # x ticks: stab labels per round
    xticks = list(range(n_rounds * n_stab))
    xlabels = [f"r{r}|{s}" for r in range(n_rounds) for s in stab_labels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)

    # y ticks: every error label, fontsize tiny
    ax.set_yticks(range(N))
    ax.set_yticklabels(errors, fontsize=4)

    # Group annotation on right: print group id once per group at its midpoint
    gid_to_rows = defaultdict(list)
    for i, g in enumerate(group_of):
        gid_to_rows[g].append(i)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ann_ticks = []
    ann_labels = []
    for gid, rows in gid_to_rows.items():
        if len(rows) >= 2:
            mid = (rows[0] + rows[-1]) / 2
            ann_ticks.append(mid)
            ann_labels.append(f"G{gid}({len(rows)})")
    ax2.set_yticks(ann_ticks)
    ax2.set_yticklabels(ann_labels, fontsize=5)
    ax2.tick_params(axis="y", length=0)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("round | stabilizer (Z0..Z3, X0..X3)")
    ax.set_ylabel("error (sorted by group)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_group_size_summary(out_path, summary):
    """summary: dict[mode][label] = list of group sizes (sorted desc)."""
    modes = sorted(summary.keys())
    labels = ["full_sequence", "last_syndrome"]
    fig, axes = plt.subplots(len(modes), len(labels),
                             figsize=(11, 3.5 * len(modes)), squeeze=False)
    for i, m in enumerate(modes):
        for j, lab in enumerate(labels):
            sizes = summary[m][lab]
            ax = axes[i][j]
            ax.bar(range(len(sizes)), sizes, color="#1f77b4")
            ax.set_title(f"{m} / {lab}\n"
                         f"{len(sizes)} groups, "
                         f"max={max(sizes)}, singletons={sum(1 for s in sizes if s == 1)}")
            ax.set_xlabel("group rank (desc by size)")
            ax.set_ylabel("group size")
            ax.set_xlim(-0.5, len(sizes) - 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ_PATH, allow_pickle=True)
    errors_all = np.asarray(data["error"], dtype=str)
    modes_all = np.asarray(data["mode"], dtype=str)
    syn_all = data["syndrome"]
    n_rounds = int(data["n_rounds"])
    n_stab = int(data["n_stabilizers"])
    stab_labels = [str(x) for x in data["stab_labels"]]

    summary = {}

    for mode in ("reset", "noreset"):
        mask = modes_all == mode
        errors = errors_all[mask]
        syn = syn_all[mask]
        assert syn.shape == (387, n_rounds, n_stab)

        full_keys = [tuple(s.flatten().tolist()) for s in syn]
        last_keys = [tuple(s[-1].tolist()) for s in syn]

        groups_full = group_by_key(errors, full_keys)
        groups_last = group_by_key(errors, last_keys)

        write_csv(OUT_DIR / f"full_sequence_groups_{mode}.csv",
                  groups_full, n_rounds, n_stab)
        write_csv(OUT_DIR / f"last_syndrome_groups_{mode}.csv",
                  groups_last, n_rounds, n_stab)

        # Build heatmap inputs: ordered list of (error, syndrome, gid)
        for grouping_label, groups in [("full_sequence", groups_full),
                                       ("last_syndrome", groups_last)]:
            ord_err = []
            ord_syn = []
            ord_gid = []
            for gid, g in enumerate(groups):
                for m in g["members"]:
                    idx = int(np.where(errors == m)[0][0])
                    ord_err.append(m)
                    ord_syn.append(syn[idx])
                    ord_gid.append(gid)
            ord_syn = np.stack(ord_syn, axis=0)
            multi_groups = sum(1 for g in groups if len(g["members"]) >= 2)
            title = (f"{grouping_label}  /  mode={mode}\n"
                     f"{len(ord_err)} errors  →  {len(groups)} groups "
                     f"(multi-member: {multi_groups}, "
                     f"singletons: {len(groups) - multi_groups})")
            render_heatmap(OUT_DIR / f"{grouping_label}_groups_{mode}.png",
                           ord_err, ord_syn, ord_gid, len(groups),
                           title, n_rounds, n_stab, stab_labels)

        summary[mode] = {
            "full_sequence": sorted([len(g["members"]) for g in groups_full], reverse=True),
            "last_syndrome": sorted([len(g["members"]) for g in groups_last], reverse=True),
        }
        print(f"[{mode}] full-seq groups={len(groups_full)}  "
              f"last-syn groups={len(groups_last)}")

    render_group_size_summary(OUT_DIR / "group_size_summary.png", summary)
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
