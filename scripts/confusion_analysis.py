"""
Confusion matrix analysis for bag-of-shots classifiers.

Re-trains logreg / deepsets for selected N values, captures full (24×24)
confusion matrix, and generates:
  1. confusion_matrix_<model>_N<N>.png  — full 24×24 heatmap
  2. ambiguity_zoom.png                 — 4 ambiguity-group sub-matrices
  3. top_confused_pairs.csv             — ranked off-diagonal confusion pairs

Usage:
    python scripts/confusion_analysis.py
    python scripts/confusion_analysis.py --models logreg --N-list 100 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TAG  = "noreset_d3_phigh0.1_pbg0.01_n5000"
OUT_DIR   = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "confusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AMBIGUITY_GROUPS = {
    "G1: {6,7}":      [6, 7],
    "G2: {14,15,17}": [14, 15, 17],
    "G3: {18,19,20}": [18, 19, 20],
    "G4: {22,23}":    [22, 23],
}
AMBIG_MEMBERS = {c for g in AMBIGUITY_GROUPS.values() for c in g}

MODEL_LABELS = {
    "logreg":   "LogReg (linear)",
    "mlp":      "MLP",
    "deepsets": "Deep Sets",
}


# ── training + confusion capture ────────────────────────────────────────────

def train_and_confuse(model_name, shots, labels, split, N, epochs=80, seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.bag_classifier import BagDataset, build_model, build_vocab

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    train_shots, train_labels = shots[train_idx], labels[train_idx]
    val_shots,   val_labels   = shots[val_idx],   labels[val_idx]
    test_shots,  test_labels  = shots[test_idx],  labels[test_idx]

    vocab   = build_vocab(train_shots.reshape(len(train_shots), -1))
    V       = len(vocab)
    shot_dim = shots.shape[1] * shots.shape[2]
    rep      = "raw" if model_name == "deepsets" else "hist"

    train_ds = BagDataset(train_shots, train_labels, vocab, N=N, rep=rep,
                          reps_per_epoch=200, seed=seed)
    val_ds   = BagDataset(val_shots,   val_labels,   vocab, N=N, rep=rep,
                          reps_per_epoch=100, seed=seed+1)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    net       = build_model(model_name, vocab_size=V, shot_dim=shot_dim).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, patience, no_improve = 0.0, None, 12, 0

    for epoch in range(1, epochs + 1):
        net.train()
        for x, y in train_loader:
            x   = x.to(device)
            y   = y.to(device) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long, device=device)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()
        scheduler.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x     = x.to(device)
                preds = net(x).argmax(dim=1).cpu().numpy()
                correct += (preds == np.asarray(y)).sum()
                total   += len(y)
        val_acc = correct / total

        if val_acc > best_val:
            best_val  = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    print(f"  best_val={best_val:.4f}")

    # ── collect confusion matrix on test bags ──────────────────────────────
    net.load_state_dict(best_state)
    net.eval()

    # generate n_bags_per_class bags per class → stable confusion estimate
    n_bags_per_class = 50
    confmat = np.zeros((24, 24), dtype=np.int32)

    rng = np.random.default_rng(seed + 99)
    from src.bag_classifier import shots_to_indices

    if rep == "hist":
        test_indices = shots_to_indices(test_shots, vocab)

    per_class_pool = {
        c: np.where(test_labels == c)[0]
        for c in range(24)
    }

    with torch.no_grad():
        for true_k in range(24):
            pool = per_class_pool[true_k]
            for _ in range(n_bags_per_class):
                chosen = rng.choice(len(pool), size=min(N, len(pool)), replace=True)
                chosen_global = pool[chosen]

                if rep == "hist":
                    bag_idx = test_indices[chosen_global]
                    counts  = np.bincount(bag_idx, minlength=V + 1)[:V]
                    x = torch.tensor(counts / max(counts.sum(), 1),
                                     dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    raw = test_shots[chosen_global].reshape(len(chosen), -1).astype(np.float32)
                    x   = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(device)

                pred = int(net(x).argmax(dim=1).cpu())
                confmat[true_k, pred] += 1

    overall_acc = confmat.diagonal().sum() / confmat.sum()
    print(f"  confusion acc={overall_acc:.4f}  (N={N}, {n_bags_per_class} bags/class)")
    return confmat, overall_acc, best_val


# ── visualisation helpers ────────────────────────────────────────────────────

def plot_confusion_matrix(confmat, model_name, N, overall_acc):
    """Full 24×24 normalised confusion matrix with ambiguity-group annotations."""
    norm = confmat.astype(float) / confmat.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Fraction predicted as column class")

    # annotate cells with value ≥ 0.05
    for i in range(24):
        for j in range(24):
            v = norm[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black" if v < 0.6 else "white")

    # highlight ambiguity group boxes
    colors = ["#e15759", "#f28e2b", "#76b7b2", "#59a14f"]
    patches = []
    for (name, members), col in zip(AMBIGUITY_GROUPS.items(), colors):
        lo, hi = min(members) - 0.5, max(members) + 0.5
        span   = hi - lo
        for lw, xy in [((lo, lo), True)]:
            rect = mpatches.FancyBboxPatch(
                (min(members) - 0.5, min(members) - 0.5),
                span, span,
                boxstyle="square,pad=0",
                linewidth=2, edgecolor=col, facecolor="none",
            )
            ax.add_patch(rect)
        patches.append(mpatches.Patch(color=col, label=name))

    ax.legend(handles=patches, loc="upper right", fontsize=8,
              title="R1 ambiguity groups", title_fontsize=8)

    ax.set_xticks(range(24))
    ax.set_yticks(range(24))
    ax.set_xticklabels([str(i) for i in range(24)], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(24)], fontsize=7)
    ax.set_xlabel("Predicted CNOT", fontsize=11)
    ax.set_ylabel("True CNOT", fontsize=11)
    ax.set_title(
        f"Confusion matrix — {MODEL_LABELS.get(model_name, model_name)}, N={N}\n"
        f"overall acc = {overall_acc:.3f}",
        fontsize=11,
    )

    out = OUT_DIR / f"confusion_matrix_{model_name}_N{N}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_ambiguity_zoom(results):
    """
    results: list of (confmat, model_name, N, overall_acc)
    One row per ambiguity group, one column per (model, N) result.
    """
    groups = list(AMBIGUITY_GROUPS.items())
    n_cols = len(results)
    n_rows = len(groups)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows))
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    if n_rows == 1:
        axes = [axes]

    for row, (gname, members) in enumerate(groups):
        for col, (confmat, model_name, N, overall_acc) in enumerate(results):
            ax  = axes[row][col]
            sub = confmat[np.ix_(members, members)].astype(float)
            sub = sub / sub.sum(axis=1, keepdims=True).clip(1)

            im  = ax.imshow(sub, vmin=0, vmax=1, cmap="RdYlGn", aspect="equal")
            for i, r in enumerate(members):
                for j, c in enumerate(members):
                    ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black" if sub[i, j] < 0.7 else "white")

            ax.set_xticks(range(len(members)))
            ax.set_yticks(range(len(members)))
            ax.set_xticklabels([f"CNOT {m}" for m in members], fontsize=8)
            ax.set_yticklabels([f"CNOT {m}" for m in members], fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("True", fontsize=8)

            title = f"{MODEL_LABELS.get(model_name, model_name)} N={N}"
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{gname}\nTrue", fontsize=8)

    fig.suptitle("Ambiguity-group confusion zoom (RdYlGn: green=correct, red=confused)",
                 fontsize=11, y=1.01)
    out = OUT_DIR / "ambiguity_zoom.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def save_top_pairs(results, top_k=15):
    """Save top confused (true, pred) pairs across all (model, N) combos."""
    import csv

    rows = []
    for confmat, model_name, N, _ in results:
        norm = confmat.astype(float) / confmat.sum(axis=1, keepdims=True).clip(1)
        for i in range(24):
            for j in range(24):
                if i == j:
                    continue
                in_group = (
                    any(i in g and j in g for g in AMBIGUITY_GROUPS.values())
                )
                rows.append({
                    "model": model_name, "N": N,
                    "true": i, "pred": j,
                    "confusion_rate": round(float(norm[i, j]), 4),
                    "in_ambiguity_group": in_group,
                })

    rows.sort(key=lambda r: -r["confusion_rate"])
    out = OUT_DIR / "top_confused_pairs.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows[:top_k * len(results)])
    print(f"  Saved: {out}")

    # print top per model/N
    for confmat, model_name, N, _ in results:
        norm = confmat.astype(float) / confmat.sum(axis=1, keepdims=True).clip(1)
        off  = [(norm[i,j], i, j)
                for i in range(24) for j in range(24) if i != j]
        off.sort(reverse=True)
        print(f"\n  Top-5 confused pairs  [{model_name} N={N}]:")
        for rate, i, j in off[:5]:
            tag = "(ambiguity group)" if any(
                i in g and j in g for g in AMBIGUITY_GROUPS.values()) else ""
            print(f"    CNOT {i:02d} → pred {j:02d} : {rate:.3f}  {tag}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=["logreg", "deepsets"])
    parser.add_argument("--N-list",  nargs="+", type=int, default=[100, 300])
    parser.add_argument("--epochs",  type=int, default=80)
    args = parser.parse_args()

    import numpy as np
    data_dir = PROJECT_ROOT / "data" / "ml_dataset" / DATA_TAG
    shots  = np.load(data_dir / "shots.npy")
    labels = np.load(data_dir / "labels.npy").astype(np.int64)
    split  = np.load(data_dir / "split.npz")

    results = []
    for model_name in args.models:
        for N in args.N_list:
            print(f"\n=== {model_name}  N={N} ===")
            confmat, acc, _ = train_and_confuse(
                model_name, shots, labels, split, N=N, epochs=args.epochs)
            results.append((confmat, model_name, N, acc))
            plot_confusion_matrix(confmat, model_name, N, acc)

    plot_ambiguity_zoom(results)
    save_top_pairs(results)
    print("\nDone. Outputs →", OUT_DIR)


if __name__ == "__main__":
    main()
