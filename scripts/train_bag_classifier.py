"""
Train and evaluate bag-of-shots classifiers for 24-class CNOT identification.

Models: empirical_bayes | logreg | mlp | deepsets
Input:  data/ml_dataset/<tag>/  (shots.npy, labels.npy, split.npz)

Usage:
    python scripts/train_bag_classifier.py --model logreg --N 100
    python scripts/train_bag_classifier.py --model deepsets --N 100 --epochs 50
    python scripts/train_bag_classifier.py --model empirical_bayes --N-sweep

Outputs: data/analysis/spatial_mixture/set_classifier/<model>_N<N>/metrics.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AMBIGUITY_GROUPS = [
    frozenset([6, 7]),
    frozenset([14, 15, 17]),
    frozenset([18, 19, 20]),
    frozenset([22, 23]),
]

DATA_TAG = "noreset_d3_phigh0.1_pbg0.01_n1000"
OUT_ROOT = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "set_classifier"


def load_data(tag=DATA_TAG):
    d = PROJECT_ROOT / "data" / "ml_dataset" / tag
    shots = np.load(d / "shots.npy")        # (N_total, 3, 8) int8
    labels = np.load(d / "labels.npy")      # (N_total,) int8
    split = np.load(d / "split.npz")
    return shots, labels.astype(np.int64), split


def group_accuracy(per_class_acc, n_classes=24):
    """Accuracy within each R1 ambiguity group (members predicted as any group member)."""
    result = {}
    for g in AMBIGUITY_GROUPS:
        members = sorted(g)
        accs = [per_class_acc.get(c, 0.0) for c in members]
        result[str(members)] = float(np.mean(accs))
    return result


# ---------------------------------------------------------------------------
# Empirical Bayes (no PyTorch)
# ---------------------------------------------------------------------------

def run_empirical_bayes(shots, labels, split, N_list, n_bags=500):
    from src.bag_classifier import EmpiricalBayes, build_vocab, shots_to_indices

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    train_shots, train_labels = shots[train_idx], labels[train_idx]
    test_shots, test_labels = shots[test_idx], labels[test_idx]

    vocab = build_vocab(train_shots.reshape(len(train_shots), -1))
    print(f"Vocabulary size: {len(vocab)}")

    model = EmpiricalBayes(n_classes=24)
    model.fit(train_shots, train_labels, vocab)

    results = {}
    for N in N_list:
        t0 = time.time()
        acc = model.evaluate(test_shots, test_labels, vocab, N=N, n_bags=n_bags, seed=0)
        elapsed = time.time() - t0
        results[N] = {"accuracy": acc, "N": N, "n_bags": n_bags, "time_s": round(elapsed, 2)}
        print(f"  N={N:4d}  acc={acc:.4f}  ({elapsed:.1f}s)")

    return results, vocab, model


# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------

def train_pytorch(model_name, shots, labels, split, N, epochs=80, lr=1e-3,
                  batch_size=64, reps_per_epoch=200, n_eval_bags=500, seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.bag_classifier import BagDataset, build_model, build_vocab

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"  device: {device}")

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    train_shots, train_labels = shots[train_idx], labels[train_idx]
    val_shots, val_labels = shots[val_idx], labels[val_idx]
    test_shots, test_labels = shots[test_idx], labels[test_idx]

    vocab = build_vocab(train_shots.reshape(len(train_shots), -1))
    V = len(vocab)
    shot_dim = shots.shape[1] * shots.shape[2]   # d * n_stab = 24

    rep = "raw" if model_name == "deepsets" else "hist"

    train_ds = BagDataset(train_shots, train_labels, vocab, N=N, rep=rep,
                          reps_per_epoch=reps_per_epoch, seed=seed)
    val_ds = BagDataset(val_shots, val_labels, vocab, N=N, rep=rep,
                        reps_per_epoch=100, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    net = build_model(model_name, vocab_size=V, shot_dim=shot_dim).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, patience, no_improve = 0.0, None, 12, 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # validation
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = net(x).argmax(dim=1).cpu().numpy()
                correct += (preds == np.asarray(y)).sum()
                total += len(y)
        val_acc = correct / total
        history["train_loss"].append(round(total_loss / len(train_loader), 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if no_improve >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    # test evaluation with best weights
    net.load_state_dict(best_state)
    net.eval()

    test_ds = BagDataset(test_shots, test_labels, vocab, N=N, rep=rep,
                         reps_per_epoch=n_eval_bags // 24 + 1, seed=seed + 2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    per_class_correct = {c: 0 for c in range(24)}
    per_class_total = {c: 0 for c in range(24)}
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = net(x).argmax(dim=1).cpu().numpy()
            for pred, true in zip(preds, y):
                per_class_correct[int(true)] += int(pred == int(true))
                per_class_total[int(true)] += 1

    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c]
        for c in range(24) if per_class_total[c] > 0
    }
    overall_acc = sum(per_class_correct.values()) / max(sum(per_class_total.values()), 1)

    return {
        "model": model_name,
        "N": N,
        "overall_acc": round(overall_acc, 4),
        "per_class_acc": {str(k): round(v, 4) for k, v in per_class_acc.items()},
        "ambiguity_group_acc": group_accuracy(per_class_acc),
        "best_val_acc": round(best_val_acc, 4),
        "history": history,
        "vocab_size": V,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg",
                        choices=["empirical_bayes", "logreg", "mlp", "deepsets"])
    parser.add_argument("--N", type=int, default=100, help="Shots per bag")
    parser.add_argument("--N-sweep", action="store_true",
                        help="Sweep N in [1,3,10,30,100,300] (empirical_bayes only for now)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tag", default=DATA_TAG)
    args = parser.parse_args()

    shots, labels, split = load_data(args.tag)
    print(f"Loaded: shots={shots.shape}, labels={labels.shape}")

    N_list = [1, 3, 10, 30, 100, 300] if args.N_sweep else [args.N]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.model == "empirical_bayes":
        print(f"\n=== Empirical Bayes  N_sweep={args.N_sweep} ===")
        results, vocab, _ = run_empirical_bayes(shots, labels, split, N_list)
        out_dir = OUT_ROOT / "empirical_bayes"
        out_dir.mkdir(exist_ok=True)
        out = {"model": "empirical_bayes", "results": {str(k): v for k, v in results.items()}}
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved → {out_dir / 'metrics.json'}")

    else:
        for N in N_list:
            print(f"\n=== {args.model}  N={N} ===")
            t0 = time.time()
            metrics = train_pytorch(args.model, shots, labels, split, N=N,
                                    epochs=args.epochs, lr=args.lr)
            metrics["train_time_s"] = round(time.time() - t0, 1)
            print(f"  overall_acc={metrics['overall_acc']:.4f}  "
                  f"({metrics['train_time_s']:.0f}s)")

            tag = f"{args.model}_N{N}"
            out_dir = OUT_ROOT / tag
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
