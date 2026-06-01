"""
Large-N sweep: N in {500, 1000} using n12000 dataset.
Models: logreg, deepsets  (fastest convergence at high N)

Outputs: data/analysis/spatial_mixture/set_classifier_largeN/
"""

import json, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TAG = "noreset_d3_phigh0.1_pbg0.01_n12000"
OUT_ROOT = PROJECT_ROOT / "data" / "analysis" / "spatial_mixture" / "set_classifier_largeN"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

AMBIGUITY_GROUPS = [
    frozenset([6, 7]),
    frozenset([14, 15, 17]),
    frozenset([18, 19, 20]),
    frozenset([22, 23]),
]


def group_accuracy(per_class_acc):
    result = {}
    for g in AMBIGUITY_GROUPS:
        members = sorted(g)
        result[str(members)] = float(np.mean([per_class_acc.get(c, 0.0) for c in members]))
    return result


def train_cell(model_name, shots, labels, split, N, epochs=80, seed=42):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from src.bag_classifier import BagDataset, build_model, build_vocab

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"  device={device}")

    train_shots = shots[split["train_idx"]]
    train_labels = labels[split["train_idx"]]
    val_shots   = shots[split["val_idx"]]
    val_labels   = labels[split["val_idx"]]
    test_shots  = shots[split["test_idx"]]
    test_labels  = labels[split["test_idx"]]

    vocab    = build_vocab(train_shots.reshape(len(train_shots), -1))
    V        = len(vocab)
    shot_dim = shots.shape[1] * shots.shape[2]
    rep      = "raw" if model_name == "deepsets" else "hist"

    train_ds = BagDataset(train_shots, train_labels, vocab, N=N, rep=rep,
                          reps_per_epoch=200, seed=seed)
    val_ds   = BagDataset(val_shots, val_labels, vocab, N=N, rep=rep,
                          reps_per_epoch=100, seed=seed+1)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    net = build_model(model_name, vocab_size=V, shot_dim=shot_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_val, best_state, patience, no_improve = 0.0, None, 12, 0

    for epoch in range(1, epochs + 1):
        net.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long, device=device)
            opt.zero_grad(); crit(net(x), y).backward(); opt.step()
        sch.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = net(x.to(device)).argmax(1).cpu().numpy()
                correct += (preds == np.asarray(y)).sum()
                total   += len(y)
        val_acc = correct / total
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  early stop @ epoch {epoch}"); break

    # ── test evaluation ─────────────────────────────────────────────────
    net.load_state_dict(best_state); net.eval()
    from src.bag_classifier import shots_to_indices
    rng = np.random.default_rng(seed + 99)

    if rep == "hist":
        test_idx_arr = shots_to_indices(test_shots, vocab)

    per_class_correct = {c: 0 for c in range(24)}
    per_class_total   = {c: 0 for c in range(24)}
    n_bags_per_class  = 50

    with torch.no_grad():
        for true_k in range(24):
            pool = np.where(test_labels == true_k)[0]
            for _ in range(n_bags_per_class):
                chosen = rng.choice(len(pool), size=min(N, len(pool)), replace=True)
                chosen_global = pool[chosen]
                if rep == "hist":
                    bag_idx = test_idx_arr[chosen_global]
                    counts  = np.bincount(bag_idx, minlength=V+1)[:V]
                    x = torch.tensor(counts/max(counts.sum(),1),
                                     dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    raw = test_shots[chosen_global].reshape(len(chosen),-1).astype(np.float32)
                    x   = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(device)
                pred = int(net(x).argmax(1).cpu())
                per_class_correct[true_k] += int(pred == true_k)
                per_class_total[true_k]   += 1

    per_class_acc = {c: per_class_correct[c]/per_class_total[c]
                     for c in range(24) if per_class_total[c] > 0}
    overall_acc   = np.mean(list(per_class_acc.values()))
    return {
        "model": model_name, "N": N,
        "overall_acc":        round(float(overall_acc), 4),
        "per_class_acc":      {str(k): round(v,4) for k,v in per_class_acc.items()},
        "ambiguity_group_acc": group_accuracy(per_class_acc),
        "best_val_acc":        round(best_val, 4),
    }


def main():
    data_dir = PROJECT_ROOT / "data" / "ml_dataset" / DATA_TAG
    shots  = np.load(data_dir / "shots.npy")
    labels = np.load(data_dir / "labels.npy").astype(np.int64)
    split  = np.load(data_dir / "split.npz")
    print(f"Loaded: shots={shots.shape}")

    for model_name in ["logreg", "deepsets"]:
        for N in [500, 1000]:
            print(f"\n{'='*50}")
            print(f"  model={model_name}  N={N}")
            print(f"{'='*50}")
            t0 = time.time()
            metrics = train_cell(model_name, shots, labels, split, N=N)
            metrics["train_time_s"] = round(time.time()-t0, 1)
            print(f"  overall_acc={metrics['overall_acc']:.4f}  ({metrics['train_time_s']:.0f}s)")

            out_dir = OUT_ROOT / f"{model_name}_N{N}"
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    # print summary
    print("\n=== LARGE-N SUMMARY ===")
    for model_name in ["logreg", "deepsets"]:
        for N in [500, 1000]:
            p = OUT_ROOT / f"{model_name}_N{N}" / "metrics.json"
            if p.exists():
                d = json.load(open(p))
                print(f"  {model_name:<10} N={N:5d}  acc={d['overall_acc']:.4f}")


if __name__ == "__main__":
    main()
