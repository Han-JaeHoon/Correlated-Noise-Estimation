"""
Train one sequence classifier (RNN / GRU / Transformer) on one
scenario (R1 / R3b) and one T value. Save the best-on-val model and
report per-class + group accuracy on the held-out test set.

Inputs (must be created by `scripts/build_classifier_dataset.py`):
  data/classifier_dataset/{scenario}_train.npz
  data/classifier_dataset/{scenario}_val.npz
  data/classifier_dataset/{scenario}_test.npz

Outputs (under `data/analysis/9_classifier/{scenario}/{model}_T{T}/`):
  model.pt              best-on-val checkpoint
  metrics.json          per-epoch loss/acc + final test acc + per-class
  confusion.npy         (24, 24) test confusion matrix
  args.json             snapshot of cli arguments
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_DIR
from src.seq_classifier import ModelConfig, build_model, count_parameters


R1_AMBIG_GROUPS = [
    [6, 7],
    [14, 15, 17],
    [18, 19, 20],
    [22, 23],
]


# ---------- data loading ---------------------------------------------------


def load_split(path: Path, T: int):
    """Load NPZ split and truncate sequences to length T."""
    data = np.load(path, allow_pickle=False)
    syn = data["syndromes"]    # (N, T_max, 8) uint8
    lab = data["labels"]       # (N,) int64
    if syn.shape[1] < T:
        raise ValueError(f"requested T={T} but dataset T_max={syn.shape[1]}")
    syn = syn[:, :T, :]
    return torch.as_tensor(syn, dtype=torch.uint8), torch.as_tensor(lab, dtype=torch.long)


def make_loader(syn: torch.Tensor, lab: torch.Tensor, batch_size: int, shuffle: bool):
    ds = TensorDataset(syn, lab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0)


# ---------- training -------------------------------------------------------


def epoch_pass(model, loader, optim, loss_fn, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            # x to device non-blocking (float data, safe).
            # y kept on CPU — MPS non_blocking label transfers can produce
            # stale reads leading to wildly wrong accuracy on Apple Silicon.
            # Loss function receives y via blocking .to(device).
            x = x.to(device, non_blocking=True)
            y_dev = y.to(device)          # blocking — labels must be ready
            logits = model(x)
            loss = loss_fn(logits, y_dev)
            if train:
                optim.zero_grad()
                loss.backward()
                # mild grad clip — helps RNN and Transformer
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
            total_loss += loss.item() * x.size(0)
            # Compare on CPU to avoid MPS int64 comparison quirks
            total_correct += (logits.argmax(dim=1).cpu() == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def evaluate_test(model, loader, device):
    """Return (per_class_acc (24,), confusion (24,24), overall_acc, group_acc)."""
    model.eval()
    confusion = np.zeros((24, 24), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            yn = y.numpy()
            for yt, yp in zip(yn, pred):
                confusion[int(yt), int(yp)] += 1
    row_sums = confusion.sum(axis=1).clip(min=1)
    per_class = np.diag(confusion) / row_sums
    overall = float(np.diag(confusion).sum() / confusion.sum())
    # group accuracy: pred is correct if same R1 ambiguity group as true.
    # Singletons (CNOTs not in any ambiguity group) are their own group.
    in_ambig = set()
    for g in R1_AMBIG_GROUPS:
        in_ambig.update(g)
    groups = list(R1_AMBIG_GROUPS) + [[k] for k in range(24) if k not in in_ambig]
    group_of = {k: gi for gi, g in enumerate(groups) for k in g}
    g_correct = 0
    g_total = 0
    for kt in range(24):
        rt = confusion[kt].sum()
        if rt == 0:
            continue
        for kp in range(24):
            if group_of[kp] == group_of[kt]:
                g_correct += int(confusion[kt, kp])
        g_total += int(rt)
    group_acc = g_correct / max(g_total, 1)
    return per_class, confusion, overall, group_acc


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["rnn", "gru", "transformer"])
    ap.add_argument("--scenario", required=True, choices=["r1", "r2", "r3b"])
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--data-dir", type=Path,
                    default=DATA_DIR / "classifier_dataset")
    ap.add_argument("--out-root", type=Path,
                    default=DATA_DIR / "analysis" / "9_classifier")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8,
                    help="early stopping patience on val accuracy plateau")

    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--input-kind", choices=["raw", "byte"], default="raw")

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[device] {device}")

    # ----- data -----
    print("[data] loading splits and truncating to T...")
    t0 = time.time()
    syn_train, lab_train = load_split(args.data_dir / f"{args.scenario}_train.npz", args.T)
    syn_val, lab_val = load_split(args.data_dir / f"{args.scenario}_val.npz", args.T)
    syn_test, lab_test = load_split(args.data_dir / f"{args.scenario}_test.npz", args.T)
    print(f"  train {syn_train.shape}   val {syn_val.shape}   test {syn_test.shape}"
          f"   ({time.time()-t0:.1f}s)")

    train_loader = make_loader(syn_train, lab_train, args.batch_size, shuffle=True)
    val_loader = make_loader(syn_val, lab_val, args.batch_size, shuffle=False)
    test_loader = make_loader(syn_test, lab_test, args.batch_size, shuffle=False)

    # ----- model -----
    cfg = ModelConfig(
        name=args.model,
        d_model=args.d_model,
        hidden=args.hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        input_kind=args.input_kind,
    )
    model = build_model(cfg, max_T=max(args.T + 1, 2048)).to(device)
    n_params = count_parameters(model)
    print(f"[model] {args.model} params={n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    loss_fn = nn.CrossEntropyLoss()

    # ----- output dir -----
    out_dir = args.out_root / args.scenario / f"{args.model}_T{args.T}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- train loop -----
    metrics = {
        "args": vars(args).copy(),
        "n_params": n_params,
        "epochs": [],
        "best_val_acc": 0.0,
        "best_epoch": -1,
    }
    # args may contain Path/Tensor — strip
    metrics["args"] = {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in metrics["args"].items()}

    best_val = 0.0
    best_state = None
    patience_left = args.patience

    train_t0 = time.time()
    for epoch in range(args.epochs):
        ep_t0 = time.time()
        train_loss, train_acc = epoch_pass(model, train_loader, optim,
                                            loss_fn, device, train=True)
        val_loss, val_acc = epoch_pass(model, val_loader, optim, loss_fn,
                                        device, train=False)
        sched.step()
        ep_time = time.time() - ep_t0

        metrics["epochs"].append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": float(optim.param_groups[0]["lr"]),
            "time_s": ep_time,
        })

        improved = val_acc > best_val + 1e-4
        flag = "*" if improved else " "
        print(f"  ep {epoch:3d}{flag} train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
              f"   val_loss={val_loss:.4f} val_acc={val_acc:.4f}   ({ep_time:.1f}s)")

        if improved:
            best_val = val_acc
            # deepcopy keeps tensors on the same device — avoids potential
            # CPU↔MPS round-trip issues in load_state_dict on Apple Silicon.
            best_state = copy.deepcopy(model.state_dict())
            metrics["best_val_acc"] = float(best_val)
            metrics["best_epoch"] = int(epoch)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  [early stop] no improvement for {args.patience} epochs")
                break

    metrics["train_time_s"] = time.time() - train_t0

    # ----- restore best and test eval -----
    if best_state is not None:
        model.load_state_dict(best_state)
    per_class, confusion, overall, group_acc = evaluate_test(model, test_loader, device)
    print(f"[test] overall_acc={overall:.4f}   group_acc={group_acc:.4f}   "
          f"per_class min/median/max = {per_class.min():.3f}/"
          f"{float(np.median(per_class)):.3f}/{per_class.max():.3f}")

    metrics["test_overall_acc"] = float(overall)
    metrics["test_group_acc"] = float(group_acc)
    metrics["test_per_class_acc"] = per_class.tolist()

    # ----- save -----
    torch.save(best_state if best_state is not None else model.state_dict(),
               out_dir / "model.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(out_dir / "confusion.npy", confusion)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
