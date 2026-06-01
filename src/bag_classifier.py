"""
Bag-of-shots classifiers for spatial CNOT identification.

Each "bag" is N i.i.d. shots drawn from P_k (one d-round syndrome per shot).
Four models in order of complexity:

  empirical_bayes  -- empirical P̂_k from train data, log-likelihood argmax (no learning)
  logreg           -- linear(p̂) → 24 logits
  mlp              -- 2-layer ReLU(p̂) → 24 logits
  deepsets         -- embed each shot → sum-pool → linear head

All models share a common vocabulary built from the training shots.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Vocabulary: maps each observed 24-bit syndrome string to an integer index
# ---------------------------------------------------------------------------

def build_vocab(shots_flat):
    """shots_flat: (N, d*n_stab) int8 array.  Returns {tuple → int} dict."""
    unique = sorted({tuple(s.tolist()) for s in shots_flat})
    return {s: i for i, s in enumerate(unique)}


def shots_to_indices(shots, vocab):
    """(N, d, n_stab) int8 → (N,) int64 syndrome indices."""
    flat = shots.reshape(len(shots), -1)
    unk = len(vocab)
    return np.array([vocab.get(tuple(r.tolist()), unk) for r in flat], dtype=np.int64)


# ---------------------------------------------------------------------------
# Empirical Plug-in Bayes (no gradient, no PyTorch)
# ---------------------------------------------------------------------------

class EmpiricalBayes:
    """
    Estimates P̂_k(s) from training shots, then classifies a bag by
        k̂ = argmax_k  Σ_s  n_s · log(P̂_k(s) + ε)

    ε prevents log(0) when a syndrome appears in a test bag but not in
    the training pool for some class k.
    """

    def __init__(self, n_classes=24, eps=1e-9):
        self.n_classes = n_classes
        self.eps = eps
        self.log_pk = None   # (n_classes, vocab_size+1)  — +1 for OOV

    def fit(self, shots, labels, vocab):
        """shots: (N, d, n_stab) int8.  labels: (N,) int."""
        indices = shots_to_indices(shots, vocab)
        V = len(vocab) + 1   # +1 for OOV bucket
        counts = np.zeros((self.n_classes, V), dtype=np.float64)
        for idx, lbl in zip(indices, labels):
            counts[int(lbl), idx] += 1
        # Laplace smoothing within each class
        pk = (counts + self.eps) / (counts + self.eps).sum(axis=1, keepdims=True)
        self.log_pk = np.log(pk)   # (n_classes, V)

    def predict_bag(self, bag_indices):
        """bag_indices: (N,) int64.  Returns predicted class int."""
        scores = self.log_pk[:, bag_indices].sum(axis=1)   # (n_classes,)
        return int(np.argmax(scores))

    def evaluate(self, shots, labels, vocab, N, n_bags=500, seed=0):
        """Sample n_bags of size N from (shots, labels) and return accuracy."""
        rng = np.random.default_rng(seed)
        indices_all = shots_to_indices(shots, vocab)
        correct = 0
        classes = np.unique(labels)
        per_class = {int(c): np.where(labels == c)[0] for c in classes}
        for _ in range(n_bags):
            k = int(rng.choice(classes))
            pool = per_class[k]
            chosen = rng.choice(pool, size=min(N, len(pool)), replace=False)
            bag_idx = indices_all[chosen]
            pred = self.predict_bag(bag_idx)
            correct += pred == k
        return correct / n_bags


# ---------------------------------------------------------------------------
# PyTorch Dataset: on-the-fly bag sampling
# ---------------------------------------------------------------------------

class BagDataset(torch.utils.data.Dataset):
    """
    Yields (bag_tensor, label) pairs by sampling N shots per bag
    uniformly from the per-class pool.

    rep="hist"  → bag_tensor shape (vocab_size,)  float32  empirical p̂
    rep="raw"   → bag_tensor shape (N, d*n_stab)  float32  raw shot bits
    """

    def __init__(self, shots, labels, vocab, N, rep="hist", reps_per_epoch=200, seed=0):
        self.vocab = vocab
        self.V = len(vocab)
        self.N = N
        self.rep = rep
        self.reps_per_epoch = reps_per_epoch
        self.rng = np.random.default_rng(seed)

        indices = shots_to_indices(shots, vocab)
        classes = np.unique(labels)
        self.classes = classes
        self.per_class_indices = {int(c): indices[labels == c] for c in classes}
        self.per_class_shots = {
            int(c): shots[labels == c].reshape(len(shots[labels == c]), -1).astype(np.float32)
            for c in classes
        }

    def __len__(self):
        return len(self.classes) * self.reps_per_epoch

    def __getitem__(self, idx):
        k = int(self.classes[idx % len(self.classes)])
        pool_idx = self.per_class_indices[k]
        chosen = self.rng.choice(len(pool_idx), size=min(self.N, len(pool_idx)), replace=False)

        if self.rep == "hist":
            counts = np.bincount(pool_idx[chosen], minlength=self.V + 1)[: self.V]
            x = torch.tensor(counts / max(counts.sum(), 1), dtype=torch.float32)
        else:   # raw
            x = torch.tensor(self.per_class_shots[k][chosen], dtype=torch.float32)

        return x, k


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LogReg(nn.Module):
    def __init__(self, vocab_size, n_classes=24):
        super().__init__()
        self.fc = nn.Linear(vocab_size, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_classes=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DeepSets(nn.Module):
    """
    φ: shot embedding (d*n_stab → embed_dim)
    ρ: head (embed_dim → n_classes)
    Pooling: mean over shots in the bag.
    """

    def __init__(self, shot_dim, embed_dim=64, hidden=128, n_classes=24):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(shot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # x: (batch, N, shot_dim)
        z = self.phi(x).mean(dim=1)   # (batch, embed_dim)
        return self.rho(z)


def build_model(model_name, vocab_size, shot_dim, n_classes=24):
    if model_name == "logreg":
        return LogReg(vocab_size, n_classes)
    elif model_name == "mlp":
        return MLP(vocab_size, n_classes=n_classes)
    elif model_name == "deepsets":
        return DeepSets(shot_dim, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
