"""
Sequence classifiers for the dominant-CNOT identification task.

All three models take an input of shape (B, T, 8) of uint8 / float32
syndrome bits (one stabilizer per channel) and emit class logits of shape
(B, 24) over the 24 directed CNOT locations.

We expose three architectures matching README §16's progression:

  * `VanillaRNN` — plain Elman RNN. Baseline that suffers from
    vanishing-gradient on long sequences (T >> 50).
  * `GRUClassifier` — gated recurrent unit. The "everyday" baseline for
    sequence classification at this scale.
  * `TransformerClassifier` — small encoder-only Transformer with a
    learned [CLS] token whose final-layer embedding is the classifier
    head's input.

Two input encodings are supported:

  * `input_kind = "raw"` (default): the 8-bit syndrome row is linearly
    projected to `d_model`. Simple, works for any model.
  * `input_kind = "byte"`: each row is packed to an int in {0..255} and
    embedded via a learned 256-entry lookup. Slightly more parameters
    but lets the model treat syndrome patterns as discrete tokens.

A learned positional embedding is added for the Transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


N_STAB = 8
N_CLASSES = 24


# ---------- input encoder --------------------------------------------------


class SyndromeEncoder(nn.Module):
    """
    Embed a (B, T, 8) syndrome stream into a (B, T, d_model) tensor.

    input_kind:
      "raw"  - linear projection of the 8 binary stabilizer bits as float.
      "byte" - pack to int in [0, 256), apply nn.Embedding(256, d_model).
    """

    def __init__(self, d_model: int, input_kind: str = "raw"):
        super().__init__()
        if input_kind == "raw":
            self.proj = nn.Linear(N_STAB, d_model)
            self.embed = None
        elif input_kind == "byte":
            self.proj = None
            self.embed = nn.Embedding(1 << N_STAB, d_model)
            # precompute bit weights for packing
            self.register_buffer(
                "bit_weights",
                (1 << torch.arange(N_STAB, dtype=torch.long)),
            )
        else:
            raise ValueError(f"unknown input_kind: {input_kind!r}")
        self.input_kind = input_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 8) uint8 or float32
        if self.input_kind == "raw":
            return self.proj(x.float())
        # byte encoding
        x_long = x.long()
        packed = (x_long * self.bit_weights).sum(dim=-1)  # (B, T)
        return self.embed(packed)


# ---------- vanilla RNN -----------------------------------------------------


class VanillaRNN(nn.Module):
    """
    Plain Elman RNN (`nn.RNN`) as a stress-test baseline. Included to show
    the vanishing-gradient pathology on long T: under T=1000 we expect this
    model to underperform GRU substantially.
    """

    def __init__(
        self,
        d_model: int = 64,
        hidden: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        n_classes: int = N_CLASSES,
        input_kind: str = "raw",
    ):
        super().__init__()
        self.encoder = SyndromeEncoder(d_model, input_kind=input_kind)
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=n_layers,
            nonlinearity="tanh",
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)            # (B, T, d_model)
        out, _ = self.rnn(emb)           # (B, T, hidden)
        # take final hidden state
        last = out[:, -1, :]
        return self.head(last)


# ---------- GRU ------------------------------------------------------------


class GRUClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        hidden: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        n_classes: int = N_CLASSES,
        input_kind: str = "raw",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.encoder = SyndromeEncoder(d_model, input_kind=input_kind)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        last_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(last_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        out, _ = self.gru(emb)            # (B, T, hidden*dir)
        return self.head(out[:, -1, :])


# ---------- Transformer -----------------------------------------------------


class TransformerClassifier(nn.Module):
    """
    Small encoder-only Transformer with learned [CLS] token prepended.

    For long sequences (T >> 1000) self-attention is O(T^2) and may need
    chunking; this class is intended for T up to a few thousand.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        n_classes: int = N_CLASSES,
        input_kind: str = "raw",
        max_T: int = 2048,
    ):
        super().__init__()
        self.encoder = SyndromeEncoder(d_model, input_kind=input_kind)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # learned positional embedding (input has T + 1 positions including CLS)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_T + 1, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.max_T = max_T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.max_T:
            raise ValueError(f"T={T} > max_T={self.max_T}")
        emb = self.encoder(x)                                  # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, d_model)
        seq = torch.cat([cls, emb], dim=1)                      # (B, T+1, d_model)
        seq = seq + self.pos_emb[:, : T + 1, :]
        out = self.transformer(seq)                             # (B, T+1, d_model)
        cls_out = self.norm(out[:, 0, :])
        return self.head(cls_out)


# ---------- registry --------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    d_model: int = 64
    hidden: int = 128
    n_layers: int = 1
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    input_kind: str = "raw"


def build_model(cfg: ModelConfig, max_T: int = 2048) -> nn.Module:
    if cfg.name == "rnn":
        return VanillaRNN(
            d_model=cfg.d_model, hidden=cfg.hidden, n_layers=cfg.n_layers,
            dropout=cfg.dropout, input_kind=cfg.input_kind,
        )
    if cfg.name == "gru":
        return GRUClassifier(
            d_model=cfg.d_model, hidden=cfg.hidden, n_layers=cfg.n_layers,
            dropout=cfg.dropout, input_kind=cfg.input_kind,
        )
    if cfg.name == "transformer":
        return TransformerClassifier(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, dropout=cfg.dropout,
            input_kind=cfg.input_kind, max_T=max_T,
        )
    raise ValueError(f"unknown model name: {cfg.name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
