# src/ceiling_r3b.py

"""
Monte-Carlo R3b ceiling for dominant-CNOT identification with the
LookupDecoder window-by-window runner.

R3b is NOT round-iid: the decoder carries a data-qubit Pauli frame from one
round to the next, so the per-round detection event distribution depends on
the entire correction history. Hence the R1 analytic closed-form
(XOR-convolution of 24 per-CNOT contributions; src/ceiling.py) does not
apply directly.

Strategy here: estimate the per-round marginal P(d | k_dom) by Monte
Carlo simulation (averaging over rounds 1..T-1 of N sequences for each
true k_dom), then apply the same Bayes-optimal classifier
    \hat{k} = argmax_k Σ_t log P(d_t | k)
that R1 uses. This is a lower bound on the true R3b ceiling — the true
optimal classifier could exploit round-to-round correlations the marginal
ignores — but it directly answers the structurally important question:
> does the R3b per-round marginal break the ambiguity groups that the R1
> per-round marginal collapses together?

If yes (marginals diverge between two former-ambiguity-group members),
that proves the §13.7 hypothesis in the most directly comparable frame
(per-round marginal vs per-round marginal). If no, a follow-up using
full-sequence likelihood is needed.

A drop-round-0 option is provided because the very first round always
starts from frame=0 and therefore has the same distribution as R1's
round; the R3b-specific behavior only emerges from round 1 onward.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ceiling import N_STAB, N_SYN
from .decoder import IdentityDecoder, LookupDecoder
from .sequence_runner import run_sequence
from .stochastic_faults import BackgroundElevatedSampler
from .surface_code_layout import enumerate_stabilizer_cnots


N_CNOT = len(enumerate_stabilizer_cnots())  # 24


def _bits_to_int_row(bits_row: np.ndarray) -> int:
    return int(np.dot(bits_row.astype(np.int64), 1 << np.arange(bits_row.size)))


def simulate_class_sequences(
    k_dom: int,
    n_samples: int,
    T: int,
    p_bg: float,
    p_high: float,
    decoder_kind: str,
    rng: np.random.Generator,
    reset_after_measure: bool = True,
) -> np.ndarray:
    """
    Run `n_samples` independent T-round sequences with dominant CNOT = k_dom,
    under either the R1 (`'identity'`) or R3b (`'lookup'`) decoder.

    Returns:
        syndromes: (n_samples, T, 8) uint8.
    """
    if decoder_kind == "identity":
        decoder = IdentityDecoder()
    elif decoder_kind == "lookup":
        decoder = LookupDecoder(reset=reset_after_measure)
    else:
        raise ValueError(f"unknown decoder_kind {decoder_kind!r}")

    syndromes = np.zeros((n_samples, T, 8), dtype=np.uint8)
    for i in range(n_samples):
        seed_i = int(rng.integers(low=0, high=2**63 - 1))
        sampler = BackgroundElevatedSampler(
            p_bg=p_bg,
            p_high=p_high,
            faulty_cnot_id=int(k_dom),
            rng=np.random.default_rng(seed_i),
        )
        out = run_sequence(
            T=T,
            sampler=sampler,
            decoder=decoder,
            reset_after_measure=reset_after_measure,
        )
        syndromes[i] = out["syndromes"]
    return syndromes


def estimate_per_round_marginal(
    syndromes: np.ndarray,
    drop_first_round: bool = True,
    laplace_alpha: float = 1.0,
) -> np.ndarray:
    """
    Pool all (sample, round) pairs and estimate P(d | k_dom).

    Args:
        syndromes: (n_samples, T, 8) uint8.
        drop_first_round: ignore round 0 (which is decoder-agnostic since
            the frame is always 0 at that point).
        laplace_alpha: additive smoothing in the histogram so unseen
            outcomes get a small probability for log-likelihood stability.

    Returns:
        (256,) float64 probability vector.
    """
    n_samples, T, n_stab = syndromes.shape
    assert n_stab == N_STAB
    start = 1 if (drop_first_round and T > 1) else 0
    used = syndromes[:, start:, :].reshape(-1, n_stab)
    counts = np.zeros(N_SYN, dtype=np.float64)
    for row in used:
        counts[_bits_to_int_row(row)] += 1.0
    counts += laplace_alpha
    counts /= counts.sum()
    return counts


def build_class_marginals(
    n_samples_train: int,
    T_train: int,
    p_bg: float,
    p_high: float,
    decoder_kind: str,
    rng: np.random.Generator,
    drop_first_round: bool = True,
    laplace_alpha: float = 1.0,
) -> np.ndarray:
    """
    For each candidate k_dom ∈ 0..23, simulate `n_samples_train` sequences
    and estimate P(d | k_dom).

    Returns:
        marginals: (24, 256) float64.
    """
    marginals = np.zeros((N_CNOT, N_SYN), dtype=np.float64)
    for k in range(N_CNOT):
        syn = simulate_class_sequences(
            k_dom=k,
            n_samples=n_samples_train,
            T=T_train,
            p_bg=p_bg,
            p_high=p_high,
            decoder_kind=decoder_kind,
            rng=rng,
        )
        marginals[k] = estimate_per_round_marginal(
            syn,
            drop_first_round=drop_first_round,
            laplace_alpha=laplace_alpha,
        )
    return marginals


def bayes_classify_with_marginals(
    test_seqs: np.ndarray,
    log_marginals: np.ndarray,
    drop_first_round: bool = True,
) -> np.ndarray:
    """
    Marginal Bayes classifier:
        \hat{k} = argmax_k Σ_t log P(d_t | k)

    Args:
        test_seqs: (n_test, T, 8) uint8.
        log_marginals: (24, 256) float64.

    Returns:
        predictions: (n_test,) int.
    """
    n_test, T, n_stab = test_seqs.shape
    assert n_stab == N_STAB
    start = 1 if (drop_first_round and T > 1) else 0
    used = test_seqs[:, start:, :]  # (n_test, T-1, 8)
    # pack each (round, sample) to int 0..255
    weights = (1 << np.arange(N_STAB)).astype(np.int64)
    packed = (used.astype(np.int64) * weights).sum(axis=2)  # (n_test, T-1)

    log_lik = log_marginals[:, packed].sum(axis=2)  # (K, n_test)
    return log_lik.argmax(axis=0)


@dataclass
class CeilingRun:
    confusion: np.ndarray            # (24, 24) int
    marginals: np.ndarray            # (24, 256) float
    p_bg: float
    p_high: float
    T_train: int
    T_test: int
    n_train: int
    n_test: int
    decoder_kind: str


def compute_ceiling(
    p_bg: float,
    p_high: float,
    T_train: int,
    T_test: int,
    n_train: int,
    n_test: int,
    decoder_kind: str,
    seed: int = 0,
    drop_first_round: bool = True,
    laplace_alpha: float = 1.0,
) -> CeilingRun:
    """
    Full pipeline: simulate train sequences → estimate marginals → simulate
    test sequences → classify → confusion matrix.

    Train and test use independent seeds (derived from the master `seed`).
    """
    master_rng = np.random.default_rng(seed)
    train_rng = np.random.default_rng(int(master_rng.integers(0, 2**63 - 1)))
    test_rng = np.random.default_rng(int(master_rng.integers(0, 2**63 - 1)))

    marginals = build_class_marginals(
        n_samples_train=n_train,
        T_train=T_train,
        p_bg=p_bg,
        p_high=p_high,
        decoder_kind=decoder_kind,
        rng=train_rng,
        drop_first_round=drop_first_round,
        laplace_alpha=laplace_alpha,
    )
    log_marginals = np.log(np.maximum(marginals, 1e-300))

    confusion = np.zeros((N_CNOT, N_CNOT), dtype=np.int64)
    for k_true in range(N_CNOT):
        test_seq = simulate_class_sequences(
            k_dom=k_true,
            n_samples=n_test,
            T=T_test,
            p_bg=p_bg,
            p_high=p_high,
            decoder_kind=decoder_kind,
            rng=test_rng,
        )
        preds = bayes_classify_with_marginals(
            test_seq, log_marginals, drop_first_round=drop_first_round
        )
        for k_pred in range(N_CNOT):
            confusion[k_true, k_pred] = int((preds == k_pred).sum())

    return CeilingRun(
        confusion=confusion,
        marginals=marginals,
        p_bg=p_bg,
        p_high=p_high,
        T_train=T_train,
        T_test=T_test,
        n_train=n_train,
        n_test=n_test,
        decoder_kind=decoder_kind,
    )


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1)
    safe = np.where(row_sums == 0, 1, row_sums)
    return np.diag(confusion).astype(np.float64) / safe


def group_accuracy(
    confusion: np.ndarray,
    groups: Optional[list] = None,
) -> tuple:
    """
    Group accuracy: a prediction is 'correct' if k_pred is in the same group
    as k_true. Returns (overall, per-group acc, per-class collapsed acc).
    """
    if groups is None:
        # Default = R1 15-Pauli ambiguity groups from README §13.2
        groups = [
            [6, 7],
            [14, 15, 17],
            [18, 19, 20],
            [22, 23],
        ]
    # add singletons for all not in any explicit group
    in_group = set()
    for g in groups:
        in_group.update(g)
    canonical_groups = list(groups)
    for k in range(N_CNOT):
        if k not in in_group:
            canonical_groups.append([k])

    # build k → group id
    group_of = {}
    for gi, g in enumerate(canonical_groups):
        for k in g:
            group_of[k] = gi

    per_class = []
    for k_true in range(N_CNOT):
        row_total = confusion[k_true].sum()
        if row_total == 0:
            per_class.append(0.0)
            continue
        correct = 0
        gid_true = group_of[k_true]
        for k_pred in range(N_CNOT):
            if group_of[k_pred] == gid_true:
                correct += int(confusion[k_true, k_pred])
        per_class.append(correct / row_total)

    overall = float(np.mean(per_class))
    return overall, per_class, canonical_groups
