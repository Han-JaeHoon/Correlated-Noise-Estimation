"""
Bayes-optimal ceiling for dominant-CNOT identification under the R1 fault model.

Under reset mode, the per-round detection event is i.i.d. given the dominant CNOT,
because the syndrome is linear in the data-qubit Pauli frame and the frame
increments are independent across rounds. The per-round detection event
distribution P(d | dominant=k) is computed as an XOR-convolution of 24
independent contributions (one per CNOT), each of which is a Bernoulli plus a
uniform draw from 15 non-identity 2-qubit Paulis.

The Bayes-optimal classifier predicts argmax_k Σ_t log P(d_t | k). Its accuracy
is estimated by Monte Carlo: sample T-round sequences from each true k* and
count argmax matches.
"""

from typing import Tuple

import numpy as np

from .postprocess import reshape_mid_measure_syndrome
from .simulator import make_repeated_stabilizer_qnode
from .surface_code_layout import enumerate_stabilizer_cnots


PAULI_PAIRS_15 = tuple(
    (a, b)
    for a in ("I", "X", "Y", "Z")
    for b in ("I", "X", "Y", "Z")
    if not (a == "I" and b == "I")
)
assert len(PAULI_PAIRS_15) == 15

N_STAB = 8
N_SYN = 1 << N_STAB  # 256


def _pair_to_error_lists(pair, control: int, target: int):
    pc, pt = pair
    types = []
    wires = []
    if pc != "I":
        types.append(pc)
        wires.append(int(control))
    if pt != "I":
        types.append(pt)
        wires.append(int(target))
    return types, wires


def _bits_to_int(bits: np.ndarray) -> int:
    """Pack an 8-bit array (LSB-first: bit 0 = first stabilizer) into 0..255."""
    return int(np.dot(bits.astype(np.int64), 1 << np.arange(bits.size)))


def compute_single_round_lookup(reset: bool = True) -> np.ndarray:
    """
    For each (CNOT, Pauli pair) compute the single-round syndrome contribution.

    Logical zero initial state, no prior fault: the round-0 syndrome equals the
    single-round detection event for that lone fault.

    Returns:
        contributions: shape (24, 15) int32. contributions[i, alpha] is the
            8-bit syndrome packed as an integer 0..255 (bit b = stabilizer b
            in the order [Z0, Z1, Z2, Z3, X0, X1, X2, X3]).
    """
    cnots = enumerate_stabilizer_cnots()
    n_cnot = len(cnots)
    n_pauli = len(PAULI_PAIRS_15)

    contributions = np.zeros((n_cnot, n_pauli), dtype=np.int32)

    qnode = make_repeated_stabilizer_qnode(
        n_rounds=1,
        shots=1,
        mid_measure=True,
        reset_after_measure=reset,
        return_probs=False,
    )

    for k, loc in enumerate(cnots):
        for alpha, pair in enumerate(PAULI_PAIRS_15):
            err_types, err_wires = _pair_to_error_lists(
                pair, loc["control"], loc["target"]
            )
            fault_schedule = [
                {
                    "round": 0,
                    "control": int(loc["control"]),
                    "target": int(loc["target"]),
                    "error_wires": err_wires,
                    "error_types": err_types,
                }
            ]
            raw = qnode(fault_schedule=fault_schedule)
            syn = reshape_mid_measure_syndrome(
                raw=raw, n_rounds=1, shots=1, n_stabilizers=N_STAB
            )[0, 0, :]  # (8,)
            contributions[k, alpha] = _bits_to_int(syn)

    return contributions


def _cnot_contribution_dist(p: float, syndromes: np.ndarray) -> np.ndarray:
    """
    Contribution distribution from one CNOT with fault probability p and a
    uniform draw over the given 15-syndrome list.
    """
    d = np.zeros(N_SYN, dtype=np.float64)
    d[0] = 1.0 - p
    n = syndromes.size
    for s in syndromes:
        d[int(s)] += p / n
    return d


def _xor_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR-convolution of two length-256 distributions, naive O(256 * nnz(b))."""
    out = np.zeros(N_SYN, dtype=np.float64)
    nz = np.flatnonzero(b)
    idx = np.arange(N_SYN)
    for s in nz:
        out += b[s] * a[idx ^ int(s)]
    return out


def per_round_distribution(
    p_bg: float, p_high: float, k_dom: int, contributions: np.ndarray
) -> np.ndarray:
    """
    P(detection event | dominant CNOT = k_dom).

    Returns: (256,) float64 probability vector.
    """
    n_cnot = contributions.shape[0]
    dist = np.zeros(N_SYN, dtype=np.float64)
    dist[0] = 1.0
    for i in range(n_cnot):
        p_i = p_high if i == k_dom else p_bg
        contrib = _cnot_contribution_dist(p_i, contributions[i])
        dist = _xor_convolve(dist, contrib)
    return dist


def compute_all_distributions(
    p_bg: float, p_high: float, contributions: np.ndarray
) -> np.ndarray:
    """Stack of per-class per-round distributions, shape (n_cnot, 256)."""
    n_cnot = contributions.shape[0]
    out = np.zeros((n_cnot, N_SYN), dtype=np.float64)
    for k in range(n_cnot):
        out[k] = per_round_distribution(p_bg, p_high, k, contributions)
    return out


def bayes_classifier_confusion(
    dists: np.ndarray,
    T: int,
    n_test: int,
    rng: np.random.Generator,
    log_eps: float = 1e-300,
) -> np.ndarray:
    """
    Monte Carlo Bayes-optimal classifier accuracy.

    For each true k* ∈ {0..23}, samples n_test sequences of length T from
    P(d | k*), then classifies each by argmax_k Σ_t log P(d_t | k).

    Returns:
        confusion: (K, K) int64. confusion[k_true, k_pred] = count.
    """
    K = dists.shape[0]
    log_dists = np.log(np.maximum(dists, log_eps))  # (K, 256)
    confusion = np.zeros((K, K), dtype=np.int64)

    outcomes = np.arange(N_SYN)
    for k_true in range(K):
        sampled = rng.choice(outcomes, size=(n_test, T), p=dists[k_true])
        # log_dists[:, sampled] -> (K, n_test, T), sum over T -> (K, n_test)
        log_lik = log_dists[:, sampled].sum(axis=2)
        predictions = log_lik.argmax(axis=0)  # (n_test,)
        for k_pred in range(K):
            confusion[k_true, k_pred] = int((predictions == k_pred).sum())
    return confusion


def overall_accuracy(confusion: np.ndarray) -> float:
    total = confusion.sum()
    if total == 0:
        return 0.0
    return float(np.diag(confusion).sum() / total)


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    return np.diag(confusion) / row_sums_safe
