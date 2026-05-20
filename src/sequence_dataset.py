# src/sequence_dataset.py

"""
R1 sequence dataset generation.

For a given (faulty_cnot_id, T, p_bg, p_high), generate N independent
T-round syndrome sequences with no decoder applied. The dataset is saved as:

    sequences.npz     # syndromes: (N, T, 8) uint8, seeds: (N,) int64
    metadata.json     # all generation parameters
    fault_logs.jsonl  # one JSON-encoded fault schedule per line (oracle log)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .decoder import IdentityDecoder
from .sequence_runner import run_sequence
from .stochastic_faults import BackgroundElevatedSampler


def generate_r1_dataset(
    n_samples: int,
    T: int,
    p_bg: float,
    p_high: float,
    faulty_cnot_id: Optional[int],
    seed: int,
    mid_measure: bool = True,
    reset_after_measure: bool = True,
    progress_every: int = 0,
) -> dict:
    """
    Generate `n_samples` R1 sequences.

    Returns:
        {
            'syndromes':   (N, T, 8) uint8,
            'seeds':       (N,) int64,
            'fault_logs':  list[list[dict]],
            'metadata':    dict,
        }
    """
    decoder = IdentityDecoder()

    rng_master = np.random.default_rng(seed)
    seeds = rng_master.integers(
        low=0, high=2**63 - 1, size=n_samples,
    ).astype(np.int64)

    syndromes = np.zeros((n_samples, T, 8), dtype=np.uint8)
    fault_logs = []

    for i in range(n_samples):
        per_sample_rng = np.random.default_rng(int(seeds[i]))
        sampler = BackgroundElevatedSampler(
            p_bg=p_bg,
            p_high=p_high,
            faulty_cnot_id=faulty_cnot_id,
            rng=per_sample_rng,
        )
        out = run_sequence(
            T=T,
            sampler=sampler,
            decoder=decoder,
            mid_measure=mid_measure,
            reset_after_measure=reset_after_measure,
        )
        syndromes[i] = out["syndromes"]
        fault_logs.append(out["fault_schedule"])

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{i + 1}/{n_samples}] done")

    metadata = {
        "regime": "R1",
        "decoder": "IdentityDecoder",
        "n_samples": int(n_samples),
        "T": int(T),
        "p_bg": float(p_bg),
        "p_high": float(p_high),
        "faulty_cnot_id": faulty_cnot_id,
        "seed": int(seed),
        "mid_measure": bool(mid_measure),
        "reset_after_measure": bool(reset_after_measure),
        "n_stabilizers": 8,
    }

    return {
        "syndromes": syndromes,
        "seeds": seeds,
        "fault_logs": fault_logs,
        "metadata": metadata,
    }


def save_r1_dataset(out_dir, dataset: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "sequences.npz",
        syndromes=dataset["syndromes"],
        seeds=dataset["seeds"],
    )

    (out_dir / "metadata.json").write_text(
        json.dumps(dataset["metadata"], indent=2)
    )

    with open(out_dir / "fault_logs.jsonl", "w") as f:
        for log in dataset["fault_logs"]:
            f.write(json.dumps(log) + "\n")
