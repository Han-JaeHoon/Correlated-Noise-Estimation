# scripts/debug_run.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fault_schedule import make_fixed_cnot_fault_schedule
from src.simulator import make_repeated_stabilizer_qnode
from src.postprocess import (
    reshape_mid_measure_syndrome,
    compute_detection_events,
    print_single_shot_sequence,
)


def main():
    n_rounds = 4
    shots = 1

    qnode_mid = make_repeated_stabilizer_qnode(
        n_rounds=n_rounds,
        shots=shots,
        mid_measure=True,
        reset_after_measure=True,
    )

    fault_schedule = make_fixed_cnot_fault_schedule(
        control=0,
        target=9,
        error_rounds=[0, 2],
        error_types=("X", "Z"),
    )

    raw = qnode_mid(fault_schedule=fault_schedule)

    syndrome = reshape_mid_measure_syndrome(
        raw=raw,
        n_rounds=n_rounds,
        shots=shots,
    )

    detection_events = compute_detection_events(syndrome)

    print_single_shot_sequence(
        syndrome=syndrome,
        detection_events=detection_events,
        shot_idx=0,
    )

    qnode_no_mid = make_repeated_stabilizer_qnode(
        n_rounds=6,
        shots=1,
        mid_measure=False,
        reset_after_measure=False,
        return_probs=False,
    )

    fault_schedule = make_fixed_cnot_fault_schedule(
        control=0,
        target=9,
        error_rounds=[0],
        error_types=("X", "Z"),
    )

    final_sample = qnode_no_mid(fault_schedule=fault_schedule)

    print()
    print("Final ancilla sample, no mid-measurement:")
    print(final_sample)


if __name__ == "__main__":
    main()