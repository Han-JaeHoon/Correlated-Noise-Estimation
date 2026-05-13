# src/fault_schedule.py

from itertools import combinations


def make_fixed_cnot_fault_schedule(
    control,
    target,
    error_rounds,
    error_types=("X", "Z"),
):
    """
    Make a fault schedule for a fixed directed CNOT(control -> target).

    Args:
        control: CNOT control wire.
        target: CNOT target wire.
        error_rounds: list[int], rounds where the fault is inserted.
        error_types: tuple/list[str], Pauli error types.

    Default:
        error_types=("X", "Z") means
            X on control
            Z on target

    Returns:
        list[dict]
    """

    fault_schedule = []

    for r in error_rounds:
        fault_schedule.append(
            {
                "round": int(r),
                "control": int(control),
                "target": int(target),
                "error_wires": [int(control), int(target)],
                "error_types": list(error_types),
            }
        )

    return fault_schedule


def get_faults_for_round(fault_schedule, current_round):
    """
    Return only the faults scheduled for current_round.
    """

    if fault_schedule is None:
        return []

    return [
        fault
        for fault in fault_schedule
        if int(fault["round"]) == int(current_round)
    ]


def all_error_round_subsets(n_rounds, include_empty=True):
    """
    Generate all subsets of {0, 1, ..., n_rounds-1}.

    Example:
        n_rounds = 3

        include_empty=True:
            []
            [0]
            [1]
            [2]
            [0, 1]
            [0, 2]
            [1, 2]
            [0, 1, 2]
    """

    subsets = []

    start_k = 0 if include_empty else 1

    for k in range(start_k, n_rounds + 1):
        for comb in combinations(range(n_rounds), k):
            subsets.append(list(comb))

    return subsets


def occurrence_vector_from_rounds(error_rounds, n_rounds):
    """
    Convert error_rounds to a binary occurrence vector.

    Example:
        error_rounds = [0, 2]
        n_rounds = 4

        output = [1, 0, 1, 0]
    """

    vector = [0] * n_rounds

    for r in error_rounds:
        vector[int(r)] = 1

    return vector