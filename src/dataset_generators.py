# src/dataset_generators.py

import json

import numpy as np

from .fault_schedule import (
    make_fixed_cnot_fault_schedule,
    all_error_round_subsets,
    occurrence_vector_from_rounds,
)
from .postprocess import (
    reshape_mid_measure_syndrome,
    compute_detection_events,
    normalize_final_sample,
    normalize_final_probs,
    flatten_bits,
    comma_join,
)
from .surface_code_layout import enumerate_stabilizer_cnots


def situation_name(mid_measure, reset_after_measure, return_probs):
    """
    Convert simulation options into a folder-friendly situation name.
    """

    if mid_measure and reset_after_measure:
        return "mid_measure_reset"

    if mid_measure and not reset_after_measure:
        return "mid_measure_no_reset"

    if (not mid_measure) and return_probs:
        return "no_mid_measure_final_probs"

    return "no_mid_measure_final_sample"


def process_qnode_output(
    raw,
    n_rounds,
    shots,
    mid_measure,
    return_probs,
):
    """
    Convert raw QNode output into standardized data fields.

    Returns:
        dict with:
            syndrome_bits
            detection_bits
            final_bits
            probabilities
            raw_output_json
    """

    syndrome_bits = ""
    detection_bits = ""
    final_bits = ""
    probabilities = ""

    if mid_measure:
        syndrome = reshape_mid_measure_syndrome(
            raw=raw,
            n_rounds=n_rounds,
            shots=shots,
            n_stabilizers=8,
        )

        detection = compute_detection_events(syndrome)

        # 현재 dataset은 shots=1을 기본으로 사용.
        # shots>1인 경우에는 모든 shot이 flatten되어 저장됨.
        syndrome_bits = flatten_bits(syndrome)
        detection_bits = flatten_bits(detection)

        raw_output_json = json.dumps(
            {
                "syndrome": syndrome.astype(int).tolist(),
                "detection_events": detection.astype(int).tolist(),
            }
        )

    else:
        if return_probs:
            probs = normalize_final_probs(raw)
            probabilities = ",".join(str(float(x)) for x in probs.flatten())

            raw_output_json = json.dumps(
                {
                    "probabilities": probs.astype(float).tolist(),
                }
            )

        else:
            final = normalize_final_sample(raw)
            final_bits = flatten_bits(final)

            raw_output_json = json.dumps(
                {
                    "final_bits": final.astype(int).tolist(),
                }
            )

    return {
        "syndrome_bits": syndrome_bits,
        "detection_bits": detection_bits,
        "final_bits": final_bits,
        "probabilities": probabilities,
        "raw_output_json": raw_output_json,
    }


def base_row(
    sample_id,
    mode,
    n_rounds,
    shots,
    mid_measure,
    reset_after_measure,
    return_probs,
    control,
    target,
    error_rounds,
    error_types,
    fault_schedule,
):
    """
    Common row fields for all datasets.
    """

    occurrence_vector = occurrence_vector_from_rounds(
        error_rounds=error_rounds,
        n_rounds=n_rounds,
    )

    return {
        "sample_id": int(sample_id),
        "mode": mode,
        "n_rounds": int(n_rounds),
        "shots": "" if shots is None else int(shots),
        "mid_measure": bool(mid_measure),
        "reset_after_measure": bool(reset_after_measure),
        "return_probs": bool(return_probs),
        "control": int(control),
        "target": int(target),
        "error_rounds": comma_join(error_rounds),
        "occurrence_vector": comma_join(occurrence_vector),
        "error_types": comma_join(error_types),
        "error_wires": f"{int(control)},{int(target)}",
        "fault_schedule_json": json.dumps(fault_schedule),
    }


def generate_fixed_cnot_round_pattern_rows(
    qnode,
    n_rounds,
    shots,
    control,
    target,
    error_types=("X", "Z"),
    include_empty=True,
    mid_measure=True,
    reset_after_measure=True,
    return_probs=False,
):
    """
    Dataset type B.

    Fixed CNOT(control -> target), sweep all round occurrence patterns.

    Example for n_rounds=4:
        []
        [0]
        [1]
        [2]
        [3]
        [0,1]
        ...
        [0,1,2,3]

    Each row:
        one occurrence pattern -> output data.
    """

    rows = []

    round_patterns = all_error_round_subsets(
        n_rounds=n_rounds,
        include_empty=include_empty,
    )

    mode = situation_name(
        mid_measure=mid_measure,
        reset_after_measure=reset_after_measure,
        return_probs=return_probs,
    )

    for sample_id, error_rounds in enumerate(round_patterns):
        fault_schedule = make_fixed_cnot_fault_schedule(
            control=control,
            target=target,
            error_rounds=error_rounds,
            error_types=error_types,
        )

        raw = qnode(fault_schedule=fault_schedule)

        row = base_row(
            sample_id=sample_id,
            mode=mode,
            n_rounds=n_rounds,
            shots=shots,
            mid_measure=mid_measure,
            reset_after_measure=reset_after_measure,
            return_probs=return_probs,
            control=control,
            target=target,
            error_rounds=error_rounds,
            error_types=error_types,
            fault_schedule=fault_schedule,
        )

        row.update(
            {
                "dataset_type": "fixed_cnot_round_patterns",
                "stab_type": "",
                "stab_index": "",
            }
        )

        row.update(
            process_qnode_output(
                raw=raw,
                n_rounds=n_rounds,
                shots=shots,
                mid_measure=mid_measure,
                return_probs=return_probs,
            )
        )

        rows.append(row)

    return rows


def generate_all_cnot_single_fault_rows(
    qnode,
    n_rounds,
    shots,
    fault_round=0,
    error_types=("X", "Z"),
    mid_measure=True,
    reset_after_measure=True,
    return_probs=False,
):
    """
    Dataset type A.

    Sweep all directed CNOT locations.

    For each CNOT:
        insert one fault at fault_round.

    Each row:
        one CNOT location -> output data.
    """

    rows = []
    cnot_locations = enumerate_stabilizer_cnots()

    mode = situation_name(
        mid_measure=mid_measure,
        reset_after_measure=reset_after_measure,
        return_probs=return_probs,
    )

    for sample_id, loc in enumerate(cnot_locations):
        control = loc["control"]
        target = loc["target"]

        fault_schedule = make_fixed_cnot_fault_schedule(
            control=control,
            target=target,
            error_rounds=[fault_round],
            error_types=error_types,
        )

        raw = qnode(fault_schedule=fault_schedule)

        row = base_row(
            sample_id=sample_id,
            mode=mode,
            n_rounds=n_rounds,
            shots=shots,
            mid_measure=mid_measure,
            reset_after_measure=reset_after_measure,
            return_probs=return_probs,
            control=control,
            target=target,
            error_rounds=[fault_round],
            error_types=error_types,
            fault_schedule=fault_schedule,
        )

        row.update(
            {
                "dataset_type": "all_cnot_single_fault",
                "stab_type": loc["stab_type"],
                "stab_index": int(loc["stab_index"]),
                "cnot_index": int(sample_id),
            }
        )

        row.update(
            process_qnode_output(
                raw=raw,
                n_rounds=n_rounds,
                shots=shots,
                mid_measure=mid_measure,
                return_probs=return_probs,
            )
        )

        rows.append(row)

    return rows


def find_duplicate_output_patterns(rows, key):
    """
    Find duplicate output patterns in generated rows.

    Args:
        rows: list[dict]
        key:
            "syndrome_bits"
            "detection_bits"
            "final_bits"
            "probabilities"

    Returns:
        dict:
            pattern -> list of rows
    """

    pattern_to_rows = {}

    for row in rows:
        pattern = row.get(key, "")

        if pattern not in pattern_to_rows:
            pattern_to_rows[pattern] = []

        pattern_to_rows[pattern].append(row)

    return {
        pattern: grouped_rows
        for pattern, grouped_rows in pattern_to_rows.items()
        if len(grouped_rows) > 1
    }