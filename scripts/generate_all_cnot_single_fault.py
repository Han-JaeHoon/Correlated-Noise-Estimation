# scripts/generate_all_cnot_single_fault.py

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    DEFAULT_N_ROUNDS,
    DEFAULT_ERROR_TYPES,
    DEFAULT_FAULT_ROUND,
    DEFAULT_SITUATIONS,
)
from src.dataset_generators import (
    generate_all_cnot_single_fault_rows,
    situation_name,
)
from src.io_utils import save_csv, save_metadata
from src.simulator import make_repeated_stabilizer_qnode
from src.surface_code_layout import enumerate_stabilizer_cnots


def parse_error_types(error_types_str):
    if "," in error_types_str:
        parts = error_types_str.split(",")
    else:
        parts = list(error_types_str)

    parts = tuple(p.strip().upper() for p in parts if p.strip())

    if len(parts) != 2:
        raise ValueError(
            f"error_types must contain two Pauli types. Got: {error_types_str}"
        )

    return parts


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate all-CNOT single-fault datasets."
    )

    parser.add_argument(
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Number of stabilizer rounds. Default: {DEFAULT_N_ROUNDS}",
    )

    parser.add_argument(
        "--fault-round",
        type=int,
        default=DEFAULT_FAULT_ROUND,
        help=f"Round where the single fault is inserted. Default: {DEFAULT_FAULT_ROUND}",
    )

    parser.add_argument(
        "--error-types",
        type=str,
        default="".join(DEFAULT_ERROR_TYPES),
        help='Pauli errors, e.g. "XZ" or "X,Z". Default: XZ',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "mid_reset",
            "mid_no_reset",
            "no_mid_sample",
            "no_mid_probs",
        ],
        help="Which simulation mode to run. Default: all",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DATA_DIR),
        help=f"Output root directory. Default: {DATA_DIR}",
    )

    return parser


def select_situations(mode):
    if mode == "all":
        return DEFAULT_SITUATIONS

    mapping = {
        "mid_reset": {
            "mid_measure": True,
            "reset_after_measure": True,
            "return_probs": False,
            "shots": 1,
        },
        "mid_no_reset": {
            "mid_measure": True,
            "reset_after_measure": False,
            "return_probs": False,
            "shots": 1,
        },
        "no_mid_sample": {
            "mid_measure": False,
            "reset_after_measure": False,
            "return_probs": False,
            "shots": 1,
        },
        "no_mid_probs": {
            "mid_measure": False,
            "reset_after_measure": False,
            "return_probs": True,
            "shots": None,
        },
    }

    return [mapping[mode]]


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    n_rounds = args.n_rounds
    fault_round = args.fault_round
    error_types = parse_error_types(args.error_types)
    output_root = Path(args.output_root)

    situations = select_situations(args.mode)
    n_cnot_locations = len(enumerate_stabilizer_cnots())

    for situation in situations:
        shots = situation["shots"]

        qnode = make_repeated_stabilizer_qnode(
            n_rounds=n_rounds,
            shots=shots,
            mid_measure=situation["mid_measure"],
            reset_after_measure=situation["reset_after_measure"],
            return_probs=situation["return_probs"],
        )

        rows = generate_all_cnot_single_fault_rows(
            qnode=qnode,
            n_rounds=n_rounds,
            shots=shots,
            fault_round=fault_round,
            error_types=error_types,
            mid_measure=situation["mid_measure"],
            reset_after_measure=situation["reset_after_measure"],
            return_probs=situation["return_probs"],
        )

        mode_name = situation_name(
            mid_measure=situation["mid_measure"],
            reset_after_measure=situation["reset_after_measure"],
            return_probs=situation["return_probs"],
        )

        output_dir = output_root / mode_name

        filename = (
            f"all_cnots_single_fault_"
            f"fault_round_{fault_round}_"
            f"{''.join(error_types)}_"
            f"rounds_{n_rounds}.csv"
        )

        output_path = output_dir / filename

        save_csv(rows, output_path)

        save_metadata(
            {
                "dataset_type": "all_cnot_single_fault",
                "mode": mode_name,
                "n_rounds": n_rounds,
                "shots": shots,
                "fault_round": fault_round,
                "error_types": list(error_types),
                "num_cnot_locations": n_cnot_locations,
                "num_samples": len(rows),
                **situation,
            },
            output_dir
            / f"metadata_all_cnots_fault_round_{fault_round}_{''.join(error_types)}_rounds_{n_rounds}.json",
        )

        print(f"Saved: {output_path}")
        print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()