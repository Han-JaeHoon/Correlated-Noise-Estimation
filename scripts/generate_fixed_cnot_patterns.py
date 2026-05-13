# scripts/generate_fixed_cnot_patterns.py

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    DEFAULT_N_ROUNDS,
    DEFAULT_CONTROL,
    DEFAULT_TARGET,
    DEFAULT_ERROR_TYPES,
    DEFAULT_INCLUDE_EMPTY,
    DEFAULT_SITUATIONS,
)
from src.dataset_generators import (
    generate_fixed_cnot_round_pattern_rows,
    situation_name,
)
from src.io_utils import save_csv, save_metadata
from src.simulator import make_repeated_stabilizer_qnode


def parse_error_types(error_types_str):
    """
    Example:
        "XZ" -> ("X", "Z")
        "X,Z" -> ("X", "Z")
    """

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
        description="Generate fixed-CNOT round-pattern datasets."
    )

    parser.add_argument(
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Number of stabilizer rounds. Default: {DEFAULT_N_ROUNDS}",
    )

    parser.add_argument(
        "--control",
        type=int,
        default=DEFAULT_CONTROL,
        help=f"CNOT control wire. Default: {DEFAULT_CONTROL}",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"CNOT target wire. Default: {DEFAULT_TARGET}",
    )

    parser.add_argument(
        "--error-types",
        type=str,
        default="".join(DEFAULT_ERROR_TYPES),
        help='Pauli errors, e.g. "XZ" or "X,Z". Default: XZ',
    )

    parser.add_argument(
        "--include-empty",
        action="store_true",
        default=DEFAULT_INCLUDE_EMPTY,
        help="Include no-fault pattern. Default follows config.py.",
    )

    parser.add_argument(
        "--exclude-empty",
        action="store_true",
        help="Exclude no-fault pattern.",
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
    control = args.control
    target = args.target
    error_types = parse_error_types(args.error_types)
    include_empty = args.include_empty and not args.exclude_empty
    output_root = Path(args.output_root)

    situations = select_situations(args.mode)

    for situation in situations:
        shots = situation["shots"]

        qnode = make_repeated_stabilizer_qnode(
            n_rounds=n_rounds,
            shots=shots,
            mid_measure=situation["mid_measure"],
            reset_after_measure=situation["reset_after_measure"],
            return_probs=situation["return_probs"],
        )

        rows = generate_fixed_cnot_round_pattern_rows(
            qnode=qnode,
            n_rounds=n_rounds,
            shots=shots,
            control=control,
            target=target,
            error_types=error_types,
            include_empty=include_empty,
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
            f"fixed_cnot_{control}_{target}_"
            f"{''.join(error_types)}_"
            f"rounds_{n_rounds}_all_round_patterns.csv"
        )

        output_path = output_dir / filename

        save_csv(rows, output_path)

        save_metadata(
            {
                "dataset_type": "fixed_cnot_round_patterns",
                "mode": mode_name,
                "n_rounds": n_rounds,
                "shots": shots,
                "control": control,
                "target": target,
                "error_types": list(error_types),
                "include_empty": include_empty,
                "num_samples": len(rows),
                **situation,
            },
            output_dir
            / f"metadata_fixed_cnot_{control}_{target}_{''.join(error_types)}_rounds_{n_rounds}.json",
        )

        print(f"Saved: {output_path}")
        print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()