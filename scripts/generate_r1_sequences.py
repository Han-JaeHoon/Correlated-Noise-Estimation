"""
Generate R1 sequence dataset (no decoder; pure syndrome accumulation).

Example:
    python scripts/generate_r1_sequences.py \
        --T 50 --n-samples 100 --faulty-cnot-id 8 \
        --p-bg 0.001 --p-high 0.01 --seed 0
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/...` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR
from src.sequence_dataset import generate_r1_dataset, save_r1_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--p-bg", type=float, default=0.001)
    parser.add_argument("--p-high", type=float, default=0.01)
    parser.add_argument(
        "--faulty-cnot-id",
        type=int,
        default=None,
        help="0..23. Pass -1 (or omit) for baseline (no elevated CNOT).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DATA_DIR / "r1_sequences",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Use mid_measure_no_reset mode (default: reset).",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    faulty_cnot_id = args.faulty_cnot_id
    if faulty_cnot_id is not None and faulty_cnot_id < 0:
        faulty_cnot_id = None

    tag = "baseline" if faulty_cnot_id is None else f"cnot{faulty_cnot_id:02d}"
    out_dir = (
        args.out_root
        / tag
        / f"T{args.T}_pbg{args.p_bg}_phigh{args.p_high}_seed{args.seed}"
        / ("no_reset" if args.no_reset else "reset")
    )

    print(
        f"[generate_r1] T={args.T} N={args.n_samples} "
        f"faulty_cnot={faulty_cnot_id} "
        f"p_bg={args.p_bg} p_high={args.p_high} seed={args.seed}"
    )
    print(f"  out: {out_dir}")

    dataset = generate_r1_dataset(
        n_samples=args.n_samples,
        T=args.T,
        p_bg=args.p_bg,
        p_high=args.p_high,
        faulty_cnot_id=faulty_cnot_id,
        seed=args.seed,
        mid_measure=True,
        reset_after_measure=(not args.no_reset),
        progress_every=args.progress_every,
    )

    save_r1_dataset(out_dir, dataset)
    print(f"  saved syndromes: shape {dataset['syndromes'].shape}")


if __name__ == "__main__":
    main()
