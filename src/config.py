# src/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Default experiment settings
DEFAULT_N_ROUNDS = 4
DEFAULT_SHOTS = 1

# Default fault setting
DEFAULT_CONTROL = 0
DEFAULT_TARGET = 9
DEFAULT_ERROR_TYPES = ("X", "Z")
DEFAULT_FAULT_ROUND = 0
DEFAULT_INCLUDE_EMPTY = True

# Default simulation situations
DEFAULT_SITUATIONS = [
    {
        "mid_measure": True,
        "reset_after_measure": True,
        "return_probs": False,
        "shots": DEFAULT_SHOTS,
    },
    {
        "mid_measure": True,
        "reset_after_measure": False,
        "return_probs": False,
        "shots": DEFAULT_SHOTS,
    },
    {
        "mid_measure": False,
        "reset_after_measure": False,
        "return_probs": False,
        "shots": DEFAULT_SHOTS,
    },
    {
        "mid_measure": False,
        "reset_after_measure": False,
        "return_probs": True,
        "shots": None,
    },
]