"""Stim-based realistic rotated surface code backend (Phase 2)."""

from .surface_code import RotatedSurfaceCode, build
from .decoding import MatchingDecoder, logical_error_rate

__all__ = ["RotatedSurfaceCode", "build", "MatchingDecoder", "logical_error_rate"]
