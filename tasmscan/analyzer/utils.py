"""Shared utility functions for the analyzer package."""

from collections.abc import Mapping
from typing import Any, Optional


def extract_int_arg(args: Any, arg_idx: int = 0) -> Optional[int]:
    """Extract an integer argument from instruction arguments.

    Handles various argument formats from the disassembler:
    - Direct integer values
    - Objects with .value attribute
    - Mapping objects with 'value' key

    Args:
        args: Instruction arguments (list, mapping, or other)
        arg_idx: Index of the argument to extract

    Returns:
        The integer value, or None if not found/convertible
    """
    if not args:
        return None

    def _coerce_int(candidate: Any) -> Optional[int]:
        if isinstance(candidate, int):
            return candidate
        value = getattr(candidate, "value", None)
        if isinstance(value, int):
            return value
        if isinstance(candidate, Mapping):
            value = candidate.get("value")
            if isinstance(value, int):
                return value
        return None

    if isinstance(args, Mapping):
        if arg_idx in args:
            return _coerce_int(args[arg_idx])
        if arg_idx == 0 and "value" in args:
            return _coerce_int(args.get("value"))
        return None

    try:
        return _coerce_int(args[arg_idx])
    except (IndexError, KeyError, TypeError):
        return None
