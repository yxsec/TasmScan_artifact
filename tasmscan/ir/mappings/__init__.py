"""
Opcode to IR Type Mappings

This package provides comprehensive mappings from TVM opcodes to IR types,
organized by functional categories for maintainability.

Submodules:
- cell_ops: Cell parse and build operations
- stack_ops: Stack manipulation operations
- control_flow: Control flow and exception operations
- arithmetic: Arithmetic, comparison, and bitwise operations
- dict_ops: Dictionary operations
- misc_ops: Crypto, app-level, tuple, const, debug operations
"""

from .cell_ops import CELL_OPS_MAPPINGS
from .stack_ops import STACK_OPS_MAPPINGS
from .control_flow import CONTROL_FLOW_MAPPINGS
from .arithmetic import ARITHMETIC_MAPPINGS
from .dict_ops import DICT_OPS_MAPPINGS
from .misc_ops import MISC_OPS_MAPPINGS

# Aggregate all mappings into the complete OPCODE_TO_IR_TYPE dictionary
# Order matters for precedence - later dicts override earlier ones
OPCODE_TO_IR_TYPE = {
    **CELL_OPS_MAPPINGS,
    **STACK_OPS_MAPPINGS,
    **CONTROL_FLOW_MAPPINGS,
    **ARITHMETIC_MAPPINGS,
    **DICT_OPS_MAPPINGS,
    **MISC_OPS_MAPPINGS,
}

__all__ = [
    "OPCODE_TO_IR_TYPE",
    "CELL_OPS_MAPPINGS",
    "STACK_OPS_MAPPINGS",
    "CONTROL_FLOW_MAPPINGS",
    "ARITHMETIC_MAPPINGS",
    "DICT_OPS_MAPPINGS",
    "MISC_OPS_MAPPINGS",
]
