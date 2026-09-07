"""
Comprehensive TVM Opcode to IR Type Mappings

This module provides complete mappings from TVM opcodes to IR types,
covering 1113 opcodes including 912 TVM cp0 specification entries, aliases, and parameterized variants.

The mappings are organized into submodules for maintainability:
- mappings/cell_ops.py: Cell parse and build operations
- mappings/stack_ops.py: Stack manipulation operations
- mappings/control_flow.py: Control flow and exception operations
- mappings/arithmetic.py: Arithmetic, comparison, and bitwise operations
- mappings/dict_ops.py: Dictionary operations
- mappings/misc_ops.py: Crypto, app-level, tuple, const, debug operations

Context-aware mapping is provided in context_mapper.py.
Security classification functions are in security_classifier.py.
"""

# Re-export the aggregated mapping from the mappings package
from .mappings import OPCODE_TO_IR_TYPE

# Re-export context-aware mapping functions
from .context_mapper import (
    get_context_aware_ir_type,
    is_message_flags_load,
    is_message_sender_load,
    is_message_value_load,
    is_message_opcode_load,
)

# Re-export security classification functions
from .security_classifier import (
    get_opcode_category,
    is_taint_source,
    is_sensitive_operation,
)

__all__ = [
    # Main mapping
    "OPCODE_TO_IR_TYPE",
    # Context-aware functions
    "get_context_aware_ir_type",
    "is_message_flags_load",
    "is_message_sender_load",
    "is_message_value_load",
    "is_message_opcode_load",
    # Security classification
    "get_opcode_category",
    "is_taint_source",
    "is_sensitive_operation",
]
