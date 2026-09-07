"""
Context-Aware IR Type Mapping

This module provides context-sensitive mapping from opcodes to IR types,
using heuristics based on instruction position and history to infer
semantic meaning of generic load instructions.
"""

from .ir_types import IRType
from .mappings import OPCODE_TO_IR_TYPE


def is_message_flags_load(opcode: str, bits: int, instruction_index: int) -> bool:
    """Check if LDU instruction is loading message flags (4 bits at message start)"""
    return opcode == "LDU" and bits == 4 and instruction_index < 5


def is_message_sender_load(opcode: str, bits: int, prev_opcodes: list) -> bool:
    """Check if LDU instruction is loading sender address (256 bits after flags)"""
    # Check if previous instructions suggest message parsing
    # Typically: LDU 4 (flags) -> ... -> LDU 256 (sender)
    if opcode != "LDU" or bits != 256:
        return False

    # Look for LDU 4 in recent history
    recent_window = prev_opcodes[-10:] if len(prev_opcodes) >= 10 else prev_opcodes
    return any(op == "LDU_4" for op in recent_window)


def is_message_value_load(opcode: str, bits: int, prev_opcodes: list) -> bool:
    """Check if LDU instruction is loading message value (120 bits for grams)"""
    if opcode != "LDU" or bits != 120:
        return False

    # Look for message parsing context
    recent_window = prev_opcodes[-15:] if len(prev_opcodes) >= 15 else prev_opcodes
    return any(op in {"LDU_4", "LDU_256"} for op in recent_window)


def is_message_opcode_load(opcode: str, bits: int, prev_opcodes: list) -> bool:
    """Check if LDU instruction is loading operation code (32 bits)"""
    if opcode != "LDU" or bits != 32:
        return False

    # Look for message parsing context
    recent_window = prev_opcodes[-20:] if len(prev_opcodes) >= 20 else prev_opcodes
    return any(op in {"LDU_4", "LDU_256", "LDMSGADDR"} for op in recent_window)


def get_context_aware_ir_type(
    opcode: str,
    operands: list,
    instruction_index: int,
    instruction_history: list
) -> IRType:
    """
    Get IR type with context-aware mapping for opcodes like LDU/STU.

    This function applies heuristics to map generic load instructions to
    more specific semantic types based on context (e.g., LDU 4 at the start
    of execution likely loads message flags).

    Note on LDMSGADDR:
        LDMSGADDR is mapped directly via OPCODE_TO_IR_TYPE to LOAD_MSG_ADDRESS
        and does not require context-aware handling here. Unlike LDU which has
        multiple semantic interpretations based on bit width and context,
        LDMSGADDR has a single, unambiguous purpose (loading a TON address).

    Args:
        opcode: The opcode mnemonic
        operands: List of operand values
        instruction_index: Current instruction index in the program
        instruction_history: List of previous (opcode, operands) tuples

    Returns:
        IRType enum value
    """
    # Get base mapping
    base_type = OPCODE_TO_IR_TYPE.get(opcode)

    # If it's not LDU/LDI, return base mapping (includes LDMSGADDR -> LOAD_MSG_ADDRESS)
    if opcode not in {"LDU", "LDI"}:
        return base_type if base_type else IRType.RAW_OPCODE

    # Extract bit count from operands
    bits = operands[0] if operands else 0

    # Build history of recent opcodes for context (convert to list for slicing if deque)
    history_list = list(instruction_history)[-20:] if instruction_history else []
    prev_opcodes = [f"{op}_{args[0] if args else ''}" for op, args in history_list]

    # Context-aware mapping for LDU
    if is_message_flags_load(opcode, bits, instruction_index):
        return IRType.LOAD_MESSAGE_FLAGS

    if is_message_sender_load(opcode, bits, prev_opcodes):
        return IRType.LOAD_MESSAGE_SENDER

    if is_message_value_load(opcode, bits, prev_opcodes):
        return IRType.LOAD_MESSAGE_VALUE

    if is_message_opcode_load(opcode, bits, prev_opcodes):
        return IRType.LOAD_MESSAGE_OPCODE

    # Default to base type
    return base_type if base_type else IRType.RAW_OPCODE


__all__ = [
    "get_context_aware_ir_type",
    "is_message_flags_load",
    "is_message_sender_load",
    "is_message_value_load",
    "is_message_opcode_load",
]
