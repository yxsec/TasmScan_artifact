"""
Taint source configuration for TVM data flow analysis.

Contains opcode classifications for:
- Continuation-related opcodes
- Multi-output taint rules
- Taint sources (unconditional and conditional)
- Guard/validation opcodes
- Sensitive operation opcodes
- Trust level classification for taint sources
"""
from typing import Dict, List, Set

from ...config import CORE_SENSITIVE_OPCODES
from .types import ValueSource


# Continuation-related opcodes where taint propagation is conservative
# Note: CALL/CALLX are not standalone TVM opcodes; they are fift aliases
# The disassembler produces specific variants like CALLXARGS, CALLCC, etc.
CONTINUATION_OPCODES: Set[str] = {
    "PUSHCONT",
    "CALLREF",
    "CALLDICT",
    "CALLDICT_LONG",
    "CALLXARGS",
    "CALLXARGS_VAR",
    "CALLXVARARGS",
    "CALLCC",
    "CALLCCARGS",
    "CALLCCVARARGS",
    "EXECUTE",
    "JMPX",
    "JMPXARGS",
    "JMPXVARARGS",
    "JMPXDATA",
    "JMPDICT",
    "JMPREF",
    "JMPREFDATA",
}

# Format: opcode -> list of taint rules for each output
# Output index 0 is pushed first (ends up deeper), last output is TOS
# For LD* instructions: s - x s' means output[0]=x (loaded value), output[1]=s' (remainder)
# After pushing in order: s' is at TOS (index 0), x is at index 1
#
# Rules:
#   - "inherit_first_input": Taint from first input (the slice being read from)
#   - "preserve_slice": The remaining slice preserves original slice's taint and metadata
#   - "inherit_all": Tainted if any input is tainted (default behavior)
#   - "no_taint": Output is never tainted
MULTI_OUTPUT_TAINT_RULES: Dict[str, List[str]] = {
    # Currency/amount loading - value is tainted from slice, remainder preserves slice taint
    # Note: LDVARUINT16 is a fift alias for LDGRAMS; the disassembler produces LDGRAMS
    "LDGRAMS": ["inherit_first_input", "preserve_slice"],
    "LDVARINT16": ["inherit_first_input", "preserve_slice"],
    # Slice loading - both outputs derive from input slice
    "LDSLICE": ["inherit_first_input", "preserve_slice"],
    "LDSLICEX": ["inherit_first_input", "preserve_slice"],
    "PLDSLICE": ["inherit_first_input", "preserve_slice"],
    "PLDSLICEX": ["inherit_first_input", "preserve_slice"],
    # Reference loading - cell from slice, slice remainder
    "LDREF": ["inherit_first_input", "preserve_slice"],
    "LDREFRTOS": ["inherit_first_input", "preserve_slice"],
    # Integer loading
    "LDU": ["inherit_first_input", "preserve_slice"],
    "LDI": ["inherit_first_input", "preserve_slice"],
    "LDUX": ["inherit_first_input", "preserve_slice"],
    "LDIX": ["inherit_first_input", "preserve_slice"],
    # Address loading
    "LDMSGADDR": ["inherit_first_input", "preserve_slice"],
    "LDMSGADDRQ": ["inherit_first_input", "preserve_slice"],
    # Note: TVM has no LDBIT/LDBITS opcodes; bit loading is done via LDU 1
    "LDVARINT32": ["inherit_first_input", "preserve_slice"],
    "LDVARUINT32": ["inherit_first_input", "preserve_slice"],
    "LDOPTREF": ["inherit_first_input", "preserve_slice"],
    "LDDICT": ["inherit_first_input", "preserve_slice"],
    "LDDICTS": ["inherit_first_input", "preserve_slice"],
    # Preload variants (single output, but for consistency)
    "PLDU": ["inherit_first_input"],
    "PLDI": ["inherit_first_input"],
}

# Opcodes that always introduce tainted (untrusted) values
TAINT_SOURCES: Dict[str, ValueSource] = {
    # Sender address loading
    # Note: SENDER and PUSHMSGADDR do not exist in TVM spec; removed
    "LDMSGADDR": ValueSource.MESSAGE_SENDER,
    "LDMSGADDRQ": ValueSource.MESSAGE_SENDER,
    "INMSG_SRC": ValueSource.MESSAGE_SENDER,
    # Value/currency loading
    # Note: LDVARUINT16 is a fift alias for LDGRAMS; the disassembler produces LDGRAMS
    "LDGRAMS": ValueSource.MESSAGE_VALUE,
    "LDVARINT16": ValueSource.MESSAGE_VALUE,
    "LDVARUINT32": ValueSource.MESSAGE_VALUE,
    "LDVARINT32": ValueSource.MESSAGE_VALUE,
    # Message body/data loading
    "LDREF": ValueSource.MESSAGE_BODY,
    "LDDICT": ValueSource.MESSAGE_BODY,
    "LDDICTS": ValueSource.MESSAGE_BODY,
    # LDOPTREF is a fift alias for LDDICT (loads Maybe ^Cell)
    "LDOPTREF": ValueSource.MESSAGE_BODY,
    # Note: TVM has no LDBIT/LDBITS opcodes; bit loading is done via LDU 1
}

# Context-dependent taint sources - only tainted if loading from message slice
CONDITIONAL_TAINT_SOURCES: Dict[str, ValueSource] = {
    "LDU": ValueSource.MESSAGE_BODY,
    "LDI": ValueSource.MESSAGE_BODY,
    "LDSLICE": ValueSource.MESSAGE_BODY,
    "LDSLICEX": ValueSource.MESSAGE_BODY,
}

# Opcodes that load message slices (indicate message context)
MESSAGE_SLICE_LOADERS: Set[str] = {
    "LDSLICE", "LDSLICEX", "PLDSLICE", "PLDSLICEX",
    # Note: CTOS removed - handled separately (Low #26)
}

# Low #26: CTOS handled separately with context checking
# CTOS converts Cell to Slice, but Cell source matters:
# - Message Cell -> taint
# - Storage Cell -> no taint
# - Code Cell -> no taint
CONTEXT_DEPENDENT_LOADERS: Set[str] = {
    "CTOS",
}

# Opcodes that use values in sensitive operations
# Derived from CORE_SENSITIVE_OPCODES (config.py) - single source of truth
SENSITIVE_OPCODES: Set[str] = set(CORE_SENSITIVE_OPCODES)

# Dynamic call opcodes that take call target from stack top
# These are security-sensitive because user-controlled targets enable arbitrary code execution
DYNAMIC_CALL_OPCODES: Set[str] = {
    "CALLX",        # Call continuation from stack
    "CALLXARGS",    # Call with argument count
    "CALLXARGS_1",  # Short form of CALLXARGS with fixed p=1
    "CALLXVARARGS", # Call with variable arguments
    "CALLXARGS_VAR",# Call with args on stack
    "CALLCC",       # Call with current continuation saved in c0
    "CALLCCARGS",   # CALLCC with argument count
    "CALLCCARGS_VAR",  # CALLCC args count variant alias
    "CALLCCVARARGS",   # CALLCC with varargs count on stack
    "EXECUTE",      # Execute continuation
    "BOOLEVAL",     # Evaluate boolean continuation from stack
    "JMP",          # Jump to continuation from stack (alias form)
    "JMPX",         # Jump to continuation from stack
    "JMPXARGS",     # Jump with argument count
    "JMPXVARARGS",  # Jump with variable arguments
    "JMPXDATA",     # Jump with data
}

# Call opcodes with varargs count on stack (p, r are on stack)
CALL_VARARGS_OPCODES: Set[str] = {
    "CALLXVARARGS",
    "CALLCCVARARGS",
}

# Continuation push opcodes (code, not data)
CONTINUATION_PUSH_OPCODES: Set[str] = {
    "PUSHCONT",
    "PUSHREFCONT",
}

# Dynamic stack shuffles that reorder unknown elements at runtime
DYNAMIC_SHUFFLE_OPCODES: Set[str] = {
    "ROLL", "ROLLX", "-ROLL", "-ROLLX",
    "BLKSWAP", "BLKSWX",
    "REVX", "ROTR",
}

# =============================================================================
# Trust level classification for taint sources
# =============================================================================

# Control register trust classification
# Maps register index to trust level string for use in taint context tracking.
# c4 = persistent storage (written by contract itself, generally trusted)
# c5 = output actions (write-only, not a taint source)
# c7 = context tuple (contains both trusted blockchain data and untrusted message fields)
REGISTER_TRUST_LEVELS: Dict[int, str] = {
    4: "trusted_storage",     # c4: persistent data, written by contract
    5: "write_only",          # c5: output actions register
    7: "mixed_trust",         # c7: context tuple (blockchain params + message data)
}

# Per-opcode trust level for taint sources.
# Complements TAINT_SOURCES (which maps opcode -> ValueSource) by adding a
# trust dimension so downstream analysis can distinguish e.g. a value loaded
# from c4 storage (trusted) vs. an incoming message field (untrusted).
#
# Trust levels:
#   "untrusted_message"  - data from incoming message body / sender
#   "trusted_storage"    - data from c4 persistent storage
#   "untrusted_external" - data from external/blockchain sources
#   "unknown"            - cannot determine without additional context
TAINT_SOURCE_TRUST_LEVELS: Dict[str, str] = {
    # Sender address loading — always untrusted message data
    "LDMSGADDR": "untrusted_message",
    "LDMSGADDRQ": "untrusted_message",
    "INMSG_SRC": "untrusted_message",
    # Currency/value loading — from message, untrusted
    "LDGRAMS": "untrusted_message",
    "LDVARINT16": "untrusted_message",
    "LDVARUINT32": "untrusted_message",
    "LDVARINT32": "untrusted_message",
    # Message body/data loading — untrusted
    "LDREF": "untrusted_message",
    "LDDICT": "untrusted_message",
    "LDDICTS": "untrusted_message",
    "LDOPTREF": "untrusted_message",
    # Conditional loaders — trust depends on slice origin (unknown without context)
    "LDU": "unknown",
    "LDI": "unknown",
    "LDSLICE": "unknown",
    "LDSLICEX": "unknown",
}


def get_taint_trust_level(opcode: str, register_index: int = -1) -> str:
    """Get the trust level for a taint source opcode.

    Checks opcode-specific trust levels first, then falls back to register
    trust levels if the opcode is a register access (PUSHCTR/POPCTR).

    Args:
        opcode: The TVM opcode name.
        register_index: Control register index (0-7) if the opcode accesses
            a control register, otherwise -1.

    Returns:
        Trust level string: ``"untrusted_message"``, ``"trusted_storage"``,
        ``"untrusted_external"``, ``"write_only"``, ``"mixed_trust"``,
        or ``"unknown"``.
    """
    # Check opcode-level trust first
    if opcode in TAINT_SOURCE_TRUST_LEVELS:
        return TAINT_SOURCE_TRUST_LEVELS[opcode]

    # For register access opcodes, use register trust levels
    if opcode in ("PUSHCTR", "POPCTR", "PUSH", "POP") and register_index >= 0:
        return REGISTER_TRUST_LEVELS.get(register_index, "unknown")

    return "unknown"
