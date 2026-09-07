"""
Constants and type definitions for TVM program analysis.

This module centralizes all opcode sets, event mappings, and configuration constants
used throughout the analyzer package.
"""
from typing import Any, Dict, Set, TYPE_CHECKING

from ..config import GUARD_OPCODES

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from pytoniq_core import Cell
else:
    Cell = Any  # type: ignore[misc,assignment]

# Using Any because continuation structures are dynamically typed
# and vary based on opcode type and disassembler version
ContinuationDict = Dict[str, Any]

# Used throughout the analyzer to identify the main code context vs continuations
MAIN_CONTEXT = "main"

# Event mapping: Maps semantic event types to sets of opcodes that trigger them.
# This mapping is used by detectors to identify security-relevant operations.
#
# Event types:
# - "accept": Gas acceptance operations (must occur before message processing)
# - "send": Message sending operations (security-sensitive)
# - "raw_reserve": Currency reservation operations
# - "sender_read": Operations that read message sender address
# - "guard": Control flow operations used for validation checks
#
# Note: This mapping is intentionally defined as a module-level constant rather than
# being loaded from external config, because:
# 1. The opcode set is tied to TVM semantics and changes rarely
# 2. Misconfiguration could cause security detectors to miss vulnerabilities
# 3. Detectors can override via ProgramAnalyzer.event_map if needed for testing
DEFAULT_EVENT_MAP: Dict[str, Set[str]] = {
    # Note: Only include opcodes that actually exist in TVM spec (cp0.json)
    # ACCEPTQ, SETGASLIMITVAR, SENDRAWMSGIMM, SENDER, LOADMSGADDR, PUSHMSGADDR do not exist
    "accept": {"ACCEPT", "SETGASLIMIT"},
    "send": {"SENDRAWMSG", "SENDMSG"},
    "raw_reserve": {"RAWRESERVE"},
    "sender_read": {"LDMSGADDR", "LDMSGADDRQ", "INMSG_SRC"},
    # Derived from config.GUARD_OPCODES (single source of truth)
    "guard": set(GUARD_OPCODES),
}

# Dictionary-dispatch opcodes that may transfer control to a continuation
# loaded from a dictionary on success.
DICT_DISPATCH_JUMP_OPCODES = {
    "DICTIGETJMP", "DICTUGETJMP",
    "DICTIGETJMPZ", "DICTUGETJMPZ",
    # Prefix dictionary jump/switch dispatch
    "PFXDICTGETJMP", "PFXDICTSWITCH",
}

DICT_DISPATCH_EXEC_OPCODES = {
    "DICTIGETEXEC", "DICTUGETEXEC",
    "DICTIGETEXECZ", "DICTUGETEXECZ",
    "PFXDICTGETEXEC",
}

DICT_DISPATCH_OPCODES = DICT_DISPATCH_JUMP_OPCODES | DICT_DISPATCH_EXEC_OPCODES

CONDITIONAL_BRANCHES = {
    "IF", "IFNOT", "IFJMP", "IFNOTJMP", "IFELSE",
    "IFREF", "IFNOTREF", "IFJMPREF", "IFNOTJMPREF",
    "IFREFELSE", "IFELSEREF", "IFREFELSEREF",
    "IFRET", "IFNOTRET", "IFRETALT", "IFNOTRETALT",
    "IFBITJMP", "IFNBITJMP", "IFBITJMPREF", "IFNBITJMPREF",
} | DICT_DISPATCH_OPCODES

# Conditional guards that may terminate but still fall through when false
CONDITIONAL_GUARDS = {
    "THROWIF", "THROWIFNOT", "THROWIF_SHORT", "THROWIFNOT_SHORT",
    "THROWANYIF", "THROWANYIFNOT",
    "THROWARGIF", "THROWARGIFNOT",
    "THROWARGANYIF", "THROWARGANYIFNOT",
}

# Conditional return instructions (terminate when condition is true)
# These are also in CONDITIONAL_BRANCHES for continuation mapping purposes
CONDITIONAL_RETURNS = {
    "IFRET", "IFNOTRET", "IFRETALT", "IFNOTRETALT",
}

# Stack analysis confidence thresholds
# These values were empirically tuned based on real-world TVM contract analysis.

# Branch opcodes that consume continuations
BRANCH_CONT_ARITY = {
    "IF": 1,
    "IFNOT": 1,
    "IFJMP": 1,
    "IFNOTJMP": 1,
    "IFREF": 1,
    "IFNOTREF": 1,
    "IFJMPREF": 1,
    "IFNOTJMPREF": 1,
    "IFRET": 0,
    "IFNOTRET": 0,
    "IFRETALT": 0,
    "IFNOTRETALT": 0,
    "IFELSE": 2,
    "IFREFELSE": 2,
    "IFELSEREF": 2,
    "IFREFELSEREF": 2,
    "IFBITJMP": 1,
    "IFNBITJMP": 1,
    "IFBITJMPREF": 1,
    "IFNBITJMPREF": 1,
    "JMPREF": 1,
    "JMPX": 1,
    "JMPXDATA": 1,
    "JMPDICT": 1,
    "JMPREFDATA": 1,
    "JMPXARGS": 1,
    "JMPXVARARGS": 1,
    "EXECUTE": 1,
    "CALLDICT": 1,
    "CALLDICT_LONG": 1,
    "CALLXARGS": 1,
    "CALLXVARARGS": 1,
    "CALLCC": 1,
    "CALLCCARGS": 1,
    "CALLCCVARARGS": 1,
    "CALLREF": 1,
    # TRY/TRYARGS consume two stack continuations (body + handler), but only body
    # is the direct control-transfer target.
    "TRY": 1,
    "TRYARGS": 1,
    # Loop opcodes that consume continuations (loop body / condition)
    "REPEAT": 1,
    "UNTIL": 1,
    "WHILE": 2,
    "AGAIN": 1,
    "REPEATBRK": 1,
    "UNTILBRK": 1,
    "WHILEBRK": 2,
    "AGAINBRK": 1,
    # *END loop variants consume body/condition continuation from cc.
    # WHILEEND keeps 1 stack continuation (cond), others consume none.
    "REPEATEND": 0,
    "REPEATENDBRK": 0,
    "UNTILEND": 0,
    "UNTILENDBRK": 0,
    "WHILEEND": 1,
    "WHILEENDBRK": 1,
    "AGAINEND": 0,
    "AGAINENDBRK": 0,
    # BOOLEVAL executes continuation and returns to caller via wrapped c0/c1.
    "BOOLEVAL": 1,
    # Short CALLXARGS alias with fixed p=1 in instruction table.
    "CALLXARGS_1": 1,
}

# Conditional operations that consume continuations from stack and may return to caller.
# Note: IFRET/IFNOTRET are not included as they have different semantics -
# they don't consume continuations but directly return based on condition.
RETURNING_CONDITIONALS = {
    "IF", "IFNOT", "IFELSE",
    "IFREF", "IFNOTREF",
    "IFREFELSE", "IFELSEREF", "IFREFELSEREF",
}

UNCONDITIONAL_BRANCHES = {
    "JMPREF", "JMPX", "JMPXDATA", "JMPDICT", "JMPREFDATA",
    "JMPXARGS", "JMPXVARARGS",
}

# Note: Only include opcodes that actually exist in TVM spec (cp0.json)
# RETALTBOOL, RETALTIF, RETALTIFNOT, RETIF, RETIFNOT, RETURN, END do not exist
# Conditional returns (IFRET, IFNOTRET, IFRETALT, IFNOTRETALT) are NOT terminators
# because they only return when condition is true - otherwise execution continues
TERMINATORS = {
    # Return instructions (unconditional)
    "RET", "RETALT", "RETBOOL", "RETARGS", "RETVARARGS", "RETDATA",
    # Throw instructions (always terminate control flow)
    "THROW", "THROW_SHORT", "THROWARG", "THROWARGANY", "THROWANY",
    # Loop *END variants consume remainder of code via extract_cc(0)
    # and do not fall through in parent context.
    "REPEATEND", "REPEATENDBRK",
    "UNTILEND", "UNTILENDBRK",
    "WHILEEND", "WHILEENDBRK",
    "AGAINEND", "AGAINENDBRK",
} | UNCONDITIONAL_BRANCHES

CALL_OPCODES = {
    "CALLDICT", "CALLDICT_LONG", "CALLREF",
    "CALLXARGS", "CALLXVARARGS",
    "CALLCC", "CALLCCARGS", "CALLCCVARARGS",
    "EXECUTE", "TRY", "TRYARGS",
    "BOOLEVAL",
    "CALLXARGS_1",
}

# Continuation call opcodes that consume a continuation and return to caller.
# Note: Some CALL* opcodes (CALLREF/CALLDICT) are resolved dynamically and may not map to a known continuation ID.
CALL_CONT_OPCODES = {
    "CALLREF", "CALLDICT", "CALLDICT_LONG",
    "CALLXARGS", "CALLXVARARGS",
    "CALLCC", "CALLCCARGS", "CALLCCVARARGS",
    "EXECUTE", "TRY", "TRYARGS",
    "BOOLEVAL",
    "CALLXARGS_1",
}

# Loop opcodes that consume continuation(s) for loop body/condition.
LOOP_CALL_OPCODES = {
    "REPEAT", "UNTIL", "WHILE", "AGAIN",
    "REPEATBRK", "UNTILBRK", "WHILEBRK", "AGAINBRK",
    "REPEATEND", "REPEATENDBRK",
    "UNTILEND", "UNTILENDBRK",
    "WHILEEND", "WHILEENDBRK",
    "AGAINEND", "AGAINENDBRK",
}
CALL_CONT_OPCODES = CALL_CONT_OPCODES | LOOP_CALL_OPCODES

# Loop variants that capture the remaining code stream via extract_cc(0).
END_LOOP_OPCODES = {
    "REPEATEND", "REPEATENDBRK",
    "UNTILEND", "UNTILENDBRK",
    "WHILEEND", "WHILEENDBRK",
    "AGAINEND", "AGAINENDBRK",
}

# Opcodes with no normal loop exit.
# They may still exit via RETALT / explicit jump / exception.
INFINITE_LOOP_OPCODES = {"AGAIN", "AGAINBRK", "AGAINEND", "AGAINENDBRK"}

# Opcodes where the continuation is below p arguments on the stack.
# For *_VARARGS variants, the argument count is taken from the stack (unknown here).
CALL_ARGS_IMM_OPCODES = {
    "CALLXARGS",
    "CALLCCARGS",
    "JMPXARGS",
}
CALL_ARGS_VAR_OPCODES = {
    "CALLXVARARGS", "CALLCCVARARGS", "JMPXVARARGS",
}
CALL_ARGS_SUFFIX_PREFIXES = ("CALLXARGS_", "CALLCCARGS_")

# Control-transfer opcodes that take continuation from a control register (not the stack).
# CALLDICT/JMPDICT use c3 (method dictionary continuation).
CALL_CTRL_REG_OPCODES = {"CALLDICT", "CALLDICT_LONG", "JMPDICT"}
CALL_CTRL_REG_INDEX = {
    "CALLDICT": 3,
    "CALLDICT_LONG": 3,
    "JMPDICT": 3,
}

# Call opcodes that do not consume a continuation from the stack unless inline.
CALL_NO_STACK_OPCODES = {"CALLREF"}
