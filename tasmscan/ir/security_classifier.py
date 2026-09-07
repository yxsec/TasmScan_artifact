"""
Security Classification Functions

This module provides functions for classifying IR types based on
security relevance - identifying taint sources and sensitive operations.
"""

from typing import Optional

from .dataflow.taint_registry import REGISTER_TRUST_LEVELS  # noqa: E402
from .ir_types import IRType
from .mappings import OPCODE_TO_IR_TYPE


def classify_taint_source_context(opcode: str, context_info: Optional[str] = None) -> str:
    """Classify a taint source with finer granularity based on opcode and context.

    Distinguishes between data loaded from persistent storage (c4), incoming
    message fields, and other external sources.  This enables downstream
    detectors to suppress false positives on trusted storage reads while still
    flagging untrusted message data.

    Args:
        opcode: The TVM opcode name (e.g. "LDGRAMS", "LDU", "PUSHCTR").
        context_info: Optional hint about the surrounding context.  Recognized
            values include ``"c4"`` / ``"storage"`` (data originates from c4
            persistent storage) and ``"c7"`` / ``"message"`` (data originates
            from an incoming message via c7 context tuple).

    Returns:
        One of:
        - ``"untrusted_message"``  -- data from incoming message body / c7 message fields
        - ``"trusted_storage"``    -- data from c4 persistent storage
        - ``"untrusted_external"`` -- data from external sources (balance, config, etc.)
        - ``"unknown"``            -- cannot determine
    """
    # --- 1. Opcodes that *always* indicate message-derived data ---------------
    message_opcodes = {
        "LDMSGADDR", "LDMSGADDRQ", "INMSG_SRC",  # sender address
    }
    if opcode in message_opcodes:
        return "untrusted_message"

    # --- 2. Context-based disambiguation for generic loaders ------------------
    if context_info is not None:
        ctx = context_info.lower()
        if ctx in ("c4", "storage"):
            return "trusted_storage"
        if ctx in ("c7", "message"):
            return "untrusted_message"
        if ctx in ("balance", "config", "now", "block_lt"):
            return "untrusted_external"

    # --- 3. Register push/pop hints ------------------------------------------
    # PUSHCTR c4 → loading persistent storage cell → trusted
    # PUSHCTR c7 → loading context tuple → mixed (but message fields untrusted)
    if opcode == "PUSHCTR":
        if context_info == "4":
            return "trusted_storage"
        if context_info == "7":
            return "untrusted_external"  # conservative: c7 has mixed trust
        return "unknown"

    # --- 4. Opcodes commonly used on message slices ---------------------------
    message_body_opcodes = {
        "LDGRAMS", "LDVARINT16",       # currency loading
        "LDVARUINT32", "LDVARINT32",    # extended currency
        "LDREF", "LDDICT", "LDDICTS",  # reference/dict loading from body
        "LDOPTREF",                     # optional ref loading
    }
    if opcode in message_body_opcodes:
        return "untrusted_message"

    # --- 5. Conditional loaders (LDU, LDI, etc.) depend on slice origin -------
    conditional_opcodes = {"LDU", "LDI", "LDSLICE", "LDSLICEX"}
    if opcode in conditional_opcodes:
        # Without context_info we cannot distinguish; default to unknown so the
        # caller can fall back to the conservative is_taint_source() check.
        return "unknown"

    return "unknown"


def get_opcode_category(opcode: str) -> str:
    """Get the functional category of an opcode"""
    ir_type = OPCODE_TO_IR_TYPE.get(opcode, IRType.UNKNOWN)

    category_map = {
        "cell_parse": [IRType.LOAD_INT, IRType.LOAD_UINT, IRType.LOAD_BITS, IRType.LOAD_REF,
                       IRType.PRELOAD_INT, IRType.PRELOAD_UINT, IRType.PRELOAD_BITS,
                       IRType.CELL_TO_SLICE, IRType.END_SLICE],
        "cell_build": [IRType.CREATE_CELL_BUILDER, IRType.BUILD_CELL, IRType.STORE_INT,
                       IRType.STORE_UINT, IRType.STORE_BITS, IRType.STORE_REF],
        "crypto": [IRType.HASH_CELL, IRType.HASH_SLICE, IRType.SHA256, IRType.CHECK_SIGNATURE,
                   IRType.HASH_EXT_SHA256, IRType.HASH_EXT_SHA512, IRType.HASH_EXT_BLAKE2B,
                   IRType.HASH_EXT_KECCAK256, IRType.HASH_EXT_KECCAK512],
        "message": [IRType.LOAD_MESSAGE_FLAGS, IRType.LOAD_MESSAGE_SENDER,
                    IRType.LOAD_MESSAGE_VALUE, IRType.LOAD_MESSAGE_BODY,
                    IRType.LOAD_MESSAGE_OPCODE, IRType.CHECK_BOUNCED],
        "gas": [IRType.ACCEPT_GAS, IRType.SET_GAS_LIMIT, IRType.GET_GAS_CONSUMED],
        "action": [IRType.SEND_MESSAGE, IRType.RESERVE_CURRENCY, IRType.SET_CODE],
    }

    for cat, types in category_map.items():
        if ir_type in types:
            return cat

    return "unknown"


def is_taint_source(ir_type: IRType) -> bool:
    """
    Check if an IR type represents a taint source (untrusted data).

    Design Decision (taint strategy): Conservative Taint Strategy
    ====================================================
    This function intentionally marks generic load operations (LOAD_UINT, LOAD_INT,
    LOAD_BITS, LOAD_REF) as taint sources. This is a deliberate CONSERVATIVE approach
    to maximize security coverage at the cost of potential false positives.

    Rationale:
    - In TVM smart contracts, data loaded from slices often originates from external
      messages which are inherently untrusted
    - Without full context-sensitive analysis to determine the slice origin, it is
      safer to assume loaded data could be attacker-controlled
    - False positives are preferable to false negatives in security analysis

    Trade-offs:
    - May flag benign operations where data is loaded from trusted storage
    - Increases noise in analysis results
    - Requires manual review to filter legitimate patterns

    Future Improvements:
    - Context-aware analysis could track slice origins to reduce false positives
    - Whitelist patterns for common safe operations (e.g., loading from c4 storage)

    Args:
        ir_type: The IR type to check

    Returns:
        True if the IR type represents a potential taint source
    """
    taint_sources = {
        # Message data (untrusted) - HIGH confidence taint sources
        IRType.LOAD_MESSAGE_FLAGS,
        IRType.LOAD_MESSAGE_SENDER,
        IRType.LOAD_MESSAGE_VALUE,
        IRType.LOAD_MESSAGE_BODY,
        IRType.LOAD_MESSAGE_OPCODE,
        IRType.LOAD_MSG_ADDRESS,
        IRType.PARSE_MSG_ADDRESS,

        # Generic loads - CONSERVATIVE marking (see docstring for rationale)
        # These are marked as taint sources because the slice origin is unknown
        # and could be from an external message
        IRType.LOAD_UINT,
        IRType.LOAD_INT,
        IRType.LOAD_BITS,
        IRType.LOAD_REF,

        # Currency (can be manipulated by sender)
        IRType.LOAD_GRAMS,
        IRType.LOAD_VARINT,
    }

    return ir_type in taint_sources


def is_sensitive_operation(ir_type: IRType) -> bool:
    """Check if an IR type represents a sensitive operation"""
    sensitive_ops = {
        # Actions that affect blockchain state
        IRType.SEND_MESSAGE,
        IRType.RESERVE_CURRENCY,
        IRType.SET_CODE,
        IRType.SET_LIB_CODE,

        # Storage modifications
        IRType.STORE_STORAGE,
        IRType.POP_CONTINUATION,  # When popping to c4

        # Gas acceptance
        IRType.ACCEPT_GAS,
        IRType.COMMIT_STATE,
    }

    return ir_type in sensitive_ops


__all__ = [
    "REGISTER_TRUST_LEVELS",
    "classify_taint_source_context",
    "get_opcode_category",
    "is_taint_source",
    "is_sensitive_operation",
]
