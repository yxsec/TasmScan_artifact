"""
Opcode to InstructionKind Mapper

Maps IRType enum values to the InstructionKind classification.
This adapter bridges the flat IRType system (used in opcode_mappings)
with the hierarchical InstructionKind classification (used in TASIR).
"""

from typing import Dict, Optional
from .ir_types import IRType
from .tasir_types import InstructionKind


# Complete mapping: IRType -> InstructionKind
IRTYPE_TO_KIND: Dict[IRType, InstructionKind] = {
    # ===== Cell Parse Operations =====
    IRType.LOAD_INT: InstructionKind.CELL_LOAD,
    IRType.LOAD_UINT: InstructionKind.CELL_LOAD,
    IRType.LOAD_BITS: InstructionKind.CELL_LOAD,
    IRType.LOAD_REF: InstructionKind.CELL_LOAD,
    IRType.PRELOAD_INT: InstructionKind.CELL_LOAD,
    IRType.PRELOAD_UINT: InstructionKind.CELL_LOAD,
    IRType.PRELOAD_BITS: InstructionKind.CELL_LOAD,
    IRType.PRELOAD_REF: InstructionKind.CELL_LOAD,
    IRType.CELL_TO_SLICE: InstructionKind.CELL_CONVERT,
    IRType.END_SLICE: InstructionKind.CELL_LOAD,  # Slice consumption check

    # ===== Cell Build Operations =====
    IRType.CREATE_CELL_BUILDER: InstructionKind.CELL_CONVERT,
    IRType.BUILD_CELL: InstructionKind.CELL_CONVERT,
    IRType.STORE_INT: InstructionKind.CELL_STORE,
    IRType.STORE_UINT: InstructionKind.CELL_STORE,
    IRType.STORE_BITS: InstructionKind.CELL_STORE,
    IRType.STORE_REF: InstructionKind.CELL_STORE,

    # ===== Message Operations (High-level abstractions) =====
    IRType.LOAD_MESSAGE_FLAGS: InstructionKind.CELL_LOAD,
    IRType.LOAD_MESSAGE_SENDER: InstructionKind.CELL_LOAD,
    IRType.LOAD_MESSAGE_VALUE: InstructionKind.CELL_LOAD,
    IRType.LOAD_MESSAGE_BODY: InstructionKind.CELL_LOAD,
    IRType.LOAD_MESSAGE_OPCODE: InstructionKind.CELL_LOAD,
    IRType.CHECK_BOUNCED: InstructionKind.COMPARISON,

    # ===== Cryptographic Operations =====
    IRType.HASH_CELL: InstructionKind.ARITHMETIC,  # Hash is a form of computation
    IRType.HASH_SLICE: InstructionKind.ARITHMETIC,
    IRType.SHA256: InstructionKind.ARITHMETIC,
    IRType.CHECK_SIGNATURE: InstructionKind.COMPARISON,  # Returns bool
    IRType.HASH_EXT_SHA256: InstructionKind.ARITHMETIC,
    IRType.HASH_EXT_SHA512: InstructionKind.ARITHMETIC,
    IRType.HASH_EXT_BLAKE2B: InstructionKind.ARITHMETIC,
    IRType.HASH_EXT_KECCAK256: InstructionKind.ARITHMETIC,
    IRType.HASH_EXT_KECCAK512: InstructionKind.ARITHMETIC,

    # ===== Stack Operations =====
    IRType.PUSH_VALUE: InstructionKind.STACK_PUSH,
    IRType.POP_VALUE: InstructionKind.STACK_POP,
    IRType.DUP_VALUE: InstructionKind.STACK_PUSH,
    IRType.STACK_EXCHANGE: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_COMPLEX: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_ROTATE: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_SWAP: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_BLOCK_SWAP: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_REVERSE: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_PICK: InstructionKind.STACK_PUSH,
    IRType.STACK_ROLL: InstructionKind.STACK_SHUFFLE,

    # ===== Control Flow - Basic =====
    IRType.CALL: InstructionKind.CONT_CALL,
    IRType.CALL_INDIRECT: InstructionKind.CONT_CALL,
    IRType.RETURN: InstructionKind.CONT_RETURN,
    IRType.UNCONDITIONAL_JUMP: InstructionKind.CONT_JUMP,

    # ===== Control Flow - Conditional =====
    IRType.CONDITIONAL_BRANCH: InstructionKind.BRANCH_CONDITIONAL,
    IRType.CONDITIONAL_JUMP: InstructionKind.BRANCH_CONDITIONAL,
    IRType.CONDITIONAL_RETURN: InstructionKind.CONT_RETURN,
    IRType.SELECT_CONDITIONAL: InstructionKind.BRANCH_CONDITIONAL,
    IRType.BIT_CONDITIONAL_JUMP: InstructionKind.BRANCH_CONDITIONAL,

    # ===== Control Flow - Loops =====
    IRType.LOOP_REPEAT: InstructionKind.LOOP,
    IRType.LOOP_UNTIL: InstructionKind.LOOP,
    IRType.LOOP_WHILE: InstructionKind.LOOP,
    IRType.LOOP_AGAIN: InstructionKind.LOOP,

    # ===== Control Flow - Registers =====
    IRType.PUSH_CONTINUATION: InstructionKind.CONT_CREATE,
    IRType.POP_CONTINUATION: InstructionKind.REGISTER_STORE,
    IRType.SET_CONTINUATION: InstructionKind.REGISTER_STORE,
    IRType.SAVE_CONTINUATION: InstructionKind.CONT_SAVE,

    # ===== Storage Operations =====
    IRType.LOAD_STORAGE: InstructionKind.REGISTER_LOAD,
    IRType.STORE_STORAGE: InstructionKind.REGISTER_STORE,

    # ===== Exception Handling =====
    IRType.THROW: InstructionKind.THROW,
    IRType.THROW_IF: InstructionKind.THROW,
    IRType.THROW_IF_NOT: InstructionKind.THROW,
    IRType.THROW_ANY: InstructionKind.THROW,
    IRType.TRY_CATCH: InstructionKind.TRY_CATCH,

    # ===== Arithmetic Operations =====
    IRType.ARITHMETIC: InstructionKind.ARITHMETIC,
    IRType.ARITHMETIC_DIV: InstructionKind.ARITHMETIC,
    IRType.ARITHMETIC_SHIFT: InstructionKind.ARITHMETIC,
    IRType.ARITHMETIC_NEGATE: InstructionKind.ARITHMETIC,
    IRType.ARITHMETIC_QUIET: InstructionKind.ARITHMETIC,

    # ===== Comparison Operations =====
    IRType.COMPARISON: InstructionKind.COMPARISON,
    IRType.COMPARISON_SGN: InstructionKind.COMPARISON,

    # ===== Logical/Bitwise Operations =====
    IRType.BITWISE: InstructionKind.BITWISE,
    IRType.BITWISE_SHIFT: InstructionKind.BITWISE,

    # ===== Slice Manipulation Operations =====
    IRType.SLICE_CUT_FIRST: InstructionKind.CELL_LOAD,
    IRType.SLICE_SKIP_FIRST: InstructionKind.CELL_LOAD,
    IRType.SLICE_CUT_LAST: InstructionKind.CELL_LOAD,
    IRType.SLICE_SKIP_LAST: InstructionKind.CELL_LOAD,
    IRType.SLICE_SPLIT: InstructionKind.CELL_LOAD,
    IRType.SLICE_INFO: InstructionKind.CELL_LOAD,
    IRType.CELL_INFO: InstructionKind.CELL_LOAD,

    # ===== Builder Info Operations =====
    IRType.BUILDER_INFO: InstructionKind.CELL_STORE,
    IRType.BUILDER_REMAINING: InstructionKind.CELL_STORE,

    # ===== Little-Endian Integer Operations =====
    IRType.LOAD_INT_LE: InstructionKind.CELL_LOAD,
    IRType.LOAD_UINT_LE: InstructionKind.CELL_LOAD,
    IRType.PRELOAD_INT_LE: InstructionKind.CELL_LOAD,
    IRType.PRELOAD_UINT_LE: InstructionKind.CELL_LOAD,
    IRType.STORE_INT_LE: InstructionKind.CELL_STORE,
    IRType.STORE_UINT_LE: InstructionKind.CELL_STORE,

    # ===== Optional Reference Operations =====
    IRType.LOAD_OPT_REF: InstructionKind.CELL_LOAD,
    IRType.STORE_OPT_REF: InstructionKind.CELL_STORE,

    # ===== Tuple Operations =====
    IRType.TUPLE_CREATE: InstructionKind.TUPLE_CREATE,
    IRType.TUPLE_INDEX: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_SET: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_LENGTH: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_OPERATIONS: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_NULL: InstructionKind.TUPLE_CREATE,
    IRType.TUPLE_IS_NULL: InstructionKind.COMPARISON,
    IRType.TUPLE_IS_TUPLE: InstructionKind.COMPARISON,
    IRType.TUPLE_UNPACK: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_UNPACK_FIRST: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_EXPLODE: InstructionKind.TUPLE_ACCESS,
    IRType.TUPLE_LAST: InstructionKind.TUPLE_ACCESS,

    # ===== Dictionary Operations =====
    IRType.DICT_GET: InstructionKind.DICT_GET,
    IRType.DICT_SET: InstructionKind.DICT_SET,
    IRType.DICT_DELETE: InstructionKind.DICT_DELETE,
    IRType.DICT_ITERATE: InstructionKind.DICT_GET,
    IRType.DICT_OPERATIONS: InstructionKind.DICT_GET,
    IRType.DICT_LOAD: InstructionKind.DICT_GET,
    IRType.DICT_STORE: InstructionKind.DICT_SET,
    IRType.DICT_EMPTY: InstructionKind.COMPARISON,
    IRType.DICT_GET_OPT: InstructionKind.DICT_GET,

    # ===== Application-Level: Gas Operations =====
    IRType.ACCEPT_GAS: InstructionKind.ACCEPT,
    IRType.SET_GAS_LIMIT: InstructionKind.ACCEPT,
    IRType.GET_GAS_CONSUMED: InstructionKind.ARITHMETIC,
    IRType.COMMIT_STATE: InstructionKind.COMMIT,

    # ===== Application-Level: Actions =====
    IRType.SEND_MESSAGE: InstructionKind.SEND_MESSAGE,
    IRType.RESERVE_CURRENCY: InstructionKind.RESERVE,
    IRType.SET_CODE: InstructionKind.SET_CODE,
    IRType.SET_LIB_CODE: InstructionKind.SET_CODE,

    # ===== Application-Level: Address Operations =====
    IRType.LOAD_MSG_ADDRESS: InstructionKind.CELL_LOAD,
    IRType.PARSE_MSG_ADDRESS: InstructionKind.CELL_LOAD,
    IRType.REWRITE_ADDRESS: InstructionKind.CELL_STORE,

    # ===== Application-Level: Config/Blockchain =====
    IRType.GET_BLOCKCHAIN_PARAM: InstructionKind.REGISTER_LOAD,
    IRType.GET_CONFIG_DICT: InstructionKind.REGISTER_LOAD,
    IRType.GET_CONFIG_PARAM: InstructionKind.REGISTER_LOAD,
    IRType.GET_GLOBAL_ID: InstructionKind.REGISTER_LOAD,
    IRType.GET_BLOCK_INFO: InstructionKind.REGISTER_LOAD,
    IRType.GET_GAS_FEE: InstructionKind.ARITHMETIC,
    IRType.GET_STORAGE_FEE: InstructionKind.ARITHMETIC,
    IRType.GET_FORWARD_FEE: InstructionKind.ARITHMETIC,

    # ===== Application-Level: Currency =====
    IRType.LOAD_GRAMS: InstructionKind.CELL_LOAD,
    IRType.STORE_GRAMS: InstructionKind.CELL_STORE,
    IRType.LOAD_VARINT: InstructionKind.CELL_LOAD,
    IRType.STORE_VARINT: InstructionKind.CELL_STORE,

    # ===== Application-Level: Global Variables =====
    IRType.GET_GLOBAL_VAR: InstructionKind.GLOBAL_LOAD,
    IRType.SET_GLOBAL_VAR: InstructionKind.GLOBAL_STORE,

    # ===== Application-Level: Misc =====
    IRType.GET_CELL_DATA_SIZE: InstructionKind.CELL_LOAD,
    IRType.GET_SLICE_DATA_SIZE: InstructionKind.CELL_LOAD,

    # ===== Constant Operations =====
    IRType.PUSH_INT: InstructionKind.STACK_PUSH,
    IRType.PUSH_NAN: InstructionKind.STACK_PUSH,
    IRType.PUSH_REF: InstructionKind.STACK_PUSH,
    IRType.PUSH_SLICE: InstructionKind.STACK_PUSH,
    IRType.PUSH_CONT: InstructionKind.CONT_CREATE,

    # ===== Stack Complex Operations =====
    IRType.STACK_PUSH2: InstructionKind.STACK_PUSH,
    IRType.STACK_XCHG: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_DEPTH: InstructionKind.ARITHMETIC,
    IRType.STACK_CHECK_DEPTH: InstructionKind.COMPARISON,
    IRType.STACK_ONLY_TOP: InstructionKind.STACK_POP,
    IRType.STACK_BLK_PUSH: InstructionKind.STACK_PUSH,
    IRType.STACK_BLK_DROP: InstructionKind.STACK_POP,
    IRType.STACK_BLK_SWAP: InstructionKind.STACK_SHUFFLE,
    IRType.STACK_TUCK: InstructionKind.STACK_SHUFFLE,

    # ===== Arithmetic Division Operations =====
    IRType.DIV_MOD: InstructionKind.ARITHMETIC,
    IRType.MUL_DIV: InstructionKind.ARITHMETIC,
    IRType.MUL_MOD: InstructionKind.ARITHMETIC,
    IRType.ADD_DIV_MOD: InstructionKind.ARITHMETIC,
    IRType.MOD_ROUND: InstructionKind.ARITHMETIC,

    # ===== Arithmetic Quiet Operations =====
    IRType.QUIET_FIT: InstructionKind.ARITHMETIC,
    IRType.QUIET_ADD: InstructionKind.ARITHMETIC,
    IRType.QUIET_MUL: InstructionKind.ARITHMETIC,

    # ===== Comparison Operations (additional) =====
    IRType.COMPARE_SLICE_EQ: InstructionKind.COMPARISON,
    IRType.COMPARE_SLICE_PREFIX: InstructionKind.COMPARISON,

    # ===== Advanced Crypto =====
    IRType.CRYPTO_BLS_G1_ADD: InstructionKind.ARITHMETIC,
    IRType.CRYPTO_BLS_G1_MUL: InstructionKind.ARITHMETIC,
    IRType.CRYPTO_BLS_PAIRING: InstructionKind.ARITHMETIC,
    IRType.CRYPTO_P256_CHKSIG: InstructionKind.COMPARISON,
    IRType.CRYPTO_RIST255_OP: InstructionKind.ARITHMETIC,
    IRType.CRYPTO_ECRECOVER: InstructionKind.ARITHMETIC,

    # ===== Debug/Other =====
    IRType.DEBUG_PRINT: InstructionKind.DEBUG,
    IRType.NOP: InstructionKind.NOP,

    # ===== Fallback for unmapped opcodes =====
    IRType.UNKNOWN: InstructionKind.UNKNOWN,
    IRType.RAW_OPCODE: InstructionKind.UNKNOWN,
}


def get_instruction_kind(ir_type: Optional[IRType]) -> InstructionKind:
    """
    Get high-level instruction classification from IRType.

    Args:
        ir_type: The IRType to classify, or None

    Returns:
        The corresponding InstructionKind, or UNKNOWN if not found
    """
    if ir_type is None:
        return InstructionKind.UNKNOWN
    return IRTYPE_TO_KIND.get(ir_type, InstructionKind.UNKNOWN)


def get_instruction_kind_from_opcode(opcode: str) -> InstructionKind:
    """
    Get instruction kind directly from opcode string.

    Args:
        opcode: The opcode mnemonic (e.g., "ADD", "PUSHINT")

    Returns:
        The corresponding InstructionKind, or UNKNOWN if not found
    """
    from .alias_registry import resolve_alias
    from .opcode_mappings import OPCODE_TO_IR_TYPE

    ir_type = OPCODE_TO_IR_TYPE.get(opcode)
    if ir_type is None:
        alias_opcode = resolve_alias(opcode)
        if alias_opcode != opcode:
            ir_type = OPCODE_TO_IR_TYPE.get(alias_opcode)
    return get_instruction_kind(ir_type)


def is_sensitive_kind(kind: InstructionKind) -> bool:
    """
    Check if an instruction kind represents a security-sensitive operation.

    Args:
        kind: The InstructionKind to check

    Returns:
        True if this kind is security-sensitive
    """
    sensitive_kinds = {
        InstructionKind.SEND_MESSAGE,
        InstructionKind.RESERVE,
        InstructionKind.SET_CODE,
        InstructionKind.ACCEPT,
        InstructionKind.COMMIT,
        InstructionKind.REGISTER_STORE,  # c4/c5 writes
        InstructionKind.GLOBAL_STORE,
        InstructionKind.DICT_SET,
        InstructionKind.DICT_DELETE,
    }
    return kind in sensitive_kinds


def is_control_flow_kind(kind: InstructionKind) -> bool:
    """
    Check if an instruction kind affects control flow.

    Args:
        kind: The InstructionKind to check

    Returns:
        True if this kind affects control flow
    """
    control_flow_kinds = {
        InstructionKind.CONT_CALL,
        InstructionKind.CONT_JUMP,
        InstructionKind.CONT_RETURN,
        InstructionKind.BRANCH_CONDITIONAL,
        InstructionKind.BRANCH_UNCONDITIONAL,
        InstructionKind.LOOP,
        InstructionKind.THROW,
        InstructionKind.TRY_CATCH,
    }
    return kind in control_flow_kinds


def is_data_access_kind(kind: InstructionKind) -> bool:
    """
    Check if an instruction kind performs data access.

    Args:
        kind: The InstructionKind to check

    Returns:
        True if this kind performs data access
    """
    data_access_kinds = {
        InstructionKind.CELL_LOAD,
        InstructionKind.CELL_STORE,
        InstructionKind.REGISTER_LOAD,
        InstructionKind.REGISTER_STORE,
        InstructionKind.GLOBAL_LOAD,
        InstructionKind.GLOBAL_STORE,
        InstructionKind.DICT_GET,
        InstructionKind.DICT_SET,
        InstructionKind.DICT_DELETE,
        InstructionKind.TUPLE_ACCESS,
    }
    return kind in data_access_kinds
