"""
Stack Operations

This module contains mappings for:
- Basic stack operations (PUSH, POP, DROP, DUP)
- Exchange operations (XCHG, SWAP)
- Complex stack manipulation (XCPU, PUXC, etc.)
- Rotation, reversal, and block operations
"""

from ..ir_types import IRType

STACK_OPS_MAPPINGS = {
    # ========== Stack Operations - Basic ==========
    "NOP": IRType.NOP,
    "PUSH": IRType.PUSH_VALUE,
    "PUSH3": IRType.PUSH_VALUE,
    "PUSH_LONG": IRType.PUSH_VALUE,

    "POP": IRType.POP_VALUE,
    "POP_LONG": IRType.POP_VALUE,
    "DROP": IRType.POP_VALUE,
    "DROP2": IRType.POP_VALUE,
    "NIP": IRType.POP_VALUE,
    "BLKDROP2": IRType.POP_VALUE,
    "DROPX": IRType.POP_VALUE,

    "DUP": IRType.DUP_VALUE,
    "DUP2": IRType.DUP_VALUE,
    "OVER": IRType.DUP_VALUE,

    # Stack - Complex
    "XCHG": IRType.STACK_EXCHANGE,
    "XCHG_0I": IRType.STACK_EXCHANGE,
    "XCHG_0I_LONG": IRType.STACK_EXCHANGE,
    "XCHG_1I": IRType.STACK_EXCHANGE,
    "XCHG_IJ": IRType.STACK_EXCHANGE,
    "XCHG2": IRType.STACK_EXCHANGE,
    "XCHG3": IRType.STACK_EXCHANGE,
    "XCHG3_ALT": IRType.STACK_EXCHANGE,

    "XCPU": IRType.STACK_COMPLEX,
    "PUXC": IRType.STACK_COMPLEX,
    "XC2PU": IRType.STACK_COMPLEX,
    "XCPUXC": IRType.STACK_COMPLEX,
    "XCPU2": IRType.STACK_COMPLEX,
    "PUXC2": IRType.STACK_COMPLEX,
    "PUXCPU": IRType.STACK_COMPLEX,
    "PU2XC": IRType.STACK_COMPLEX,

    "ROT": IRType.STACK_ROTATE,
    "ROTREV": IRType.STACK_ROTATE,
    "-ROT": IRType.STACK_ROTATE,

    "SWAP": IRType.STACK_SWAP,
    "SWAP2": IRType.STACK_SWAP,

    "BLKSWX": IRType.STACK_BLOCK_SWAP,

    "REVERSE": IRType.STACK_REVERSE,
    "REVX": IRType.STACK_REVERSE,

    "PICK": IRType.STACK_PICK,
    "PICKX": IRType.STACK_PICK,

    "ROLL": IRType.STACK_ROLL,
    "ROLLREV": IRType.STACK_ROLL,
    "ROLLX": IRType.STACK_ROLL,
    "-ROLL": IRType.STACK_ROLL,
    "-ROLLX": IRType.STACK_ROLL,

    # Stack Complex Operations (additional)
    "PUSH2": IRType.STACK_PUSH2,
    "OVER2": IRType.STACK_PUSH2,
    "XCHGX": IRType.STACK_XCHG,
    "DEPTH": IRType.STACK_DEPTH,
    "CHKDEPTH": IRType.STACK_CHECK_DEPTH,
    "ONLYTOPX": IRType.STACK_ONLY_TOP,
    "ONLYX": IRType.STACK_ONLY_TOP,
    "BLKPUSH": IRType.STACK_BLK_PUSH,
    "BLKDROP": IRType.STACK_BLK_DROP,
    "BLKSWAP": IRType.STACK_BLK_SWAP,
    "TUCK": IRType.STACK_TUCK,
}
