"""
Unified Opcode Alias Registry

Single source of truth for TVM opcode aliases, used by both
stack effect resolution and IR type mapping.

Previously aliases were scattered across:
- StackEffectDatabase.INSTRUCTION_ALIASES (stack_effects.py)
- Duplicate entries in OPCODE_TO_IR_TYPE (mappings/*.py)
- cp0.json aliases field

This module consolidates them into one registry.
"""

# Alias mapping: disassembler name -> canonical cp0.json name
# The disassembler may use simplified or variant names for common operations.
# This mapping translates them to the official TVM specification names
# for consistent stack effect and IR type lookup.
OPCODE_ALIASES = {
    # Note: SWAP, DUP, DROP, OVER have hardcoded handling in dataflow.py for performance.
    # MANUAL_STACK_EFFECTS in stack_effects.py still defines their stack effects for the
    # effect database.

    # Stack exchange operations
    # "SWAP": "XCHG_0I",  # Would need parameter i=1
    # "XCHG": "XCHG_0I",  # Generic exchange

    # Parametric variants (disassembler adds underscore for immediate operand versions)
    # Arithmetic shift operations with immediate shift amount
    "MULRSHIFT_": "MULRSHIFT",        # x y - floor(x*y/2^z) where z is immediate
    "MULRSHIFTR_": "MULRSHIFTR",      # x y - round(x*y/2^z) where z is immediate
    "MULRSHIFTC_": "MULRSHIFTC",      # x y - ceil(x*y/2^z) where z is immediate
    "RSHIFTR_": "RSHIFTR",            # x - round(x/2^z) where z is immediate
    "RSHIFTC_": "RSHIFTC",            # x - ceil(x/2^z) where z is immediate
    "RSHIFT_": "RSHIFT",              # x - floor(x/2^z) where z is immediate
    "RSHIFT_ALT": "RSHIFT_VAR",       # stack-based variant (x y -> result)
    "RSHIFT_MOD": "RSHIFTMOD",        # immediate variant (x -> q r)
    "RSHIFTMODR": "RSHIFTMODR_VAR",   # stack-based variant (x y -> q r)
    "RSHIFTMODC": "RSHIFTMODC_VAR",   # stack-based variant (x y -> q r)

    # Quiet arithmetic/compare variants
    "QBITSIZE": "BITSIZE",
    "QUBITSIZE": "UBITSIZE",
    "QMIN": "MIN",
    "QMAX": "MAX",
    "QMINMAX": "MINMAX",
    "QABS": "ABS",
    "QSGN": "SGN",
    "QLESS": "LESS",
    "QEQUAL": "EQUAL",
    "QLEQ": "LEQ",
    "QGREATER": "GREATER",
    "QNEQ": "NEQ",
    "QGEQ": "GEQ",
    "QCMP": "CMP",
    "QEQINT": "EQINT",
    "QLESSINT": "LESSINT",
    "QGTINT": "GTINT",
    "QNEQINT": "NEQINT",

    # Quiet shift/mod variants
    "QRSHIFT_ALT": "QRSHIFT_VAR",
    "QRSHIFTMODR": "QRSHIFTMODR_VAR",
    "QRSHIFTMODC": "QRSHIFTMODC_VAR",

    # Control register operations
    "SAVECTR": "SAVE",                # Alias: disassembler name for SAVE instruction
    "SAVEALTCTR": "SAVEALT",
    "SAVEBOTHCTR": "SAVEBOTH",
    "SAVECONT": "SAVE",

    # Codepage operations
    "SETCP_SHORT": "SETCP",           # Sets codepage (no stack effect)
    "SETCP0": "SETCP",
    "SETCPR": "SETCP",

    # Control flow aliases
    "RETBOOL": "BRANCH",              # Maps to BRANCH for stack effect lookup (both pop 1 boolean)
    "JMP_JMPREF": "JMPREF",           # Disassembler variant; same stack shape as JMPREF
    "SETNUMARGS": "SETCONTARGS_N",
    "POPROOT": "POPCTR",
    "PUSHROOT": "PUSHCTR",
    "BLESSNUMARGS": "BLESSARGS",

    # Stack operation aliases
    "2DUP": "DUP2",  # Alternative name for DUP2
    "NIP": "POP",
    "ROLLREV": "ROLL",
    "ROTR": "ROTREV",
    "ROT2": "BLKSWAP",
    "XCHG_LONG": "XCHG_0I_LONG",
    "POPREF": "POP",

    # Call variants
    "CALLX": "EXECUTE",               # Fift alias: execute continuation from stack
    "CALLCCARGS_VAR": "CALLCCARGS",   # Parameterized alias with same base stack shape

    # Dictionary aliases
    "PFXDICTSWITCH": "PFXDICTCONSTGETJMP",
    "LDOPTREF": "LDDICT",
    "PLDOPTREF": "PLDDICT",
    "NEWDICT": "NULL",
    "STDICTS": "STSLICE",
    "STONE": "STSLICECONST",

    # Tuple aliases
    "NIL": "TUPLE",
    "SINGLE": "TUPLE",
    "PAIR": "TUPLE",
    "TRIPLE": "TUPLE",
    "UNSINGLE": "UNTUPLE",
    "UNPAIR": "UNTUPLE",
    "UNTRIPLE": "UNTUPLE",
    "FIRST": "INDEX",
    "SECOND": "INDEX",
    "THIRD": "INDEX",
    "FIRSTQ": "INDEXQ",
    "SECONDQ": "INDEXQ",
    "THIRDQ": "INDEXQ",
    "SETFIRST": "SETINDEX",
    "SETSECOND": "SETINDEX",
    "SETTHIRD": "SETINDEX",
    "SETFIRSTQ": "SETINDEXQ",
    "SETSECONDQ": "SETINDEXQ",
    "SETTHIRDQ": "SETINDEXQ",
    "CADR": "INDEX2",
    "CDDR": "INDEX2",
    "CADDR": "INDEX3",
    "CDDDR": "INDEX3",
    "CHKTUPLE": "UNPACKFIRST",

    # Integer literal aliases
    "ZERO": "PUSHINT_4",
    "ONE": "PUSHINT_4",
    "TWO": "PUSHINT_4",
    "TEN": "PUSHINT_4",
    "TRUE": "PUSHINT_4",

    # Misc aliases
    "CHKBOOL": "FITS",
    "CHKBIT": "UFITS",
    "PLDREF": "PLDREFIDX",
    "ACCEPTQ": "ACCEPT",
    "PSEUDO_PUSHSLICE": "PUSHSLICE",
    "PSEUDO_PUSHREF": "PUSHREF",
    "PSEUDO_EXOTIC": "PUSHREF",

    # Note: Most disassembler names don't have direct cp0 equivalents
    # because the disassembler generates parameterized variants as distinct names
    # Example: DUP is PUSH with specific parameters in cp0.json
    # These cases are better handled by MANUAL_STACK_EFFECTS in stack_effects.py
}


def resolve_alias(opcode: str) -> str:
    """Resolve an opcode alias to its canonical name.

    Returns the canonical name if an alias exists, otherwise returns
    the original opcode unchanged.
    """
    return OPCODE_ALIASES.get(opcode, opcode)
