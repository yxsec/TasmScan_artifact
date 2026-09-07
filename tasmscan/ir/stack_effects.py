"""
Stack Effects Loader for TVM Opcodes

Loads stack effect information from cp0.json specification to enable
comprehensive stack simulation for data flow analysis.

The stack-effect model covers the supported TVM instruction set and uses
conservative fallbacks for instructions that cannot be modeled precisely.
"""
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from .alias_registry import OPCODE_ALIASES

logger = logging.getLogger(__name__)

# Track unknown opcodes for reporting (module-level)
# Limited to prevent unbounded memory growth in long-running processes
MAX_UNKNOWN_OPCODES_TRACKED = 1000
_unknown_opcodes_encountered: Set[str] = set()
import threading
_unknown_opcodes_lock = threading.Lock()

# Environment variable to enable strict mode
STRICT_MODE = os.environ.get("TasmScan_STRICT", "").lower() in ("1", "true", "yes")


@dataclass
class StackEffect:
    """Represents the stack effect of a TVM opcode"""
    # NOTE: inputs/outputs represent the *minimum* guaranteed pops/pushes.
    # For dynamic instructions, use min_* / max_* for interval semantics.
    inputs: int  # Minimum number of values popped from stack
    outputs: int  # Minimum number of values pushed to stack
    input_names: List[str]  # Names of input values (for debugging)
    output_names: List[str]  # Names of output values (for debugging)
    is_dynamic: bool = False  # True if effect depends on runtime/immediate parameters
    min_inputs: Optional[int] = None  # Lower bound (defaults to inputs)
    max_inputs: Optional[int] = None  # Upper bound (None = unbounded/unknown)
    min_outputs: Optional[int] = None  # Lower bound (defaults to outputs)
    max_outputs: Optional[int] = None  # Upper bound (None = unbounded/unknown)

    def __post_init__(self) -> None:
        if self.min_inputs is None:
            self.min_inputs = self.inputs
        if self.min_outputs is None:
            self.min_outputs = self.outputs
        if self.max_inputs is None and not self.is_dynamic:
            self.max_inputs = self.inputs
        if self.max_outputs is None and not self.is_dynamic:
            self.max_outputs = self.outputs

    @property
    def net_effect(self) -> int:
        """Net change to stack depth using minimum inputs/outputs."""
        return self.outputs - self.inputs

    def net_effect_range(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (min_net_effect, max_net_effect) or None for unknown bounds."""
        min_effect = None
        max_effect = None
        if self.max_inputs is not None:
            min_effect = self.min_outputs - self.max_inputs
        if self.max_outputs is not None:
            max_effect = self.max_outputs - self.min_inputs
        return min_effect, max_effect


class StackEffectDatabase:
    """
    Database of stack effects for all TVM opcodes.

    Loads from cp0.json specification and provides fast lookup by mnemonic.
    Also includes manual supplements for opcodes missing from the spec.
    """

    # See alias_registry.py for alias definitions.
    INSTRUCTION_ALIASES = OPCODE_ALIASES

    # Manual stack effect definitions for opcodes missing from cp0.json
    # Format: mnemonic -> (min_inputs, min_outputs, input_names, output_names, is_dynamic)
    # is_dynamic=True means effect depends on runtime/immediate parameters
    MANUAL_STACK_EFFECTS = {
        # Stack manipulation - basic
        "NOP": (0, 0, [], [], False),
        "NIP": (2, 1, ["x", "y"], ["y"], False),  # x y -> y
        "PUSHINT": (0, 1, [], ["int"], False),  # Push integer constant
        "PUSHNULL": (0, 1, [], ["null"], False),
        "DROP2": (2, 0, ["x", "y"], [], False),
        "DROP": (1, 0, ["x"], [], False),
        "DUP": (1, 2, ["x"], ["x", "x"], False),
        "OVER": (2, 3, ["x", "y"], ["x", "y", "x"], False),
        "TUCK": (2, 3, ["a", "b"], ["b", "a", "b"], False),  # a b -> b a b
        "DUP2": (2, 4, ["x", "y"], ["x", "y", "x", "y"], False),
        "OVER2": (4, 6, ["x", "y", "z", "w"], ["x", "y", "z", "w", "x", "y"], False),
        "REVERSE": (2, 2, ["x", "y"], ["y", "x"], False),
        # ROT/ROTREV: 3-element rotations
        "ROT": (3, 3, ["a", "b", "c"], ["b", "c", "a"], False),  # a b c -> b c a
        "ROTREV": (3, 3, ["a", "b", "c"], ["c", "a", "b"], False),  # a b c -> c a b
        "ROTR": (3, 3, ["a", "b", "c"], ["c", "a", "b"], False),  # Right rotate (same as ROTREV)
        "-ROT": (3, 3, ["a", "b", "c"], ["c", "a", "b"], False),  # Alias for ROTREV
        "DEPTH": (0, 1, [], ["depth"], False),
        # Stack exchanges (parametric) - minimum required depth
        "XCHG_0I": (2, 2, ["s0", "si"], ["s0", "si"], False),
        "XCHG_0I_LONG": (2, 2, ["s0", "si"], ["s0", "si"], False),  # Exchange s0 with s[i]
        "XCHG_1I": (3, 3, ["s0", "s1", "si"], ["s0", "s1", "si"], False),
        "XCHG_IJ": (3, 3, ["s0", "si", "sj"], ["s0", "si", "sj"], False),
        "XCHG2": (2, 2, ["s0", "s1"], ["s0", "s1"], False),
        "XCHG3": (3, 3, ["s0", "s1", "s2"], ["s0", "s1", "s2"], False),
        "XCHG3_ALT": (3, 3, ["s0", "s1", "s2"], ["s0", "s1", "s2"], False),
        # Parametric stack copy/pop
        "PUSH": (1, 2, ["s[i]"], ["s[i]", "s[i]"], False),
        "POP": (1, 0, ["s0"], [], False),
        "POPREF": (1, 0, ["x"], [], False),  # Pop to register via cell ref
        "PUSH2": (2, 4, ["s[i]", "s[j]"], ["s[i]", "s[j]", "s[i]", "s[j]"], False),  # Push two copies
        "PUXC": (2, 3, ["s0", "s[i]"], ["s0", "s[i]", "s0"], False),  # Push s(i) then xchg
        # Basic shuffles (aliases)
        "SWAP": (2, 2, ["s0", "s1"], ["s0", "s1"], False),
        "XCHG": (2, 2, ["s0", "s1"], ["s0", "s1"], False),

        # Stack manipulation - with DYNAMIC effects (conservative estimates)
        # NOTE: These have runtime-dependent behavior. Values are conservative approximations.
        #
        # PICK/ROLL/ROLLX Dynamic Behavior:
        # ---------------------------------
        # These instructions read their operand 'i' from the stack at runtime, making
        # their stack effect unpredictable at static analysis time. For example:
        # - PICK: Pops index i, then copies s[i] to TOS. True effect: (i+1) -> 1
        # - ROLL: Pops index i, then rotates i+1 elements. True effect: (i+1) -> i
        #
        # We use minimum guaranteed pops/pushes and mark is_dynamic=True so
        # analyzers can apply interval-based heuristics.
        #
        # Actual: PICK pops i, pushes s[i] → (i+1, 1) inputs/outputs
        "PICK": (1, 1, ["i"], ["s[i]"], True),
        # Actual: ROLL pops i, rotates stack → (i+1, i) inputs/outputs
        "ROLL": (1, 0, ["i"], [], True),
        # Actual: ROLLX pops i, performs BLKSWAP → (i+1, i) inputs/outputs
        "ROLLX": (1, 0, ["i"], [], True),
        "-ROLL": (1, 0, ["i"], [], True),
        "-ROLLX": (1, 0, ["i"], [], True),
        # Actual: DROPX pops n, drops n items → (n+1, 0) inputs/outputs
        "DROPX": (1, 0, ["n"], [], True),
        # Actual: XCHGX pops i, then swaps s0 with s[i] (dynamic reordering)
        "XCHGX": (1, 0, ["i"], [], True),
        # Actual: ONLYX keeps only n items → dynamic effect
        "ONLYX": (1, 0, ["n"], [], True),
        "ONLYTOPX": (1, 0, ["n"], [], True),

        # Block operations - with DYNAMIC effects (immediate parameter)
        "BLKDROP": (0, 0, [], [], True),  # Parameterized, depends on immediate
        "BLKDROP2": (0, 0, [], [], True),
        "BLKPUSH": (0, 0, [], [], True),
        "BLKSWAP": (0, 0, [], [], True),
        "BLKSWX": (1, 0, ["i"], [], True),  # Pops i parameter

        # Control flow - no stack effect
        "ACCEPT": (0, 0, [], [], False),  # Sets gas limit
        "ACCEPTQ": (0, 0, [], [], False),  # Quiet ACCEPT - no stack effect
        "COMMIT": (0, 0, [], [], False),  # Commits current state
        "JMP": (0, 0, [], [], False),  # Jump (doesn't return, consumes continuation)
        "JMPREF": (0, 0, [], [], False),  # Jump (doesn't return)
        "JMPREFDATA": (0, 0, [], [], False),
        # CALLREF: Calls a continuation stored in a cell reference.
        # Dynamic because: (1) The called continuation may have arbitrary stack effects
        # that cannot be determined statically. (2) The continuation code is resolved
        # at runtime from the cell reference. Analyzer must treat conservatively.
        "CALLREF": (0, 0, [], [], True),
        "SETCPR": (0, 0, [], [], False),  # Set codepage register variant

        # Debug operations
        "DEBUG": (0, 0, [], [], False),  # no-op per cp0.json (exec_dummy_debug); observes stack without popping
        "DEBUGSTR": (0, 0, [], [], False),  # Debug string
        "DUMPSTK": (0, 0, [], [], False),  # Dump stack
        "STRDUMP": (0, 0, [], [], False),  # alias of DEBUG, no-op per cp0.json doc_stack: "-"

        # Pseudo instructions emitted by the disassembler
        "PSEUDO_PUSHSLICE": (0, 1, [], ["slice"], False),
        "PSEUDO_PUSHREF": (0, 1, [], ["ref"], False),
        "PSEUDO_EXOTIC": (0, 1, [], ["cell"], False),

        # Dictionary get + execute (complex) - i D n -> i or nothing (dynamic output)
        "DICTIGETEXECZ": (3, 0, ["i", "D", "n"], [], True),  # Get and execute, or return i
        "DICTIGETJMPZ": (3, 0, ["i", "D", "n"], [], True),   # Get and jump, or return i
        "DICTUGETEXECZ": (3, 0, ["i", "D", "n"], [], True),  # Unsigned: get and execute, or return i
        "DICTUGETJMPZ": (3, 0, ["i", "D", "n"], [], True),   # Unsigned: get and jump, or return i
        "DICTGETEXECZ": (3, 0, ["k", "D", "n"], [], True),   # Slice key: get and execute
        "DICTGETJMPZ": (3, 0, ["k", "D", "n"], [], True),    # Slice key: get and jump

        # Prefix dictionary operations - based on cp0.json spec
        # PFXDICTCONSTGETJMP: s - s' s'' or s (combines DICTPUSHCONST with PFXDICTGETJMP)
        "PFXDICTCONSTGETJMP": (1, 0, ["s"], [], True),  # Dynamic: success=2 outputs, fail=1
        # PFXDICTGETJMP: s D n - s' s'' or s
        "PFXDICTGETJMP": (3, 0, ["s", "D", "n"], [], True),  # Dynamic: success=2 outputs, fail=1

        # Loop primitives - based on cp0.json spec
        # *END variants operate on current continuation (cc), *BRK variants take explicit continuation
        "AGAINEND": (0, 0, [], [], False),      # - (no stack effect, operates on cc)
        "AGAINENDBRK": (0, 0, [], [], False),   # - (operates on cc with break support)
        "AGAINBRK": (1, 0, ["c"], [], False),   # c - (takes continuation from stack)
        "REPEATEND": (1, 0, ["n"], [], False),  # n - (pops count, loops cc)
        "REPEATENDBRK": (1, 0, ["n"], [], False),  # n - (operates on cc with break support)
        "REPEATBRK": (2, 0, ["n", "c"], [], False),  # n c - (takes count and continuation)
        "UNTILEND": (0, 0, [], [], False),      # - (condition checked inside loop body)
        "UNTILENDBRK": (0, 0, [], [], False),   # - (operates on cc with break support)
        "UNTILBRK": (1, 0, ["c"], [], False),   # c - (takes continuation from stack)
        "WHILEEND": (1, 0, ["c'"], [], False),  # c' - (pops condition continuation)
        "WHILEENDBRK": (1, 0, ["c"], [], False),  # c - (operates on cc with break support)
        "WHILEBRK": (2, 0, ["c'", "c"], [], False),  # c' c - (takes cond and body continuations)

        # Bitwise
        "INVERT": (1, 1, ["x"], ["~x"], False),  # Bitwise NOT

        # Tuple operations
        "TPUSH": (2, 1, ["t", "x"], ["t'"], False),  # Push to tuple
        "TPOP": (1, 2, ["t"], ["t'", "x"], False),  # Pop from tuple

        # Exception handling
        "THROWANY": (1, 0, ["n"], [], False),  # Throw exception

        # Continuation operations
        "PUSHCONT": (0, 1, [], ["cont"], False),
        "POPCTR": (1, 0, ["x"], [], False),  # Pop to c[i]
        "POPCTRX": (2, 0, ["x", "i"], [], False),
        "PUSHCTR": (0, 1, [], ["c[i]"], False),
        "PUSHCTRX": (1, 1, ["i"], ["c[i]"], False),
        "POPSAVE": (1, 0, ["x"], [], False),
        "SAVECONT": (1, 1, ["cont"], ["cont'"], False),

        # Block manipulation (long forms)
        "POP_LONG": (0, 0, [], [], False),
        "PUSH_LONG": (1, 2, ["s[i]"], ["s[i]", "s[i]"], False),  # Long-form PUSH s(i) copy
        "XCHG_LONG": (0, 0, [], [], False),

        # ===== ADDITIONAL MISSING INSTRUCTIONS =====

        # Message/Balance Information Primitives (IMPORTANT for security)
        "BALANCE": (0, 1, [], ["balance"], False),  # Get remaining balance
        "INCOMINGVALUE": (0, 1, [], ["value"], False),  # Get incoming message value
        "STORAGEFEES": (0, 1, [], ["fees"], False),  # Get storage fees paid
        "DUEPAYMENT": (0, 1, [], ["payment"], False),  # Get due payment

        # Incoming Message Parameters (IMPORTANT - taint sources)
        "INMSGPARAM": (0, 1, [], ["param"], False),  # Get msg parameter by immediate index
        "INMSGPARAMS": (0, 1, [], ["params"], False),  # Get all msg parameters
        "INMSG_VALUE": (0, 1, [], ["value"], False),  # Get incoming msg value
        "INMSG_ORIGVALUE": (0, 1, [], ["value"], False),  # Get original msg value
        "INMSG_VALUEEXTRA": (0, 1, [], ["extra"], False),  # Get extra currencies
        "INMSG_FWDFEE": (0, 1, [], ["fee"], False),  # Get forwarding fee
        "INMSG_LT": (0, 1, [], ["lt"], False),  # Get logical time
        "INMSG_UTIME": (0, 1, [], ["utime"], False),  # Get Unix time
        "INMSG_SRC": (0, 1, [], ["src"], False),  # Get source address
        "INMSG_BOUNCE": (0, 1, [], ["bounce"], False),  # Get bounce flag
        "INMSG_BOUNCED": (0, 1, [], ["bounced"], False),  # Get bounced flag
        "INMSG_STATEINIT": (0, 1, [], ["init"], False),  # Get state init

        # Configuration Parameters
        "CONFIGROOT": (0, 1, [], ["root"], False),  # Get config root cell
        "GETPARAMLONG": (0, 1, [], ["param"], False),  # Get config param by immediate index
        "GETPARAMLONG2": (2, 1, ["idx", "def"], ["param"], False),  # Get param with default
        "UNPACKEDCONFIGTUPLE": (0, 1, [], ["tuple"], False),  # Unpack config tuple

        # Arithmetic Operations (shortcuts/variants)
        "ADDINT": (1, 1, ["x"], ["sum"], False),  # x + immediate
        "SUBINT": (1, 1, ["x"], ["diff"], False),  # x - immediate (alias of ADDINT with negated operand)
        "MULINT": (1, 1, ["x"], ["product"], False),  # x * immediate
        "QADDINT": (1, 1, ["x"], ["sum"], False),  # Quiet: x + immediate
        "QSUBINT": (1, 1, ["x"], ["diff"], False),  # Quiet: x - immediate
        "QMULINT": (1, 1, ["x"], ["product"], False),  # Quiet: x * immediate

        # Boolean/Logic Operations
        "BOOLAND": (2, 1, ["x", "y"], ["result"], False),  # Boolean AND
        "BOOLOR": (2, 1, ["x", "y"], ["result"], False),  # Boolean OR

        # Call Operations with Fixed Args
        "CALLXARGS_1": (2, 0, ["arg", "cont"], [], False),  # Call with 1 argument

        # Hash Operations
        "HASHBU": (1, 1, ["s"], ["hash"], False),  # Hash builder/slice
        "HASHEXT": (1, 1, ["n"], ["hash"], True),  # s_1..s_n n -> hash (dynamic array input)
        "HASHEXTA": (1, 1, ["n"], ["hash"], True),  # s_1..s_n n -> hash (dynamic, appends)
        "HASHEXTR": (1, 1, ["n"], ["hash"], True),  # s_1..s_n n -> hash (dynamic, ref)
        "HASHEXTAR": (1, 1, ["n"], ["hash"], True),  # s_1..s_n n -> hash (dynamic, append+ref)
        "BLS_G1_FROMHASH": (1, 1, ["h"], ["g1_point"], False),  # hash -> G1 point (analogous to BLS_MAP_TO_G1)
        "BLS_G2_FROMHASH": (1, 1, ["h"], ["g2_point"], False),  # hash -> G2 point (analogous to BLS_MAP_TO_G2)

        # Time/Block Info
        "BLOCKLT": (0, 1, [], ["lt"], False),  # Get block logical time
        "LTIME": (0, 1, [], ["lt"], False),  # Get logical time

        # Type Conversions
        "BTOS": (1, 1, ["b"], ["s"], False),  # Builder to Slice
        "ENDCST": (2, 1, ["b", "c"], ["c'"], False),  # End cell and store to builder

        # Debug Operations (safe to ignore in analysis)
        "DEBUG_1": (0, 0, [], [], False),  # no-op per cp0.json (all DEBUG variants are no-ops)
        "DEBUG_2": (0, 0, [], [], False),  # no-op per cp0.json (all DEBUG variants are no-ops)
        "DEBUGMARK": (0, 0, [], [], False),  # Debug marker
        "DUMP": (0, 0, [], [], False),  # alias of DEBUG, no-op per cp0.json doc_stack: "-"

        # Shift-arithmetic combinations (complex operations)
        "LSHIFT_DIV": (3, 1, ["x", "y", "z"], ["result"], False),
        "LSHIFT_DIVC": (3, 1, ["x", "y", "z"], ["result"], False),
        "LSHIFT_DIVR": (3, 1, ["x", "y", "z"], ["result"], False),
        "LSHIFT_MOD": (3, 1, ["x", "y", "z"], ["result"], False),
        "LSHIFT_MODC": (3, 1, ["x", "y", "z"], ["result"], False),
        "LSHIFT_MODR": (3, 1, ["x", "y", "z"], ["result"], False),
        "LSHIFT_DIVMOD": (3, 2, ["x", "y", "z"], ["q", "r"], False),
        "LSHIFT_DIVMODC": (3, 2, ["x", "y", "z"], ["q", "r"], False),
        "LSHIFT_DIVMODR": (3, 2, ["x", "y", "z"], ["q", "r"], False),
        "LSHIFT_ADDDIVMOD": (3, 2, ["x", "y", "z"], ["q", "r"], False),
        "LSHIFT_ADDDIVMODC": (3, 2, ["x", "y", "z"], ["q", "r"], False),
        "LSHIFT_ADDDIVMODR": (3, 2, ["x", "y", "z"], ["q", "r"], False),

        # Modulo power of 2 operations
        "MODPOW2_": (1, 1, ["x"], ["result"], False),    # Immediate variant: shift from operand
        "MODPOW2C_": (1, 1, ["x"], ["result"], False),
        "MODPOW2R_": (1, 1, ["x"], ["result"], False),

        # Shift-with-remainder (immediate variants): x -> q r
        "RSHIFTR_MOD": (1, 2, ["x"], ["q", "r"], False),
        "RSHIFTC_MOD": (1, 2, ["x"], ["q", "r"], False),

        # Multiply-add-shift operations (for fixed-point math)
        "MULADDRSHIFT_MOD": (3, 1, ["x", "y", "z"], ["result"], False),
        "MULADDRSHIFTC_MOD": (3, 1, ["x", "y", "z"], ["result"], False),
        "MULADDRSHIFTR_MOD": (3, 1, ["x", "y", "z"], ["result"], False),

        # Multiply-mod power of 2 (immediate variants): x y -> result
        "MULMODPOW2_": (2, 1, ["x", "y"], ["result"], False),
        "MULMODPOW2R_": (2, 1, ["x", "y"], ["result"], False),
        "MULMODPOW2C_": (2, 1, ["x", "y"], ["result"], False),

        # Add-shift operations
        "ADDRSHIFT_MOD": (3, 1, ["x", "y", "z"], ["result"], False),
        "ADDRSHIFTC_MOD": (3, 1, ["x", "y", "z"], ["result"], False),
        "ADDRSHIFTR_MOD": (3, 1, ["x", "y", "z"], ["result"], False),

        # ===== SEND OPERATIONS =====
        "SENDRAWMSG": (2, 0, ["msg", "mode"], [], False),

        # ===== DICTIONARY OPERATIONS =====
        # Dictionary operations: marked is_dynamic=True to preserve cp0.json conditional
        # interval semantics (success vs failure paths have different stack depths).
        # cp0.json correctly models these as conditional outputs; setting is_dynamic=True
        # prevents the manual supplement from overriding the spec-derived precision.

        # Basic dictionary get/set/del
        "DICTGET": (3, 2, ["key", "dict", "keylen"], ["value", "found"], True),
        "DICTGETREF": (3, 2, ["key", "dict", "keylen"], ["cell", "found"], True),
        "DICTSET": (4, 1, ["value", "key", "dict", "keylen"], ["dict'"], True),
        "DICTSETREF": (4, 1, ["cell", "key", "dict", "keylen"], ["dict'"], True),
        "DICTDEL": (3, 2, ["key", "dict", "keylen"], ["dict'", "found"], True),

        # Dictionary serialization
        "STDICT": (2, 1, ["dict", "builder"], ["builder'"], True),
        "LDDICT": (1, 2, ["slice"], ["dict", "slice'"], True),
        "LDDICTS": (1, 2, ["slice"], ["slice_dict", "slice'"], True),

        # Dictionary optional reference
        "DICTGETOPTREF": (3, 1, ["key", "dict", "keylen"], ["maybe_cell"], True),

        # Unsigned key dictionary operations
        "DICTUGET": (3, 2, ["key", "dict", "keylen"], ["value", "found"], True),
        "DICTUGETREF": (3, 2, ["key", "dict", "keylen"], ["cell", "found"], True),
        "DICTUSET": (4, 1, ["value", "key", "dict", "keylen"], ["dict'"], True),
        "DICTUSETREF": (4, 1, ["cell", "key", "dict", "keylen"], ["dict'"], True),
        "DICTUDEL": (3, 2, ["key", "dict", "keylen"], ["dict'", "found"], True),

        # Signed key dictionary operations
        "DICTIGET": (3, 2, ["key", "dict", "keylen"], ["value", "found"], True),
        "DICTIGETREF": (3, 2, ["key", "dict", "keylen"], ["cell", "found"], True),
        "DICTISET": (4, 1, ["value", "key", "dict", "keylen"], ["dict'"], True),
        "DICTISETREF": (4, 1, ["cell", "key", "dict", "keylen"], ["dict'"], True),
        "DICTIDEL": (3, 2, ["key", "dict", "keylen"], ["dict'", "found"], True),

        # Dictionary add (fail if key exists) - returns D' -1 or D 0
        "DICTADD": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTADDREF": (4, 2, ["cell", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTUADD": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTUADDREF": (4, 2, ["cell", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTIADD": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTIADDREF": (4, 2, ["cell", "key", "dict", "keylen"], ["dict'", "success"], True),

        # Dictionary replace (fail if key doesn't exist) - returns D' -1 or D 0
        "DICTREPLACE": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTREPLACEREF": (4, 2, ["cell", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTUREPLACE": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTUREPLACEREF": (4, 2, ["cell", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTIREPLACE": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),
        "DICTIREPLACEREF": (4, 2, ["cell", "key", "dict", "keylen"], ["dict'", "success"], True),

        # Dictionary setget (set and return old value) - returns D' y -1 or D' 0
        "DICTSETGET": (4, 3, ["value", "key", "dict", "keylen"], ["dict'", "old_value", "found"], True),
        "DICTSETGETREF": (4, 3, ["cell", "key", "dict", "keylen"], ["dict'", "old_cell", "found"], True),
        "DICTUSETGET": (4, 3, ["value", "key", "dict", "keylen"], ["dict'", "old_value", "found"], True),
        "DICTUSETGETREF": (4, 3, ["cell", "key", "dict", "keylen"], ["dict'", "old_cell", "found"], True),
        "DICTISETGET": (4, 3, ["value", "key", "dict", "keylen"], ["dict'", "old_value", "found"], True),
        "DICTISETGETREF": (4, 3, ["cell", "key", "dict", "keylen"], ["dict'", "old_cell", "found"], True),

        # Dictionary delget (delete and return old value)
        "DICTDELGET": (3, 3, ["key", "dict", "keylen"], ["dict'", "value", "found"], True),
        "DICTDELGETREF": (3, 3, ["key", "dict", "keylen"], ["dict'", "cell", "found"], True),
        "DICTUDELGET": (3, 3, ["key", "dict", "keylen"], ["dict'", "value", "found"], True),
        "DICTUDELGETREF": (3, 3, ["key", "dict", "keylen"], ["dict'", "cell", "found"], True),
        "DICTIDELGET": (3, 3, ["key", "dict", "keylen"], ["dict'", "value", "found"], True),
        "DICTIDELGETREF": (3, 3, ["key", "dict", "keylen"], ["dict'", "cell", "found"], True),

        # Dictionary min/max operations
        "DICTMIN": (2, 3, ["dict", "keylen"], ["key", "value", "found"], True),
        "DICTMINREF": (2, 3, ["dict", "keylen"], ["key", "cell", "found"], True),
        "DICTMAX": (2, 3, ["dict", "keylen"], ["key", "value", "found"], True),
        "DICTMAXREF": (2, 3, ["dict", "keylen"], ["key", "cell", "found"], True),
        "DICTUMIN": (2, 3, ["dict", "keylen"], ["key", "value", "found"], True),
        "DICTUMINREF": (2, 3, ["dict", "keylen"], ["key", "cell", "found"], True),
        "DICTUMAX": (2, 3, ["dict", "keylen"], ["key", "value", "found"], True),
        "DICTUMAXREF": (2, 3, ["dict", "keylen"], ["key", "cell", "found"], True),
        "DICTIMIN": (2, 3, ["dict", "keylen"], ["key", "value", "found"], True),
        "DICTIMINREF": (2, 3, ["dict", "keylen"], ["key", "cell", "found"], True),
        "DICTIMAX": (2, 3, ["dict", "keylen"], ["key", "value", "found"], True),
        "DICTIMAXREF": (2, 3, ["dict", "keylen"], ["key", "cell", "found"], True),

        # Dictionary remove min/max operations
        "DICTREMMIN": (2, 4, ["dict", "keylen"], ["dict'", "key", "value", "found"], True),
        "DICTREMMINREF": (2, 4, ["dict", "keylen"], ["dict'", "key", "cell", "found"], True),
        "DICTREMMAX": (2, 4, ["dict", "keylen"], ["dict'", "key", "value", "found"], True),
        "DICTREMMAXREF": (2, 4, ["dict", "keylen"], ["dict'", "key", "cell", "found"], True),
        "DICTUREMMIN": (2, 4, ["dict", "keylen"], ["dict'", "key", "value", "found"], True),
        "DICTUREMMINREF": (2, 4, ["dict", "keylen"], ["dict'", "key", "cell", "found"], True),
        "DICTUREMMAX": (2, 4, ["dict", "keylen"], ["dict'", "key", "value", "found"], True),
        "DICTUREMMAXREF": (2, 4, ["dict", "keylen"], ["dict'", "key", "cell", "found"], True),
        "DICTIREMMIN": (2, 4, ["dict", "keylen"], ["dict'", "key", "value", "found"], True),
        "DICTIREMMINREF": (2, 4, ["dict", "keylen"], ["dict'", "key", "cell", "found"], True),
        "DICTIREMMAX": (2, 4, ["dict", "keylen"], ["dict'", "key", "value", "found"], True),
        "DICTIREMMAXREF": (2, 4, ["dict", "keylen"], ["dict'", "key", "cell", "found"], True),

        # Dictionary empty check
        "DICTEMPTY": (1, 1, ["dict"], ["empty"], True),

        # Prefix dictionary operations
        "PFXDICTGET": (3, 3, ["key", "dict", "keylen"], ["prefix", "value", "found"], True),
        "PFXDICTSET": (4, 2, ["value", "key", "dict", "keylen"], ["dict'", "success"], True),  # x k D n - D' -1 or D 0
        "PFXDICTDEL": (3, 2, ["key", "dict", "keylen"], ["dict'", "found"], True),

        # Arithmetic operations with modulo (quotient + remainder)
        # These return both quotient and remainder: x y - q r
        "MULRSHIFTR_MOD": (2, 2, ["x", "y"], ["q", "r"], False),  # x y - q=round(x*y/2^z) r=remainder
        "MULRSHIFTC_MOD": (2, 2, ["x", "y"], ["q", "r"], False),  # x y - q=ceil(x*y/2^z) r=remainder
        "MULRSHIFT_MOD": (2, 2, ["x", "y"], ["q", "r"], False),   # x y - q=floor(x*y/2^z) r=remainder
    }

    # Manual interval overrides for dynamic instructions
    # Format: mnemonic -> (min_inputs, max_inputs, min_outputs, max_outputs)
    # Use None for unbounded/unknown upper bounds.
    MANUAL_STACK_EFFECT_RANGES: Dict[str, Tuple[int, Optional[int], int, Optional[int]]] = {
        "PICK": (1, None, 1, 1),  # Pops index, pushes duplicate
        "ROLL": (1, None, 0, 0),  # Pops index, rotates existing items
        "ROLLX": (1, None, 0, 0),
        "-ROLL": (1, None, 0, 0),
        "-ROLLX": (1, None, 0, 0),
        "DROPX": (1, None, 0, 0),  # Pops index, drops that many items
        "ONLYX": (1, None, 0, 0),
        "ONLYTOPX": (1, None, 0, 0),
        "BLKDROP": (0, None, 0, 0),
        "BLKDROP2": (0, None, 0, 0),
        "BLKPUSH": (0, 0, 0, None),
        "BLKSWAP": (0, 0, 0, 0),
        "BLKSWX": (1, None, 0, 0),
        "CALLREF": (0, None, 0, None),
    }

    def __init__(self):
        self.effects: Dict[str, StackEffect] = {}
        self._load_from_spec()
        self._apply_manual_supplements()

    @staticmethod
    def _sum_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None or b is None:
            return None
        return a + b

    @staticmethod
    def _min_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    @staticmethod
    def _max_optional_pair(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None or b is None:
            return None
        return max(a, b)

    @staticmethod
    def _max_optional(values: List[Optional[int]]) -> Optional[int]:
        if any(v is None for v in values):
            return None
        return max(values) if values else 0

    def _stack_entry_range(
        self,
        entry: object,
        name_prefix: str,
        name_index: int,
        operands: Optional[Dict[str, int]] = None,
    ) -> Tuple[int, Optional[int], List[str], bool]:
        if not isinstance(entry, dict):
            return 1, 1, [f"{name_prefix}{name_index}"], False

        entry_type = entry.get("type")
        if entry_type == "array":
            length_var = entry.get("length_var")
            if operands and length_var in operands:
                try:
                    length = int(operands[length_var])
                except (TypeError, ValueError):
                    length = None
                if length is not None and length >= 0:
                    base_name = None
                    array_entry = entry.get("array_entry", []) or []
                    if array_entry and isinstance(array_entry[0], dict):
                        base_name = array_entry[0].get("name")
                    names = [f"{base_name}{i}" for i in range(length)] if base_name else []
                    return length, length, names, False
            # Variable number of stack items
            return 0, None, [], True

        if entry_type == "conditional":
            # Stack effect depends on runtime condition
            branch_ranges: List[Tuple[int, Optional[int], List[str], bool]] = []
            for case in entry.get("match", []) or []:
                case_stack = case.get("stack", []) or []
                branch_ranges.append(self._stack_list_range(case_stack, name_prefix, operands))
            if "else" in entry:
                else_stack = entry.get("else", []) or []
                branch_ranges.append(self._stack_list_range(else_stack, name_prefix, operands))

            if not branch_ranges:
                return 0, 0, [], True

            min_count = min(r[0] for r in branch_ranges)
            max_count = self._max_optional([r[1] for r in branch_ranges])
            return min_count, max_count, [], True

        name = entry.get("name") or f"{name_prefix}{name_index}"
        return 1, 1, [name], False

    def _stack_list_range(
        self,
        stack_list: List[object],
        name_prefix: str,
        operands: Optional[Dict[str, int]] = None,
    ) -> Tuple[int, Optional[int], List[str], bool]:
        min_count = 0
        max_count: Optional[int] = 0
        names: List[str] = []
        is_dynamic = False
        name_index = 0

        for entry in stack_list or []:
            entry_min, entry_max, entry_names, entry_dynamic = self._stack_entry_range(
                entry, name_prefix, name_index, operands
            )
            min_count += entry_min
            max_count = self._sum_optional(max_count, entry_max)
            if entry_names:
                names.extend(entry_names)
                name_index += len(entry_names)
            is_dynamic = is_dynamic or entry_dynamic

        return min_count, max_count, names, is_dynamic

    def _effect_from_value_flow(
        self,
        value_flow: Dict[str, object],
        operands: Optional[Dict[str, int]] = None,
    ) -> StackEffect:
        inputs = value_flow.get("inputs", {})
        outputs = value_flow.get("outputs", {})
        stack_inputs = inputs.get("stack", [])
        stack_outputs = outputs.get("stack", [])

        min_inputs, max_inputs, input_names, in_dynamic = self._stack_list_range(
            stack_inputs, "in", operands
        )
        min_outputs, max_outputs, output_names, out_dynamic = self._stack_list_range(
            stack_outputs, "out", operands
        )

        is_dynamic = in_dynamic or out_dynamic
        if max_inputs is None or max_outputs is None:
            is_dynamic = True
        if (max_inputs is not None and min_inputs != max_inputs) or (
            max_outputs is not None and min_outputs != max_outputs
        ):
            is_dynamic = True

        return StackEffect(
            inputs=min_inputs,
            outputs=min_outputs,
            input_names=input_names,
            output_names=output_names,
            is_dynamic=is_dynamic,
            min_inputs=min_inputs,
            max_inputs=max_inputs,
            min_outputs=min_outputs,
            max_outputs=max_outputs,
        )

    def _load_from_spec(self):
        """Load stack effects from cp0.json specification"""
        # Find cp0.json relative to this file
        spec_path = Path(__file__).parent.parent / "spec" / "cp0.json"
        signature_path = Path(__file__).parent.parent / "disassembler" / "stack-signatures-data.json"
        instruction_table_path = Path(__file__).parent.parent / "disassembler" / "instruction_table.json"

        if not spec_path.exists():
            if STRICT_MODE:
                raise RuntimeError(f"cp0.json not found at {spec_path} in strict mode")
            logger.warning(
                f"cp0.json not found at {spec_path}. Stack simulation will be limited. "
                f"Set TasmScan_STRICT=1 to fail fast on missing specs."
            )
            return

        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                spec = json.load(f)
            with open(signature_path, 'r', encoding='utf-8') as f:
                signatures = json.load(f)
            with open(instruction_table_path, 'r', encoding='utf-8') as f:
                instruction_table = json.load(f)

            alias_map: Dict[str, str] = {}
            alias_operands: Dict[str, Dict[str, int]] = {}
            instr_by_mnemonic: Dict[str, Dict[str, object]] = {}
            instruction_kind: Dict[str, str] = {}
            instruction_args: Dict[str, List[Dict[str, object]]] = {}

            for item in instruction_table.get("instructions", []):
                name = item.get("name")
                if name:
                    instruction_kind[name] = item.get("kind", "")
                    instruction_args[name] = item.get("args", []) or []

            # Parse each instruction
            for instr in spec.get("instructions", []):
                mnemonic = instr.get("mnemonic")
                if not mnemonic:
                    continue
                instr_by_mnemonic[mnemonic] = instr
                alias_of = instr.get("alias_of")

                # Extract value_flow information
                value_flow = instr.get("value_flow", {})
                if not value_flow:
                    if alias_of:
                        alias_map[mnemonic] = alias_of
                    continue

                effect = self._effect_from_value_flow(value_flow)

                self.effects[mnemonic] = effect

            for alias in spec.get("aliases", []):
                mnemonic = alias.get("mnemonic")
                alias_of = alias.get("alias_of")
                if mnemonic and alias_of:
                    alias_map.setdefault(mnemonic, alias_of)
                    operands = alias.get("operands")
                    if isinstance(operands, dict):
                        alias_operands[mnemonic] = operands

            if alias_map:
                def resolve_alias_effect(start: str) -> Optional[StackEffect]:
                    seen: Set[str] = set()
                    current = start
                    while current and current not in seen:
                        if current in self.effects:
                            return self.effects[current]
                        seen.add(current)
                        current = alias_map.get(current, "")
                    return None

                for alias, target in alias_map.items():
                    effect = resolve_alias_effect(target)
                    if effect:
                        operands = alias_operands.get(alias)
                        if operands:
                            target_instr = instr_by_mnemonic.get(target)
                            value_flow = target_instr.get("value_flow") if target_instr else None
                            if isinstance(value_flow, dict) and value_flow:
                                self.effects[alias] = self._effect_from_value_flow(value_flow, operands)
                                continue

                        self.effects[alias] = StackEffect(
                            inputs=effect.inputs,
                            outputs=effect.outputs,
                            input_names=list(effect.input_names),
                            output_names=list(effect.output_names),
                            is_dynamic=effect.is_dynamic,
                            min_inputs=effect.min_inputs,
                            max_inputs=effect.max_inputs,
                            min_outputs=effect.min_outputs,
                            max_outputs=effect.max_outputs,
                        )

            # Fallback: load stack-signatures-data.json for missing opcodes
            for mnemonic, signature in signatures.items():
                if mnemonic.startswith("$"):
                    continue
                if mnemonic in self.effects:
                    continue
                value_flow = {
                    "inputs": signature.get("inputs", {}),
                    "outputs": signature.get("outputs", {}),
                }
                self.effects[mnemonic] = self._effect_from_value_flow(value_flow)

            # Reconcile cp0 vs stack-signatures for simple opcodes (no immediates)
            for mnemonic, signature in signatures.items():
                if mnemonic.startswith("$"):
                    continue
                existing = self.effects.get(mnemonic)
                if not existing:
                    continue
                kind = instruction_kind.get(mnemonic, "")
                args = instruction_args.get(mnemonic, [])
                if kind != "simple" or args:
                    continue

                value_flow = {
                    "inputs": signature.get("inputs", {}),
                    "outputs": signature.get("outputs", {}),
                }
                sig_effect = self._effect_from_value_flow(value_flow)

                if (
                    sig_effect.min_inputs == existing.min_inputs
                    and sig_effect.max_inputs == existing.max_inputs
                    and sig_effect.min_outputs == existing.min_outputs
                    and sig_effect.max_outputs == existing.max_outputs
                ):
                    continue

                merged_min_inputs = min(existing.min_inputs, sig_effect.min_inputs)
                merged_max_inputs = self._max_optional_pair(existing.max_inputs, sig_effect.max_inputs)
                merged_min_outputs = min(existing.min_outputs, sig_effect.min_outputs)
                merged_max_outputs = self._max_optional_pair(existing.max_outputs, sig_effect.max_outputs)

                self.effects[mnemonic] = StackEffect(
                    inputs=merged_min_inputs,
                    outputs=merged_min_outputs,
                    input_names=list(existing.input_names),
                    output_names=list(existing.output_names),
                    is_dynamic=True,
                    min_inputs=merged_min_inputs,
                    max_inputs=merged_max_inputs,
                    min_outputs=merged_min_outputs,
                    max_outputs=merged_max_outputs,
                )

        except Exception as e:
            if STRICT_MODE:
                raise RuntimeError(f"Failed to load cp0.json in strict mode: {e}")
            logger.warning(
                f"Failed to load cp0.json: {e}. Stack simulation will be limited. "
                f"Set TasmScan_STRICT=1 to fail fast on missing specs."
            )

    def _apply_manual_supplements(self):
        """Apply manual stack effect definitions for opcodes missing from spec"""
        added_count = 0
        updated_count = 0
        dynamic_count = 0

        for mnemonic, (inputs, outputs, input_names, output_names, is_dynamic) in self.MANUAL_STACK_EFFECTS.items():
            range_override = self.MANUAL_STACK_EFFECT_RANGES.get(mnemonic)
            if range_override:
                min_inputs, max_inputs, min_outputs, max_outputs = range_override
            else:
                min_inputs, min_outputs = inputs, outputs
                if is_dynamic:
                    max_inputs = None
                    max_outputs = None
                else:
                    max_inputs = inputs
                    max_outputs = outputs

            effect = StackEffect(
                inputs=inputs,
                outputs=outputs,
                input_names=input_names,
                output_names=output_names,
                is_dynamic=is_dynamic,
                min_inputs=min_inputs,
                max_inputs=max_inputs,
                min_outputs=min_outputs,
                max_outputs=max_outputs,
            )

            if mnemonic in self.effects:
                # Only update if the existing effect is empty (0 in, 0 out)
                existing = self.effects[mnemonic]
                if existing.inputs == 0 and existing.outputs == 0:
                    self.effects[mnemonic] = effect
                    updated_count += 1
                elif existing.is_dynamic and not effect.is_dynamic:
                    # Manual definition explicitly marks as non-dynamic; prefer it.
                    self.effects[mnemonic] = effect
                    updated_count += 1
            else:
                # Add new effect
                self.effects[mnemonic] = effect
                added_count += 1

            if is_dynamic:
                dynamic_count += 1

    def get_effect(self, opcode: str) -> Optional[StackEffect]:
        """
        Get stack effect for a given opcode.

        Args:
            opcode: The mnemonic (e.g., "ADD", "SWAP", "LDU")

        Returns:
            StackEffect if known, None otherwise
        """
        # Try direct lookup first
        effect = self.effects.get(opcode)
        if effect:
            return effect

        # Try alias mapping if direct lookup fails
        canonical_name = self.INSTRUCTION_ALIASES.get(opcode)
        if canonical_name:
            effect = self.effects.get(canonical_name)
            if effect:
                return effect

        # Log unknown opcode (only once per opcode, with capacity limit)
        with _unknown_opcodes_lock:
            if opcode not in _unknown_opcodes_encountered:
                if len(_unknown_opcodes_encountered) < MAX_UNKNOWN_OPCODES_TRACKED:
                    _unknown_opcodes_encountered.add(opcode)
                    logger.warning(f"Unknown opcode '{opcode}' - stack simulation may be imprecise")
                # Beyond limit, silently skip to prevent unbounded memory growth

        return None

    def has_effect(self, opcode: str) -> bool:
        """Check if we have stack effect information for an opcode"""
        # Check direct lookup first
        if opcode in self.effects:
            return True
        # Check via alias mapping
        canonical_name = self.INSTRUCTION_ALIASES.get(opcode)
        return canonical_name in self.effects if canonical_name else False

    def get_coverage_stats(self) -> Dict[str, int]:
        """Get statistics about opcode coverage"""
        return {
            "total_opcodes": len(self.effects),
            "with_stack_effects": sum(1 for e in self.effects.values() if e.inputs > 0 or e.outputs > 0),
            "no_stack_effect": sum(1 for e in self.effects.values() if e.inputs == 0 and e.outputs == 0),
        }

    @staticmethod
    def get_unknown_opcodes() -> Set[str]:
        """Return set of unknown opcodes encountered during analysis."""
        with _unknown_opcodes_lock:
            return _unknown_opcodes_encountered.copy()

    @staticmethod
    def clear_unknown_opcodes() -> None:
        """Clear the set of encountered unknown opcodes.

        Call this between analysis sessions to prevent memory accumulation
        in long-running processes. The set is also bounded by
        MAX_UNKNOWN_OPCODES_TRACKED (default 1000) as a safety limit.
        """
        with _unknown_opcodes_lock:
            _unknown_opcodes_encountered.clear()


@lru_cache(maxsize=1)
def get_stack_effect_db() -> StackEffectDatabase:
    """Get or create the global StackEffectDatabase instance (thread-safe via lru_cache)"""
    return StackEffectDatabase()


def get_stack_effect(opcode: str) -> Optional[StackEffect]:
    """
    Convenience function to get stack effect for an opcode.

    Args:
        opcode: The mnemonic (e.g., "ADD", "SWAP")

    Returns:
        StackEffect if known, None otherwise

    Example:
        effect = get_stack_effect("ADD")
        if effect:
            print(f"ADD pops {effect.inputs} and pushes {effect.outputs}")
    """
    return get_stack_effect_db().get_effect(opcode)
