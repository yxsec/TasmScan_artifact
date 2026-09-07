"""
TASIR - TVM Intermediate Representation for Security Analysis

This module defines a security-oriented IR for the TON Virtual Machine,
designed to support precise taint analysis and vulnerability detection.

Key innovations:
1. First-class Continuation modeling with SaveList semantics
2. Layered type system (TVMType, TVMLocation, TVMInstruction)
3. Integration with existing stack_effects and opcode_mappings

Academic contribution: This IR enables precise cross-continuation
data flow tracking, addressing the cross-continuation limitation in path-sensitive
analysis.
"""

from __future__ import annotations

from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import copy as copy_module

if TYPE_CHECKING:
    from .dataflow.types import DataFlowGraph, ValueSource
    from ..analyzer.facts import AnalysisFacts


# =============================================================================
# 1. TVMType - TVM Runtime Value Types
# =============================================================================

class TVMType(Enum):
    """
    TVM runtime value types.

    Represents the fundamental data types in the TON Virtual Machine.
    Each type has specific semantics and constraints in TVM execution.
    """

    INT = auto()        # 257-bit signed integer
    CELL = auto()       # Cell reference (up to 1023 bits data, 4 refs)
    SLICE = auto()      # Slice - Cell view with read cursor
    BUILDER = auto()    # Cell builder for constructing cells
    CONT = auto()       # Continuation - first-class control flow
    TUPLE = auto()      # Polymorphic tuple (up to 255 elements)
    NULL = auto()       # Null value
    DICT = auto()       # Dictionary (= Cell + key_bits metadata)
    UNKNOWN = auto()    # Analysis placeholder for unknown types


# =============================================================================
# 2. TVMTypeConstraint - Type Constraints for Analysis
# =============================================================================

@dataclass(frozen=True)
class TVMTypeConstraint:
    """
    Type constraint for abstract interpretation.

    Extends base TVMType with additional constraints useful for
    static analysis, such as bit width bounds and reference counts.

    Attributes:
        base_type: The fundamental TVM type
        bit_width: Optional constraint on integer bit width
        ref_count: Optional constraint on cell reference count
        tuple_len: Optional constraint on tuple length
    """

    base_type: TVMType
    bit_width: Optional[int] = None
    ref_count: Optional[int] = None
    tuple_len: Optional[int] = None


# =============================================================================
# 3. TVMLocation Hierarchy - Storage Location Abstractions
# =============================================================================

@dataclass(frozen=True)
class TVMLocation(ABC):
    """
    Abstract base class for TVM storage locations.

    Represents any location where a value can be stored or read from
    in the TVM execution model. Used for data flow tracking.
    """
    pass


@dataclass(frozen=True)
class StackLocation(TVMLocation):
    """
    Stack position in TVM.

    TVM uses a stack-based execution model. Stack positions are
    indexed from the top, with 0 being the top of stack (TOS).

    Attributes:
        depth: Stack depth (0 = TOS, 1 = second element, etc.)
    """

    depth: int


@dataclass(frozen=True)
class RegisterLocation(TVMLocation):
    """
    Control register c0-c7 in TVM.

    TVM has 8 control registers with specific semantics:
    - c0: next continuation (continuation after current)
    - c1: return continuation (for RETALT)
    - c2: exception handler continuation
    - c3: code dictionary (current smart contract code)
    - c4: persistent storage root cell
    - c5: output actions list
    - c6: reserved (unused in standard TVM)
    - c7: context tuple (blockchain context)

    Attributes:
        index: Register index (0-7)
    """

    index: int

    @property
    def semantic(self) -> str:
        """Get the semantic meaning of this control register."""
        SEMANTICS = {
            0: "next_continuation",
            1: "return_continuation",
            2: "exception_handler",
            3: "code_dictionary",
            4: "persistent_storage",
            5: "output_actions",
            7: "context_tuple",
        }
        return SEMANTICS.get(self.index, f"c{self.index}")

    @property
    def security_level(self) -> str:
        """Get the security classification of this control register.

        Differentiates between trusted data sources and sensitive operation
        targets, enabling detectors to prioritize analysis of writes to
        security-critical registers.

        Returns:
            "write_critical" for c5 (output_actions) - controls message sends
            "storage_critical" for c4 (persistent_storage) - controls state
            "read_trusted" for c7 (context_tuple) - trusted blockchain context
            "control_flow" for c0-c3 (continuations/exception handling)
            "neutral" for other registers
        """
        SECURITY_LEVELS = {
            0: "control_flow",      # c0: next continuation
            1: "control_flow",      # c1: return continuation
            2: "control_flow",      # c2: exception handler
            3: "control_flow",      # c3: code dictionary
            4: "storage_critical",  # c4: persistent storage root cell
            5: "write_critical",    # c5: output actions list
            7: "read_trusted",      # c7: context tuple (blockchain context)
        }
        return SECURITY_LEVELS.get(self.index, "neutral")


@dataclass(frozen=True)
class GlobalLocation(TVMLocation):
    """
    Global variable g0-g254 in TVM.

    TVM supports up to 255 global variables that persist
    across continuation switches within the same execution.

    Attributes:
        index: Global variable index (0-254)
    """

    index: int


# =============================================================================
# 4. SaveList - Continuation Saved Register State
# =============================================================================

@dataclass
class SaveList:
    """
    Saved register state for continuation.

    When a continuation is created, it can capture (save) the current
    values of certain control registers. When the continuation is
    executed, these saved values are restored.

    This is a core concept for understanding TVM control flow,
    especially for exception handling and return continuations.

    Attributes:
        registers: Backward-compatible primary saved value per register.
        register_candidates: Disjunctive saved candidates per register.
    """

    registers: Dict[int, TVMAbstractValue] = field(default_factory=dict)
    register_candidates: Dict[int, List[TVMAbstractValue]] = field(default_factory=dict)

    @staticmethod
    def _value_key(value: TVMAbstractValue) -> Tuple[Any, Any, bool, bool, str]:
        """Stable key for de-duplicating candidate abstract values."""
        return (
            value.definition_site,
            value.source,
            bool(value.tainted),
            bool(value.checked),
            repr(sorted(value.metadata.items())) if value.metadata else "",
        )

    @staticmethod
    def _widen_candidates(candidates: List[TVMAbstractValue]) -> TVMAbstractValue:
        """Widen many candidates into a single conservative abstract value."""
        definition_sites = sorted(
            {
                candidate.definition_site
                for candidate in candidates
                if candidate.definition_site is not None
            }
        )
        tainted = any(candidate.tainted for candidate in candidates)
        checked = all(candidate.checked for candidate in candidates if candidate.tainted)
        metadata: Dict[str, Any] = {"savelist_widened": True}
        if definition_sites:
            metadata["candidate_definition_sites"] = definition_sites
        return TVMAbstractValue(
            definition_site=definition_sites[0] if len(definition_sites) == 1 else None,
            tainted=tainted,
            checked=checked if tainted else False,
            source="savelist_widened",
            metadata=metadata,
        )

    def save(
        self,
        reg_index: int,
        value: TVMAbstractValue,
        *,
        max_candidates: int = 4,
    ) -> SaveList:
        """
        Create a new SaveList with an additional saved register.

        Args:
            reg_index: Register index to save
            value: Abstract value to save
            max_candidates: Candidate cap per register (widen above this bound)

        Returns:
            New SaveList with the register saved
        """
        new_regs = dict(self.registers)
        new_candidates: Dict[int, List[TVMAbstractValue]] = {
            reg: [candidate.copy() for candidate in values]
            for reg, values in self.register_candidates.items()
        }

        existing = new_candidates.get(reg_index)
        if existing is None:
            if reg_index in new_regs:
                existing = [new_regs[reg_index].copy()]
            else:
                existing = []

        incoming = value.copy()
        existing.append(incoming)

        deduped: List[TVMAbstractValue] = []
        seen = set()
        for candidate in existing:
            key = self._value_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)

        if max_candidates > 0 and len(deduped) > max_candidates:
            deduped = [self._widen_candidates(deduped)]

        primary = deduped[-1] if deduped else incoming
        new_regs[reg_index] = primary
        new_candidates[reg_index] = deduped
        return SaveList(registers=new_regs, register_candidates=new_candidates)

    def restore(self, reg_index: int) -> Optional[TVMAbstractValue]:
        """
        Get the saved value for a register.

        Args:
            reg_index: Register index to restore

        Returns:
            Saved abstract value, or None if not saved
        """
        return self.registers.get(reg_index)

    def restore_all(self, reg_index: int) -> List[TVMAbstractValue]:
        """Get all disjunctive saved values for a register."""
        candidates = self.register_candidates.get(reg_index)
        if candidates is not None:
            return [candidate.copy() for candidate in candidates]
        single = self.registers.get(reg_index)
        return [single.copy()] if single is not None else []

    def iter_saved_values(self) -> List[Tuple[int, List[TVMAbstractValue]]]:
        """Return all registers with their disjunctive candidates."""
        reg_indices = set(self.registers.keys()) | set(self.register_candidates.keys())
        result: List[Tuple[int, List[TVMAbstractValue]]] = []
        for reg_idx in sorted(reg_indices):
            values = self.restore_all(reg_idx)
            if values:
                result.append((reg_idx, values))
        return result

    def has_saved_values(self) -> bool:
        """Return True when any register has at least one saved candidate."""
        return any(self.restore_all(reg_idx) for reg_idx in set(self.registers) | set(self.register_candidates))

    def copy(self) -> SaveList:
        """Create a deep copy of this SaveList.

        Each TVMAbstractValue in the registers is deep-copied to prevent
        state pollution from shared mutable metadata dictionaries.
        """
        return SaveList(
            registers={k: v.copy() for k, v in self.registers.items()},
            register_candidates={
                k: [candidate.copy() for candidate in values]
                for k, values in self.register_candidates.items()
            },
        )


# =============================================================================
# 5. TVMAbstractValue - Abstract Value for Data Flow Analysis
# =============================================================================

@dataclass
class TVMAbstractValue:
    """
    Abstract value for data flow analysis.

    Represents a value during static analysis with additional
    metadata for taint tracking and security analysis.

    Attributes:
        type_constraint: Optional type constraint for this value
        definition_site: Instruction index where this value was defined
        tainted: Whether this value is tainted (from untrusted source)
        checked: Whether this value has been validated/sanitized
        source: Origin description (e.g., "message_sender", "storage")
        metadata: Additional analysis-specific metadata
    """

    type_constraint: Optional[TVMTypeConstraint] = None
    definition_site: Optional[int] = None
    tainted: bool = False
    checked: bool = False
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> TVMAbstractValue:
        """Create a deep copy of this abstract value."""
        return TVMAbstractValue(
            type_constraint=self.type_constraint,
            definition_site=self.definition_site,
            tainted=self.tainted,
            checked=self.checked,
            source=self.source,
            metadata=copy_module.deepcopy(self.metadata),
        )


# =============================================================================
# 6. ContinuationDescriptor - Complete Continuation Representation
# =============================================================================

@dataclass
class ContinuationDescriptor:
    """
    Complete continuation representation - core innovation for TVM IR.

    Continuations are first-class values in TVM that represent
    suspended computations. This descriptor captures all aspects
    of a continuation needed for precise security analysis.

    Key innovation: By explicitly modeling SaveList semantics,
    we can track data flow across continuation boundaries,
    addressing the cross-continuation limitation in path-sensitive analysis.

    Attributes:
        code_ref: BasicBlock ID or external reference to code
        savelist: Saved register state to restore on execution
        nargs: Expected argument count (-1 for varargs)
        captured_stack_depth: For closure semantics
        cont_type: Type of continuation (ordinary, exceptional, etc.)
        gas_limit: Optional gas limit for this continuation
        is_return_cont: Whether this is a return continuation (c1)
    """

    code_ref: Union[int, str]
    savelist: SaveList = field(default_factory=SaveList)
    nargs: int = -1
    captured_stack_depth: Optional[int] = None
    cont_type: str = "ordinary"  # ordinary | exceptional | while | until | repeat
    gas_limit: Optional[int] = None
    is_return_cont: bool = False

    def with_savelist(self, savelist: SaveList) -> ContinuationDescriptor:
        """
        Create a new ContinuationDescriptor with updated savelist.

        Args:
            savelist: New SaveList to use

        Returns:
            New ContinuationDescriptor with updated savelist
        """
        return ContinuationDescriptor(
            code_ref=self.code_ref,
            savelist=savelist,
            nargs=self.nargs,
            captured_stack_depth=self.captured_stack_depth,
            cont_type=self.cont_type,
            gas_limit=self.gas_limit,
            is_return_cont=self.is_return_cont,
        )


# =============================================================================
# 7. SemanticLabel - Domain-Specific Semantic Labels
# =============================================================================

class SemanticLabel(Enum):
    """High-level semantic labels for instruction sequences.

    These labels provide domain-specific meaning to TVM instruction patterns,
    enabling detectors to reason at a higher abstraction level.
    """
    # Message handling
    MESSAGE_RECEIVE = "message_receive"        # Entry point for incoming messages
    MESSAGE_SEND = "message_send"              # SENDRAWMSG, SENDMSG
    MESSAGE_PARSE = "message_parse"            # Parsing message body/header

    # Value transfer
    TOKEN_TRANSFER = "token_transfer"          # Send with value (SENDRAWMSG with value)
    RESERVE_BALANCE = "reserve_balance"        # RAWRESERVE operations

    # Authentication / Authorization
    AUTH_CHECK = "auth_check"                  # Sender verification pattern
    BOUNCED_CHECK = "bounced_check"            # Bounced message flag check
    SIGNATURE_VERIFY = "signature_verify"      # CHKSIGNU, CHKSIGNS

    # State management
    STORAGE_READ = "storage_read"              # PUSHCTR c4 + CTOS
    STORAGE_WRITE = "storage_write"            # ... + POPCTR c4

    # Gas/Accept
    GAS_ACCEPT = "gas_accept"                  # ACCEPT, SETGASLIMIT

    # Randomness
    RANDOM_GENERATE = "random_generate"        # RANDU256, RAND
    RANDOM_SEED = "random_seed"                # SETRAND, ADDRAND

    # Control flow
    DICT_DISPATCH = "dict_dispatch"            # Dictionary-based method dispatch
    EXCEPTION_HANDLER = "exception_handler"    # TRY/TRYARGS setup

    # Cryptographic
    HASH_COMPUTE = "hash_compute"              # SHA256, HASHCU, etc.

    # Arithmetic
    QUIET_ARITHMETIC = "quiet_arithmetic"      # QADD, QSUB, QMUL, QDIV (no overflow exceptions)

    # Generic
    GUARD = "guard"                            # Any value-checking guard (IF+THROW pattern)
    UNKNOWN = "unknown"                        # No semantic label assigned


# =============================================================================
# 8. InstructionKind - High-Level Instruction Classification
# =============================================================================

class InstructionKind(Enum):
    """
    High-level instruction classification.

    Groups TVM opcodes into semantic categories for easier
    pattern matching and analysis rule application.
    """

    # Pure computation
    ARITHMETIC = auto()     # ADD, SUB, MUL, DIV, MOD, etc.
    BITWISE = auto()        # AND, OR, XOR, NOT, LSHIFT, RSHIFT
    COMPARISON = auto()     # LESS, EQUAL, GREATER, etc.

    # Stack operations
    STACK_PUSH = auto()     # PUSHINT, PUSHCONT, PUSHREF, etc.
    STACK_POP = auto()      # DROP, NIP, etc.
    STACK_SHUFFLE = auto()  # XCHG, SWAP, ROLL, etc.

    # Cell operations
    CELL_LOAD = auto()      # LDREF, LDSLICE, etc.
    CELL_STORE = auto()     # STREF, STSLICE, etc.
    CELL_CONVERT = auto()   # CTOS, ENDC, etc.

    # Continuation control (core innovation)
    CONT_CREATE = auto()    # PUSHCONT, BLESS, etc.
    CONT_CALL = auto()      # EXECUTE, CALLCC, etc.
    CONT_JUMP = auto()      # JMPX, IFELSE, etc.
    CONT_RETURN = auto()    # RET, RETALT, etc.
    CONT_SAVE = auto()      # SAVE, SAVEALT, SAVEBOTH, POPSAVE
    CONT_COMPOSE = auto()   # ATEXIT, COMPOS, etc.

    # Control flow
    BRANCH_CONDITIONAL = auto()   # IF, IFNOT, IFELSE, etc.
    BRANCH_UNCONDITIONAL = auto() # JMP (unconditional)
    LOOP = auto()                 # WHILE, UNTIL, REPEAT, etc.

    # Exceptions
    THROW = auto()          # THROW, THROWIF, etc.
    TRY_CATCH = auto()      # TRY, CATCH

    # State access
    REGISTER_LOAD = auto()  # PUSH c0-c7
    REGISTER_STORE = auto() # POP c0-c7
    GLOBAL_LOAD = auto()    # GETGLOB
    GLOBAL_STORE = auto()   # SETGLOB

    # Blockchain actions
    SEND_MESSAGE = auto()   # SENDRAWMSG
    RESERVE = auto()        # RAWRESERVE
    SET_CODE = auto()       # SETCODE

    # Gas management
    ACCEPT = auto()         # ACCEPT
    COMMIT = auto()         # COMMIT

    # Dictionary operations
    DICT_GET = auto()       # DICTGET, DICTIGET, etc.
    DICT_SET = auto()       # DICTSET, DICTISET, etc.
    DICT_DELETE = auto()    # DICTDEL, DICTIDEL, etc.

    # Tuple operations
    TUPLE_CREATE = auto()   # TUPLE, NIL, CONS, etc.
    TUPLE_ACCESS = auto()   # FIRST, SECOND, INDEX, etc.

    # Other
    NOP = auto()            # NOP
    DEBUG = auto()          # DEBUG, DUMP, etc.
    UNKNOWN = auto()        # Unknown/unsupported opcodes


# =============================================================================
# 9. TVMInstruction - Single TASIR Instruction
# =============================================================================

@dataclass
class TVMInstruction:
    """
    Single TASIR instruction.

    Represents one operation in the intermediate representation,
    with full metadata for data flow and security analysis.

    Attributes:
        index: Unique instruction index in the function
        kind: High-level classification of the instruction
        opcode: Original TVM opcode name
        inputs: List of input locations (stack, registers, etc.)
        outputs: List of output locations
        immediates: Immediate values (integers, strings, continuations)
        continuation_refs: References to continuation blocks
        stack_effect: Integration with existing stack_effects.py
        taint_transfer: How taint flows through this instruction
        is_sensitive: Whether this is a security-sensitive operation
        is_taint_source: Whether this instruction introduces taint
        original_args: Original instruction arguments for reference
    """

    index: int
    kind: InstructionKind
    opcode: str

    inputs: List[TVMLocation] = field(default_factory=list)
    outputs: List[TVMLocation] = field(default_factory=list)
    immediates: List[Union[int, str, ContinuationDescriptor]] = field(default_factory=list)

    continuation_refs: List[str] = field(default_factory=list)

    # Integration with existing stack_effects.py
    stack_effect: Optional[Any] = None

    # Security attributes
    taint_transfer: str = "propagate"  # propagate | source | sink | sanitize | check
    is_sensitive: bool = False
    is_taint_source: bool = False

    # Original instruction reference
    original_args: List[Any] = field(default_factory=list)

    # Semantic labels for high-level pattern matching
    semantic_labels: List[SemanticLabel] = field(default_factory=list)


# =============================================================================
# 10. BlockEdge - CFG Edge
# =============================================================================

@dataclass
class BlockEdge:
    """
    Control Flow Graph edge.

    Represents a transition between basic blocks, with metadata
    about the nature of the transition.

    Attributes:
        source_block: Source basic block ID
        target_block: Target basic block ID
        edge_kind: Type of edge (fallthrough, branch, call, return, exception)
        condition: Optional condition for conditional branches
        continuation_ref: Optional reference to continuation descriptor
        restored_registers: List of register indices restored on this edge
    """

    source_block: int
    target_block: int
    edge_kind: str  # fallthrough | branch | call | return | exception
    condition: Optional[str] = None
    continuation_ref: Optional[str] = None
    restored_registers: Optional[List[int]] = None


# =============================================================================
# 11. PhiNode - SSA Phi Node (Optional)
# =============================================================================

@dataclass
class PhiNode:
    """
    SSA phi node for advanced analysis.

    Used when converting to SSA form for more precise data flow
    analysis. Merges values from different predecessor blocks.

    Attributes:
        target: Location where merged value is stored
        sources: Mapping from predecessor block ID to source location
    """

    target: TVMLocation
    sources: Dict[int, TVMLocation]  # predecessor_block_id -> location


# =============================================================================
# 12. TVMBasicBlock - Basic Block
# =============================================================================

@dataclass
class TVMBasicBlock:
    """
    TASIR basic block.

    A sequence of instructions with single entry and single exit.
    Forms the nodes of the control flow graph.

    Attributes:
        id: Unique block identifier within the function
        context_id: "main" or continuation_id this block belongs to
        instructions: List of instructions in this block
        predecessors: Incoming CFG edges
        successors: Outgoing CFG edges
        entry_stack_height: Stack height at block entry (if known)
        exit_stack_height: Stack height at block exit (if known)
        has_unknown_successor: Whether block has dynamic/unknown targets
        phi_nodes: SSA phi nodes at block entry
        metadata: Additional analysis metadata
    """

    id: int
    context_id: str  # "main" or continuation_id
    instructions: List[TVMInstruction] = field(default_factory=list)

    predecessors: List[BlockEdge] = field(default_factory=list)
    successors: List[BlockEdge] = field(default_factory=list)

    entry_stack_height: Optional[int] = None
    exit_stack_height: Optional[int] = None

    has_unknown_successor: bool = False

    phi_nodes: List[PhiNode] = field(default_factory=list)

    # Semantic labels aggregated from block instructions
    semantic_labels: List[SemanticLabel] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 13. FunctionSignature - Function Signature
# =============================================================================

@dataclass
class FunctionSignature:
    """
    Function signature for TASIR functions.

    Describes the expected inputs and outputs of a function,
    including type constraints when known.

    Attributes:
        min_stack_inputs: Minimum number of stack inputs required
        max_stack_inputs: Maximum number of stack inputs (None = unbounded)
        min_stack_outputs: Minimum number of stack outputs produced
        max_stack_outputs: Maximum number of stack outputs (None = unbounded)
        input_types: Type constraints for input values
        output_types: Type constraints for output values
    """

    min_stack_inputs: int = 0
    max_stack_inputs: Optional[int] = None
    min_stack_outputs: int = 0
    max_stack_outputs: Optional[int] = None

    input_types: List[TVMTypeConstraint] = field(default_factory=list)
    output_types: List[TVMTypeConstraint] = field(default_factory=list)


# =============================================================================
# 14. TVMFunction - Function/Method
# =============================================================================

@dataclass
class TVMFunction:
    """
    Function/method in TASIR.

    Represents a callable unit in the smart contract, typically
    corresponding to a method_id handler or internal function.

    Attributes:
        method_id: Method ID (0 for recv_internal, -1 for recv_external)
        name: Optional human-readable name
        signature: Function signature
        entry_block_id: ID of the entry basic block
        blocks: Mapping from block ID to basic block
        continuations: Local continuation descriptors
        is_recv_internal: Whether this handles internal messages
        is_recv_external: Whether this handles external messages
        metadata: Additional function metadata
    """

    method_id: int
    name: Optional[str] = None
    signature: FunctionSignature = field(default_factory=FunctionSignature)

    entry_block_id: int = 0
    blocks: Dict[int, TVMBasicBlock] = field(default_factory=dict)

    continuations: Dict[str, ContinuationDescriptor] = field(default_factory=dict)

    is_recv_internal: bool = False
    is_recv_external: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def all_blocks(self) -> List[TVMBasicBlock]:
        """Return all blocks in this function."""
        return list(self.blocks.values())

    def all_instructions(self) -> List[TVMInstruction]:
        """Return all function instructions sorted by instruction index."""
        instructions: List[TVMInstruction] = []
        for block in self.blocks.values():
            instructions.extend(block.instructions)
        return sorted(instructions, key=lambda i: i.index)


# =============================================================================
# 15. InterProceduralEdge / InterProceduralView - Cross-Function Analysis
# =============================================================================

@dataclass
class InterProceduralEdge:
    """Edge representing data flow across function boundaries."""
    caller_method_id: int
    callee_method_id: int
    caller_instruction_index: int  # The call instruction
    taint_propagated: bool = False  # Does taint flow across this edge?
    value_sources: List["ValueSource"] = field(default_factory=list)


@dataclass
class InterProceduralView:
    """Cross-function analysis view for the entire module.

    Provides a unified view of taint flow across function boundaries,
    enabling detection of vulnerabilities that span multiple functions.
    """
    # Cross-function taint edges
    edges: List[InterProceduralEdge] = field(default_factory=list)

    # Per-function taint summary: method_id -> {has_sender_read, has_send, has_accept, etc.}
    function_summaries: Dict[int, Dict[str, bool]] = field(default_factory=dict)

    # Cross-function taint paths: list of (source_method, source_inst, sink_method, sink_inst)
    taint_paths: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def get_tainted_callees(self, method_id: int) -> List[int]:
        """Get callee method IDs that receive tainted data from the given method."""
        return [e.callee_method_id for e in self.edges
                if e.caller_method_id == method_id and e.taint_propagated]

    def get_cross_function_taint_paths(self) -> List[Tuple[int, int, int, int]]:
        """Get all cross-function taint paths (source_method, source_inst, sink_method, sink_inst)."""
        return self.taint_paths

    def function_has_property(self, method_id: int, prop: str) -> bool:
        """Check if a function has a specific property (has_sender_read, has_send, etc.)."""
        summary = self.function_summaries.get(method_id, {})
        return summary.get(prop, False)

    def build_security_summaries(
        self, module: "TVMModule",
    ) -> Dict[str, "FunctionSecuritySummary"]:
        """Build security summaries for all functions in the module.

        Iterates every function in *module*, inspecting each instruction for
        security-relevant properties.  Caller/callee relationships are filled
        from ``module.call_graph``.

        Returns:
            Mapping from function identifier (``"method_<id>"``) to its
            :class:`FunctionSecuritySummary`.
        """
        summaries: Dict[str, "FunctionSecuritySummary"] = {}

        for method_id, func in module.functions.items():
            func_id = func.name or f"method_{method_id}"
            summary = FunctionSecuritySummary(function_id=func_id)

            for block in func.blocks.values():
                for inst in block.instructions:
                    # --- message send ---
                    if inst.kind == InstructionKind.SEND_MESSAGE:
                        summary.sends_message = True

                    # --- gas accept ---
                    if inst.kind == InstructionKind.ACCEPT:
                        summary.calls_accept = True

                    # --- sender read ---
                    if inst.opcode in ("LDMSGADDR", "LDMSGADDRQ"):
                        summary.reads_sender = True

                    # --- throws / guards ---
                    if inst.kind in (
                        InstructionKind.THROW,
                        InstructionKind.BRANCH_CONDITIONAL,
                    ):
                        summary.has_throws = True
                        # Recognise sender-guard via semantic label
                        if SemanticLabel.GUARD in getattr(
                            inst, "semantic_labels", []
                        ):
                            summary.has_sender_guard = True

                    # --- storage access via control register 4 ---
                    if inst.opcode == "PUSHCTR":
                        operands = getattr(inst, "original_args", [])
                        if operands and operands[0] == 4:
                            summary.reads_storage = True
                    if inst.opcode == "POPCTR":
                        operands = getattr(inst, "original_args", [])
                        if operands and operands[0] == 4:
                            summary.writes_storage = True

            summaries[func_id] = summary

        # --- fill caller / callee from call graph ---
        # module.call_graph stores (caller_method_id, callee_method_id) tuples
        method_id_to_func_id: Dict[int, str] = {}
        for mid, func in module.functions.items():
            method_id_to_func_id[mid] = func.name or f"method_{mid}"

        for caller_mid, callee_mid in module.call_graph:
            caller_fid = method_id_to_func_id.get(caller_mid)
            callee_fid = method_id_to_func_id.get(callee_mid)
            if caller_fid and caller_fid in summaries:
                summaries[caller_fid].callees.append(callee_fid or f"method_{callee_mid}")
            if callee_fid and callee_fid in summaries:
                summaries[callee_fid].callers.append(caller_fid or f"method_{caller_mid}")

        return summaries

    def has_accept_on_path_to(
        self,
        summaries: Dict[str, "FunctionSecuritySummary"],
        target_func: str,
    ) -> bool:
        """Check if ACCEPT exists on any call path leading to *target_func*.

        Walks the caller chain transitively (BFS) starting from
        *target_func* and returns ``True`` as soon as a function that
        calls ACCEPT is found.
        """
        visited: set = set()
        queue = deque([target_func])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            summary = summaries.get(current)
            if summary is None:
                continue
            if summary.calls_accept:
                return True
            queue.extend(summary.callers)
        return False


# =============================================================================
# 15b. FunctionSecuritySummary - Per-Function Security Properties
# =============================================================================

@dataclass
class FunctionSecuritySummary:
    """Security-relevant summary of a function's behavior.

    Enables precise cross-function security analysis by summarizing
    each function's security-relevant properties for caller/callee
    relationship reasoning.
    """

    function_id: str                                          # Function/continuation ID
    reads_sender: bool = False                                # Contains LDMSGADDR or equivalent
    has_sender_guard: bool = False                            # Has IF/THROW checking sender
    calls_accept: bool = False                                # Contains ACCEPT/SETGASLIMIT
    sends_message: bool = False                               # Contains SENDRAWMSG
    reads_storage: bool = False                               # Contains PUSHCTR c4 -> CTOS
    writes_storage: bool = False                              # Contains POPCTR c4
    has_throws: bool = False                                  # Contains THROW/THROWIF
    callees: List[str] = field(default_factory=list)          # Called function IDs
    callers: List[str] = field(default_factory=list)          # Functions that call this
    tainted_params: Set[int] = field(default_factory=set)     # Parameter positions receiving taint
    tainted_returns: Set[int] = field(default_factory=set)    # Return positions carrying taint


# =============================================================================
# 16. TVMModule - Complete TASIR Module
# =============================================================================

@dataclass
class TVMModule:
    """
    Complete TASIR module.

    Represents an entire smart contract's code in intermediate
    representation form, ready for security analysis.

    Attributes:
        functions: Mapping from method_id to function
        global_blocks: Blocks not belonging to any specific function
        call_graph: List of (caller_method_id, callee_method_id) edges
        cross_function_continuations: Continuations that span functions
        code_hash: Hash of the original contract code
        analysis_metadata: Metadata from analysis passes
    """

    functions: Dict[int, TVMFunction] = field(default_factory=dict)
    global_blocks: List[TVMBasicBlock] = field(default_factory=list)

    call_graph: List[Tuple[int, int]] = field(default_factory=list)

    cross_function_continuations: Dict[str, ContinuationDescriptor] = field(default_factory=dict)

    code_hash: Optional[str] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

    dataflow_graph: Optional["DataFlowGraph"] = None

    # Inter-procedural analysis view
    inter_procedural: Optional[InterProceduralView] = None

    # Reference to underlying AnalysisFacts for query delegation
    _facts: Optional["AnalysisFacts"] = field(default=None, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Unified query interface (delegates to AnalysisFacts)
    # -------------------------------------------------------------------------

    def events_of(self, event_type: str) -> list:
        """Get events of a specific type (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.events_of(event_type)
        return []

    def opcodes(self, opcode_name: str) -> list:
        """Get instructions with specific opcode (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.opcodes(opcode_name)
        return []

    def entry_block_ids(self) -> list:
        """Get entry block IDs (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.entry_block_ids()
        return []

    @property
    def basic_blocks(self) -> list:
        """Get basic blocks (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.basic_blocks
        return []

    @property
    def cfg_edges(self) -> list:
        """Get CFG edges (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.cfg_edges
        return []

    @property
    def stack_states(self) -> list:
        """Get stack states (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.stack_states
        return []

    @property
    def instructions(self) -> list:
        """Get raw instruction facts (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.instructions
        return []

    @property
    def summaries(self) -> list:
        """Get block summaries (delegates to AnalysisFacts)."""
        if self._facts:
            return self._facts.summaries
        return []

    # -------------------------------------------------------------------------
    # Original TVMModule methods
    # -------------------------------------------------------------------------

    def get_function(self, method_id: int) -> Optional[TVMFunction]:
        """
        Get a function by method ID.

        Args:
            method_id: The method ID to look up

        Returns:
            The function, or None if not found
        """
        return self.functions.get(method_id)

    def get_recv_internal(self) -> Optional[TVMFunction]:
        """
        Get the recv_internal handler (method_id 0).

        Returns:
            The recv_internal function, or None if not defined
        """
        return self.functions.get(0)

    def get_recv_external(self) -> Optional[TVMFunction]:
        """
        Get the recv_external handler (method_id -1).

        Returns:
            The recv_external function, or None if not defined
        """
        return self.functions.get(-1)

    def all_blocks(self) -> List[TVMBasicBlock]:
        """
        Get all basic blocks from all functions and global scope.

        Returns:
            List of all basic blocks in the module
        """
        blocks = list(self.global_blocks)
        for func in self.functions.values():
            blocks.extend(func.blocks.values())
        return blocks

    def all_instructions(self) -> List[TVMInstruction]:
        """
        Get all instructions in program order.

        Returns:
            List of all instructions, sorted by index
        """
        instructions = []
        for block in self.all_blocks():
            instructions.extend(block.instructions)
        return sorted(instructions, key=lambda i: i.index)

    def get_instructions_by_label(self, label: SemanticLabel) -> List[TVMInstruction]:
        """Get all instructions with a specific semantic label."""
        return [inst for inst in self.all_instructions() if label in inst.semantic_labels]

    def has_pattern(self, *labels: SemanticLabel) -> bool:
        """Check if the module contains instructions with all specified labels."""
        found: set = set()
        for inst in self.all_instructions():
            for label in inst.semantic_labels:
                found.add(label)
        return all(l in found for l in labels)

    def get_tainted_instructions(self) -> List[int]:
        """Get instruction indices that produce tainted values."""
        if not self.dataflow_graph:
            return []
        return [idx for idx, values in self.dataflow_graph.values.items()
                if any(v.tainted for v in values)]

    def is_instruction_tainted(self, instruction_index: int) -> bool:
        """Check if a specific instruction produces tainted values."""
        if not self.dataflow_graph:
            return False
        values = self.dataflow_graph.values.get(instruction_index, [])
        return any(v.tainted for v in values)

    def get_taint_flow(self, from_idx: int, to_idx: int) -> bool:
        """Check if taint flows from one instruction to another."""
        if not self.dataflow_graph:
            return False
        return (from_idx, to_idx) in self.dataflow_graph.tainted_propagation

    def get_guarded_taint_at(self, instruction_index: int) -> bool:
        """Check if tainted value at instruction has been guarded."""
        if not self.dataflow_graph:
            return False
        values = self.dataflow_graph.values.get(instruction_index, [])
        return any(v.tainted and v.checked for v in values)

    def build_inter_procedural_view(self) -> "InterProceduralView":
        """Build inter-procedural analysis view from function data and call graph.

        Analyzes each function for security-relevant properties and tracks
        taint flow across function boundaries via the call graph.
        """
        view = InterProceduralView()

        # Step 1: Build per-function summaries
        for method_id, func in self.functions.items():
            summary: Dict[str, bool] = {
                "has_sender_read": False,
                "has_send": False,
                "has_accept": False,
                "has_storage_read": False,
                "has_storage_write": False,
                "has_guard": False,
                "has_signature_verify": False,
                "has_random": False,
            }

            # Iterate through all blocks in the function
            for block in func.blocks.values():
                for inst in block.instructions:
                    labels = getattr(inst, 'semantic_labels', [])
                    if SemanticLabel.MESSAGE_SEND in labels:
                        summary["has_send"] = True
                    if SemanticLabel.GAS_ACCEPT in labels:
                        summary["has_accept"] = True
                    if SemanticLabel.STORAGE_READ in labels:
                        summary["has_storage_read"] = True
                    if SemanticLabel.STORAGE_WRITE in labels:
                        summary["has_storage_write"] = True
                    if SemanticLabel.GUARD in labels:
                        summary["has_guard"] = True
                    if SemanticLabel.SIGNATURE_VERIFY in labels:
                        summary["has_signature_verify"] = True
                    if SemanticLabel.RANDOM_GENERATE in labels:
                        summary["has_random"] = True
                    # Check for sender read via opcode
                    if inst.opcode in ("LDMSGADDR", "LDMSGADDRQ", "INMSG_SRC"):
                        summary["has_sender_read"] = True

            view.function_summaries[method_id] = summary

        # Step 2: Build cross-function edges from call graph
        for caller_id, callee_id in self.call_graph:
            caller_has_taint = view.function_summaries.get(caller_id, {}).get("has_sender_read", False)
            edge = InterProceduralEdge(
                caller_method_id=caller_id,
                callee_method_id=callee_id,
                caller_instruction_index=-1,  # Simplified
                taint_propagated=caller_has_taint,
            )
            view.edges.append(edge)

        # Build adjacency list from call graph
        adjacency: Dict[int, set] = {}
        for caller, callee in self.call_graph:
            adjacency.setdefault(caller, set()).add(callee)

        def reachable_from(start: int) -> set:
            """BFS to find all reachable method IDs from start."""
            visited: set = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            return visited

        # Step 3: Find cross-function taint paths with transitive closure
        for method_a, summary_a in view.function_summaries.items():
            if not summary_a.get("has_sender_read"):
                continue
            reachable = reachable_from(method_a)
            for method_b in reachable:
                if method_a == method_b:
                    continue
                summary_b = view.function_summaries.get(method_b, {})
                if summary_b.get("has_send") and not summary_b.get("has_guard"):
                    view.taint_paths.append((method_a, -1, method_b, -1))

        self.inter_procedural = view
        return view

    def __repr__(self) -> str:
        """String representation of the module."""
        return f"TVMModule({len(self.functions)} functions, {len(self.global_blocks)} global blocks)"
