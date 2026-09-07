"""
Continuation extraction and resolution for TVM program analysis.

This module handles:
- Extracting continuations from instruction trees
- Tracking PUSHCONT -> continuation relationships
- Simulating stack to resolve which continuations branches consume
"""
import logging
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

from ..config import MAX_PROGRAM_ANALYZE_DEPTH
from .constants import (
    BRANCH_CONT_ARITY,
    CALL_ARGS_IMM_OPCODES,
    CALL_ARGS_SUFFIX_PREFIXES,
    CALL_ARGS_VAR_OPCODES,
    CALL_CONT_OPCODES,
    CALL_CTRL_REG_INDEX,
    CALL_CTRL_REG_OPCODES,
    CALL_NO_STACK_OPCODES,
    CONDITIONAL_BRANCHES,
    DICT_DISPATCH_EXEC_OPCODES,
    DICT_DISPATCH_OPCODES,
    END_LOOP_OPCODES,
    LOOP_CALL_OPCODES,
    MAIN_CONTEXT,
    RETURNING_CONDITIONALS,
    TERMINATORS,
    UNCONDITIONAL_BRANCHES,
)
from .facts import Continuation, InstructionFact
from .utils import extract_int_arg
from ..ir.stack_effects import get_stack_effect

logger = logging.getLogger(__name__)


UNKNOWN_CONT_SENTINEL = "__unknown__"
UNKNOWN_CONT_PREFIX = "__unknown_from_"


class UncertainInfo(NamedTuple):
    """Describes why a branch continuation target is uncertain."""

    reason: str
    source_index: int
    source_opcode: str
    detail: str = ""


def make_unknown_cont(source_index: int, source_opcode: str) -> str:
    """Build an unknown-continuation marker with source provenance."""
    safe_opcode = (source_opcode or "unknown").strip().replace(" ", "_")
    return f"{UNKNOWN_CONT_PREFIX}{source_index}_{safe_opcode}__"


def parse_unknown_cont(cont_id: str) -> Optional[Tuple[int, str]]:
    """Parse unknown-cont marker into (source_index, source_opcode)."""
    if not cont_id or not cont_id.startswith(UNKNOWN_CONT_PREFIX) or not cont_id.endswith("__"):
        return None
    suffix = cont_id[len(UNKNOWN_CONT_PREFIX):-2]
    parts = suffix.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), parts[1]
    except ValueError:
        return None


def is_unknown_cont(cont_id: Optional[str]) -> bool:
    """Return True for legacy and sourced unknown-cont sentinels."""
    if cont_id is None:
        return False
    return cont_id == UNKNOWN_CONT_SENTINEL or cont_id.startswith(UNKNOWN_CONT_PREFIX)


class ContinuationResolver:
    """Extracts and resolves continuations from TVM bytecode."""

    def __init__(self):
        """Initialize the continuation resolver."""
        self._extract_int_arg = extract_int_arg

    def extract_continuations(
        self, instructions: List, max_depth: int = MAX_PROGRAM_ANALYZE_DEPTH
    ) -> Tuple[Dict[Tuple[str, int], List[str]], Dict[Tuple[str, int], List[str]], Dict[str, Continuation]]:
        """
        Extract continuations from instructions without flattening.

        Args:
            instructions: List of Instruction objects
            max_depth: Maximum recursion depth (default MAX_PROGRAM_ANALYZE_DEPTH, prevents DoS via malformed input)

        Returns:
            Tuple of:
            - push_map: Dict mapping (context_id, local_index) -> list[continuation_id] for PUSHCONT
            - inline_map: Dict mapping (context_id, local_index) -> list[continuation_id] for inline code args
            - continuations: Dict mapping continuation_id -> Continuation object

        Raises:
            RecursionError: If max_depth is exceeded
        """
        push_map: Dict[Tuple[str, int], List[str]] = {}
        inline_map: Dict[Tuple[str, int], List[str]] = {}
        continuations: Dict[str, Continuation] = {}
        cont_counter = 0

        def process_list(inst_list: List, context_id: Optional[str], depth: int = 0) -> None:
            nonlocal cont_counter
            if depth > max_depth: 
                raise RecursionError(f"Continuation extraction exceeded max depth {max_depth}")
            ctx = context_id or MAIN_CONTEXT # Use provided context_id or default to MAIN_CONTEXT for top-level
            for local_idx, inst in enumerate(inst_list):
                if hasattr(inst, 'args') and inst.args: # Only process instructions with arguments
                    for arg_idx, arg in enumerate(inst.args):
                        if hasattr(arg, 'value'):
                            val = arg.value
                            if hasattr(val, 'entries') and val.entries:
                                for entry in val.entries:
                                    code = getattr(entry, "code", None)
                                    code_instructions = getattr(code, "instructions", None) if code is not None else None
                                    if code is None or code_instructions is None:
                                        continue
                                    try:
                                        nested_instructions = list(code_instructions)
                                    except TypeError:
                                        continue
                                    cont_id = f"cont_{cont_counter}"
                                    cont_counter += 1
                                    continuation = Continuation(
                                        id=cont_id,
                                        instructions=nested_instructions,
                                        parent_context=ctx,
                                        parent_local_index=local_idx,
                                        parent_arg_index=arg_idx,
                                        entry_index=0,
                                        kind="method",
                                        method_id=getattr(entry, "key", None),
                                    )
                                    continuations[cont_id] = continuation
                                    process_list(nested_instructions, cont_id, depth + 1)
                            instructions_attr = getattr(val, "instructions", None)
                            if instructions_attr is not None:
                                try:
                                    nested_instructions = list(instructions_attr)
                                except TypeError:
                                    continue
                                cont_id = f"cont_{cont_counter}"
                                cont_counter += 1
                                continuation = Continuation(
                                    id=cont_id,
                                    instructions=nested_instructions,
                                    parent_context=ctx,
                                    parent_local_index=local_idx,
                                    parent_arg_index=arg_idx,
                                    entry_index=0,
                                )
                                inst_name = getattr(inst, "name", "").upper()
                                if (
                                    inst_name.startswith("PUSHCONT")
                                    or inst_name.startswith("PUSHREFCONT")
                                ):
                                    push_map.setdefault((ctx, local_idx), []).append(cont_id)
                                else:
                                    inline_map.setdefault((ctx, local_idx), []).append(cont_id)
                                continuations[cont_id] = continuation
                                process_list(nested_instructions, cont_id, depth + 1)

        process_list(instructions, None, 0)

        return push_map, inline_map, continuations

    def map_continuations(
        self,
        facts: Sequence[InstructionFact],
        pushcont_to_cont_ids: Dict[int, List[str]],
        inline_cont_map: Dict[int, List[str]],
        entry_shape: Optional[Tuple[List[Tuple[str, Optional[str]]], bool]] = None,
        cont_entry_shapes: Optional[
            Dict[str, List[Tuple[List[Tuple[str, Optional[str]]], bool]]]
        ] = None,
        cont_entry_ctrl_regs: Optional[Dict[str, List[Dict[int, Optional[str]]]]] = None,
        initial_ctrl_regs: Optional[Dict[int, Optional[str]]] = None,
        enable_known_prefix_fallback: bool = True,
        enable_varargs_depth_fix: bool = True,
    ) -> Tuple[
        Dict[int, List[str]],
        Set[int],
        Dict[str, List[int]],
        Dict[int, List[UncertainInfo]],
        Dict[int, Optional[str]],
    ]:
        """
        Simulate a lightweight stack to determine which continuation each branch consumes.
        This function is intentionally kept as a single unit because the stack
        simulation logic is tightly coupled and benefits from shared state.
        Sections:

        1. INITIALIZATION: Set up result containers and stack state
        2. HELPER FUNCTIONS: Stack manipulation utilities
           - push_value/push_cont/pop_value/pop_token: Basic stack operations
           - apply_stack_transform: Model common stack shuffles (DUP, SWAP, etc.)
           - is_unmodeled_shuffle: Detect complex shuffles that invalidate tracking
        3. MAIN LOOP: Process each instruction
           - Phase A: Handle PUSHCONT instructions
           - Phase B: Classify instruction type (conditional/unconditional branch)
           - Phase C: Resolve continuation consumption for branches
           - Phase D: Update return target mapping for returning branches
           - Phase E: Apply stack effects

        Returns:
            branch_cont_map: branch instruction index -> list[continuation id]
            returning_branches: conditional branches that expect to resume execution
            cont_return_targets: continuation id -> list of return instruction indices
            uncertain_branches: branches where continuation could not be resolved due to stack uncertainty
            final_ctrl_regs: final control register state after processing this context
        """
        # ========== SECTION 1: STACK INITIALIZATION ==========
        branch_cont_map: Dict[int, List[str]] = {}
        returning_branches: Set[int] = set()
        cont_return_targets: Dict[str, List[int]] = {}
        uncertain_branches: Dict[int, List[UncertainInfo]] = {}

        STACK_CONT = "cont"
        STACK_VALUE = "value"
        STACK_UNKNOWN = "unknown"

        # Initialize stack state from entry shape if provided, otherwise start with empty stack and unknown tail.
        if entry_shape is not None:
            stack = list(entry_shape[0])
            unknown_below = bool(entry_shape[1])
        else:
            stack = []
            unknown_below = bool(facts) and facts[0].continuation_id is not None
        ctrl_regs: Dict[int, Optional[str]] = dict(initial_ctrl_regs) if initial_ctrl_regs else {}

        # ========== SECTION 2: HELPER FUNCTIONS ==========
        def _suffix_arg_count(op: str) -> Optional[int]:
            """Parse argument count from opcode suffix (e.g., CALLXARGS_1)."""
            upper_op = op.upper()
            for prefix in CALL_ARGS_SUFFIX_PREFIXES:
                if upper_op.startswith(prefix):
                    suffix = upper_op[len(prefix):]
                    if suffix.isdigit():
                        try:
                            return int(suffix)
                        except ValueError:
                            return None
            return None

        def _int_arg(fact: Optional[InstructionFact], arg_idx: int = 0) -> Optional[int]:
            if fact is None:
                return None
            return self._extract_int_arg(fact.arguments, arg_idx)

        def _is_call_cont_opcode(op: str) -> bool:
            upper_op = op.upper()
            if upper_op in CALL_CONT_OPCODES:
                return True
            return _suffix_arg_count(upper_op) is not None

        ALT_TOP_CONT_CONDITIONALS = {
            "IF", "IFNOT",
            "IFJMP", "IFNOTJMP", "IFJMPREF", "IFNOTJMPREF",
            "IFBITJMP", "IFNBITJMP", "IFBITJMPREF", "IFNBITJMPREF",
            "IFELSE", "IFREFELSE", "IFELSEREF", "IFREFELSEREF",
        }
        COND_COMPATIBLE = "compatible"
        COND_INCOMPATIBLE = "incompatible"
        COND_UNKNOWN = "unknown"
        last_invalidate_reason: Optional[UncertainInfo] = None

        def add_uncertain_reason(branch_idx: int, info: UncertainInfo) -> None:
            bucket = uncertain_branches.setdefault(branch_idx, [])
            if info not in bucket:
                bucket.append(info)

        def source_reason_from_opcode(source_opcode: str) -> str:
            upper_source = (source_opcode or "").upper()
            if upper_source.startswith("BLESS"):
                return "bless"
            if (
                upper_source.startswith("POPCTR")
                or upper_source in {
                    "SETRETCTR",
                    "SETALTCTR",
                    "ATEXIT",
                    "ATEXITALT",
                    "SETEXITALT",
                    "SAMEALT",
                    "SAMEALTSAVE",
                }
                or upper_source.startswith("INVERT_C")
            ):
                return "ctrl_reg_unknown"
            if upper_source.startswith("SETCONT"):
                return "dynamic_effect"
            return "unresolved"

        def push_value(count: int = 1, value: Optional[int] = None) -> None:
            for i in range(count):
                stack.insert(0, (STACK_VALUE, value if i == 0 else None))

        def push_cont(cont_id: str) -> None:
            stack.insert(0, (STACK_CONT, cont_id))

        def pop_value(count: int = 1) -> None:
            nonlocal unknown_below
            for _ in range(count):
                if stack:
                    stack.pop(0)
                elif unknown_below:
                    # Pop from unknown tail; nothing to track explicitly.
                    continue
                else:
                    # Underflow/insufficient info - mark stack as unknown.
                    unknown_below = True

        def pop_token() -> Optional[Tuple[str, Optional[str]]]:
            if stack:
                return stack.pop(0)
            if unknown_below:
                return (STACK_UNKNOWN, None)
            return None

        def peek_token(depth: int) -> Optional[Tuple[str, Optional[str]]]:
            if depth < 0:
                return None
            if depth < len(stack):
                return stack[depth]
            if unknown_below:
                return (STACK_UNKNOWN, None)
            return None

        def invalidate_stack(
            reason: str = "stack_underflow",
            source_idx: int = -1,
            source_op: str = "",
            detail: str = "",
        ) -> None:
            nonlocal unknown_below, last_invalidate_reason
            stack.clear()
            unknown_below = True
            last_invalidate_reason = UncertainInfo(reason, source_idx, source_op, detail)

        def _token_int(token: Optional[Tuple[str, Optional[str]]], min_value: int = 0) -> Optional[int]:
            if token is None or token[0] != STACK_VALUE or not isinstance(token[1], int):
                return None
            if token[1] < min_value:
                return None
            return token[1]

        def _condition_token_compat(depth: int) -> str:
            """
            Classify IF*/IFELSE* condition-slot compatibility.

            VM requires an integer/boolean (`pop_bool`); a known continuation token
            at the condition depth is definitely ill-typed. Missing/unknown data is
            kept as COND_UNKNOWN to avoid false negatives in conservative CFG mode.
            """
            token = peek_token(depth)
            if token is None or token[0] == STACK_UNKNOWN:
                return COND_UNKNOWN
            if token[0] == STACK_CONT:
                return COND_INCOMPATIBLE
            if token[0] == STACK_VALUE:
                value = token[1]
                if isinstance(value, int):
                    return COND_COMPATIBLE
                return COND_UNKNOWN
            return COND_UNKNOWN

        def _is_valid_ctrl_reg_idx(idx_val: int) -> bool:
            # VM ControlRegs::valid_idx(): c0..c5 and c7 are valid; c6 is invalid.
            return idx_val in {0, 1, 2, 3, 4, 5, 7}

        def _resolve_call_varargs_p() -> Optional[int]:
            # CALL*VARARGS stack layout in VM:
            #   s0=r, s1=p, s2=cont, s3..=args
            # p/r both allow -1.
            r_val = _token_int(peek_token(0), -1)
            p_val = _token_int(peek_token(1), -1)
            cont_token = peek_token(2)
            if (
                r_val is None
                or p_val is None
                or cont_token is None
                or cont_token[0] != STACK_CONT
                or cont_token[1] is None
            ):
                return None
            return p_val

        def _resolve_call_varargs_cont_depth() -> Optional[int]:
            p_val = _resolve_call_varargs_p()
            if p_val is None:
                return None
            # With unknown tail, multiple known continuation tokens in prefix are ambiguous.
            # Resolve only when the known continuation candidate is unique at depth 2.
            if unknown_below:
                known_cont_depths = [
                    depth
                    for depth, token in enumerate(stack)
                    if token[0] == STACK_CONT and token[1] is not None
                ]
                if known_cont_depths != [2]:
                    return None
            return 2

        def _resolve_call_varargs_unknown_tail_cont_depth() -> Optional[int]:
            """Recover CALL*VARARGS continuation depth from known prefix over unknown tail."""
            if not unknown_below or len(stack) < 3:
                return None
            r_val = _token_int(peek_token(0), -1)
            p_val = _token_int(peek_token(1), -1)
            cont_token = peek_token(2)
            if (
                r_val is None
                or p_val is None
                or cont_token is None
                or cont_token[0] != STACK_CONT
                or cont_token[1] is None
            ):
                return None
            known_cont_depths = [
                depth
                for depth, token in enumerate(stack)
                if token[0] == STACK_CONT and token[1] is not None
            ]
            if known_cont_depths != [2]:
                return None
            return 2

        def _resolve_jmpx_varargs_p() -> Optional[int]:
            # JMPXVARARGS stack layout in VM:
            #   s0=p, s1=cont, s2..=args
            # p allows -1.
            p_val = _token_int(peek_token(0), -1)
            cont_token = peek_token(1)
            if (
                p_val is None
                or cont_token is None
                or cont_token[0] != STACK_CONT
                or cont_token[1] is None
            ):
                return None
            return p_val

        def _resolve_jmpx_varargs_cont_depth() -> Optional[int]:
            if _resolve_jmpx_varargs_p() is None:
                return None
            if unknown_below:
                known_cont_depths = [
                    depth
                    for depth, token in enumerate(stack)
                    if token[0] == STACK_CONT and token[1] is not None
                ]
                if known_cont_depths != [1]:
                    return None
            return 1

        def _resolve_jmpx_varargs_unknown_tail_cont_depth() -> Optional[int]:
            if not unknown_below or len(stack) < 2:
                return None
            p_val = _token_int(peek_token(0), -1)
            cont_token = peek_token(1)
            if (
                p_val is None
                or cont_token is None
                or cont_token[0] != STACK_CONT
                or cont_token[1] is None
            ):
                return None
            known_cont_depths = [
                depth
                for depth, token in enumerate(stack)
                if token[0] == STACK_CONT and token[1] is not None
            ]
            if known_cont_depths != [1]:
                return None
            return 1

        def _resolve_known_prefix_conts(stack_needed: int) -> Optional[List[str]]:
            """Recover continuation IDs from known stack prefix when depth is unknown."""
            if stack_needed <= 0:
                return []
            known_conts = [
                token[1]
                for token in stack
                if token[0] == STACK_CONT and token[1] is not None
            ]
            if len(known_conts) == stack_needed:
                return known_conts
            return None

        def _project_stack_after_pop(
            pop_count: int,
        ) -> Tuple[List[Tuple[str, Optional[str]]], bool]:
            """Project caller stack shape after consuming ``pop_count`` top items."""
            projected_stack = list(stack)
            projected_unknown = unknown_below
            for _ in range(max(pop_count, 0)):
                if projected_stack:
                    projected_stack.pop(0)
                elif projected_unknown:
                    continue
                else:
                    projected_unknown = True
            return projected_stack, projected_unknown

        def _project_tryargs_body_entry_shape(params: Optional[int]) -> Tuple[List[Tuple[str, Optional[str]]], bool]:
            """Project TRYARGS callee entry stack (top ``params`` args after popping handler/body)."""
            if params is None or params < 0:
                return [], True
            projected_stack, projected_unknown = _project_stack_after_pop(2)
            if params == 0:
                return [], False
            if len(projected_stack) >= params:
                # TRYARGS passes exactly ``params`` top values to callee and drops the rest.
                return list(projected_stack[:params]), False
            # Not enough known prefix to recover all passed args precisely.
            if projected_unknown:
                return list(projected_stack), True
            return list(projected_stack), True

        def _record_entry_shapes(
            cont_ids: Sequence[str],
            pop_count: Optional[int],
            shape_override: Optional[Tuple[List[Tuple[str, Optional[str]]], bool]] = None,
        ) -> None:
            """Record candidate entry stack shapes for callees consumed at this site."""
            if cont_entry_shapes is None or not cont_ids:
                return
            if shape_override is not None:
                shape_stack, shape_unknown = shape_override
            elif pop_count is None:
                shape_stack: List[Tuple[str, Optional[str]]] = []
                shape_unknown = True
            else:
                shape_stack, shape_unknown = _project_stack_after_pop(pop_count)
            for cont_id in cont_ids:
                cont_entry_shapes.setdefault(cont_id, []).append(
                    (list(shape_stack), shape_unknown)
                )

        def _prepend_entry_shape_token(cont_ids: Sequence[str], token: Tuple[str, Optional[str]]) -> None:
            if cont_entry_shapes is None:
                return
            for cont_id in cont_ids:
                shape_list = cont_entry_shapes.get(cont_id)
                if not shape_list:
                    continue
                shape_stack, shape_unknown = shape_list[-1]
                new_stack = [token, *shape_stack]
                shape_list[-1] = (new_stack, shape_unknown)

        def _record_entry_ctrl_regs(
            cont_ids: Sequence[str],
            snapshot_override: Optional[Dict[int, Optional[str]]] = None,
        ) -> None:
            """Record candidate caller control-register snapshots for callees."""
            if cont_entry_ctrl_regs is None or not cont_ids:
                return
            snapshot = dict(snapshot_override) if snapshot_override is not None else dict(ctrl_regs)
            for cont_id in cont_ids:
                cont_entry_ctrl_regs.setdefault(cont_id, []).append(dict(snapshot))

        def apply_stack_transform(opcode: str, fact: Optional[InstructionFact] = None) -> bool:
            """
            Apply modeled stack transformations.

            Extended support for: Extended to handle more common stack operations:
            - DUP2, OVER2 for double duplication
            - DROP, DROP2 for stack consumption
            - -ROT for reverse rotation

            Added support for: Added support for:
            - POP (equivalent to DROP)
            - 2DUP (equivalent to DUP2)
            - PUSH s(i) with parameter extraction
            - XCHG_0I, XCHG_1I, XCHG_IJ parametric exchanges
            - REVERSE (simple 2-element reversal)
            """
            nonlocal unknown_below
            upper = opcode.upper()

            # Helper to extract integer argument from instruction
            def get_stack_index(arg_idx: int = 0) -> Optional[int]:
                if fact is None:
                    return None
                return self._extract_int_arg(fact.arguments, arg_idx)

            def pop_stack_index(min_value: int = 0) -> Optional[int]:
                token = pop_token()
                if (
                    token
                    and token[0] == STACK_VALUE
                    and isinstance(token[1], int)
                    and token[1] >= min_value
                ):
                    return token[1]
                return None

            if upper == "DUP":
                if len(stack) >= 1:
                    stack.insert(0, stack[0])
                else:
                    invalidate_stack()
                return True
            if upper in {"DUP2", "2DUP"}:
                # Duplicate top two elements: a b -> a b a b
                if len(stack) >= 2:
                    first = stack[0]
                    second = stack[1]
                    stack.insert(0, second)
                    stack.insert(0, first)
                else:
                    invalidate_stack()
                return True
            if upper == "OVER":
                if len(stack) >= 2:
                    stack.insert(0, stack[1])
                else:
                    invalidate_stack()
                return True
            if upper == "OVER2":
                # Copy second pair: a b c d -> a b c d a b
                if len(stack) >= 4:
                    first = stack[2]
                    second = stack[3]
                    stack.insert(0, second)
                    stack.insert(0, first)
                else:
                    invalidate_stack()
                return True
            if upper in {"SWAP", "XCHG"}:
                if len(stack) >= 2:
                    stack[0], stack[1] = stack[1], stack[0]
                else:
                    invalidate_stack()
                return True

            if upper in {"XCHG_0I", "XCHG_0I_LONG"}:
                # Exchange s0 with s(i)
                i = get_stack_index(0)
                if i is not None and 0 <= i < len(stack):
                    stack[0], stack[i] = stack[i], stack[0]
                    return True
                invalidate_stack()
                return True
            if upper == "XCHG_1I":
                # Exchange s1 with s(i)
                i = get_stack_index(0)
                if i is not None and 1 <= i < len(stack) and len(stack) >= 2:
                    stack[1], stack[i] = stack[i], stack[1]
                    return True
                invalidate_stack()
                return True
            if upper == "XCHG_IJ":
                # Exchange s(i) with s(j)
                i = get_stack_index(0)
                j = get_stack_index(1)
                if i is not None and j is not None:
                    if 0 <= i < len(stack) and 0 <= j < len(stack):
                        stack[i], stack[j] = stack[j], stack[i]
                        return True
                invalidate_stack()
                return True

            if upper == "XCHG2":
                # XCHG2 i,j: first s1 <-> s(i), then s0 <-> s(j)
                i = get_stack_index(0)
                j = get_stack_index(1)
                if i is not None and j is not None:
                    if len(stack) >= 2 and 1 <= i < len(stack) and 0 <= j < len(stack):
                        stack[1], stack[i] = stack[i], stack[1]
                        stack[0], stack[j] = stack[j], stack[0]
                        return True
                invalidate_stack()
                return True
            if upper in {"XCHG3", "XCHG3_ALT"}:
                # XCHG3 i,j,k: s2 <-> s(i), then s1 <-> s(j), then s0 <-> s(k)
                i = get_stack_index(0)
                j = get_stack_index(1)
                k = get_stack_index(2) if fact and len(fact.arguments) > 2 else None
                if i is not None and j is not None and k is not None:
                    if len(stack) >= 3 and 2 <= i < len(stack) and 1 <= j < len(stack) and 0 <= k < len(stack):
                        stack[2], stack[i] = stack[i], stack[2]
                        stack[1], stack[j] = stack[j], stack[1]
                        stack[0], stack[k] = stack[k], stack[0]
                        return True
                invalidate_stack()
                return True

            if upper == "ROT":
                # Rotate left: a b c -> b c a
                if len(stack) >= 3:
                    stack[0], stack[1], stack[2] = stack[1], stack[2], stack[0]
                else:
                    invalidate_stack()
                return True
            if upper in {"-ROT", "ROTREV"}:
                # Reverse rotation: a b c -> c a b
                if len(stack) >= 3:
                    stack[0], stack[1], stack[2] = stack[2], stack[0], stack[1]
                else:
                    invalidate_stack()
                return True
            if upper == "NIP":
                if len(stack) >= 2:
                    stack.pop(1)
                else:
                    invalidate_stack()
                return True
            if upper == "TUCK":
                if len(stack) >= 2:
                    stack.insert(2, stack[0])
                else:
                    invalidate_stack()
                return True
            if upper in {"SWAP2", "2SWAP"}:
                # SWAP2/2SWAP: a b c d -> c d a b
                if len(stack) >= 4:
                    stack[0], stack[2] = stack[2], stack[0]
                    stack[1], stack[3] = stack[3], stack[1]
                else:
                    invalidate_stack()
                return True
            if upper == "PICK":
                # PICK: pop dynamic index i, then push s(i) from the remaining stack.
                i_val = pop_stack_index(0)
                if i_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_pick_index")
                    return True
                if 0 <= i_val < len(stack):
                    stack.insert(0, stack[i_val])
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "pick_depth_out_of_range")
                return True
            if upper == "ONLYTOPX":
                # ONLYTOPX: pop x, keep only top x stack items.
                x_val = pop_stack_index(0)
                if x_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_onlytopx_count")
                    return True
                if x_val <= len(stack):
                    if x_val == 0:
                        stack.clear()
                    else:
                        del stack[x_val:]
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "onlytopx_depth_out_of_range")
                return True
            if upper == "ONLYX":
                # ONLYX: pop x, keep only bottom x stack items.
                x_val = pop_stack_index(0)
                if x_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_onlyx_count")
                    return True
                if x_val <= len(stack):
                    if x_val == 0:
                        stack.clear()
                    else:
                        stack[:] = stack[-x_val:]
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "onlyx_depth_out_of_range")
                return True
            if upper in {"DROP", "POP"}:
                if stack:
                    stack.pop(0)
                else:
                    if unknown_below:
                        # Drop from unknown tail.
                        return True
                    invalidate_stack()
                return True
            if upper == "DROP2":
                # Drop top two elements
                if len(stack) >= 2:
                    stack.pop(0)
                    stack.pop(0)
                else:
                    invalidate_stack()
                return True
            if upper == "DROPX":
                # DROPX: pop dynamic count n, then drop n values.
                n = pop_stack_index(0)
                if n is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_drop_count")
                    return True
                pop_value(n)
                return True

            if upper == "PUSH":
                i = get_stack_index(0)
                if i is not None and 0 <= i < len(stack):
                    stack.insert(0, stack[i])
                    return True
                # If no valid index, treat as pushing unknown value
                push_value(1)
                return True

            if upper == "PUSH3":
                # PUSH3 i,j,k: equivalent to s[i] PUSH; s[j+1] s[k+1] PUSH2.
                i_val = get_stack_index(0)
                j_val = get_stack_index(1)
                k_val = get_stack_index(2)

                if i_val is not None and 0 <= i_val < len(stack):
                    stack.insert(0, stack[i_val])
                else:
                    push_value(1)

                if j_val is not None:
                    j_adj = j_val + 1
                    if 0 <= j_adj < len(stack):
                        stack.insert(0, stack[j_adj])
                    else:
                        push_value(1)
                else:
                    push_value(1)

                if k_val is not None:
                    k_adj = k_val + 2
                    if 0 <= k_adj < len(stack):
                        stack.insert(0, stack[k_adj])
                    else:
                        push_value(1)
                else:
                    push_value(1)
                return True

            # REVERSE n, i: reverse n elements starting at position i
            if upper == "REVERSE":
                n = get_stack_index(0)
                i = get_stack_index(1)
                if n is None:
                    n = 2  # Default to reversing top 2
                if i is None:
                    i = 0  # Default to starting from TOS
                if i + n <= len(stack):
                    stack[i:i+n] = stack[i:i+n][::-1]
                elif i < len(stack):
                    # Partial reverse up to what we know, then invalidate
                    invalidate_stack()
                else:
                    invalidate_stack()
                return True

            # Model additional stack operations used by the CFG.

            # BLKSWAP i,j: swap top i elements with next j elements
            if upper == "BLKSWAP":
                i_val = get_stack_index(0)
                j_val = get_stack_index(1)
                if i_val is not None and j_val is not None and i_val >= 0 and j_val >= 0:
                    total = i_val + j_val
                    if total <= len(stack):
                        top_i = stack[:i_val]
                        mid_j = stack[i_val:total]
                        rest = stack[total:]
                        stack[:] = mid_j + top_i + rest
                    else:
                        invalidate_stack()
                else:
                    invalidate_stack()
                return True

            # BLKDROP n: drop top n elements
            if upper == "BLKDROP":
                n = get_stack_index(0)
                if n is not None and n >= 0:
                    if n <= len(stack):
                        del stack[:n]
                    else:
                        invalidate_stack()
                else:
                    invalidate_stack()
                return True

            # BLKDROP2 n,j: keep top j elements, drop next n elements
            if upper == "BLKDROP2":
                n = get_stack_index(0)
                j = get_stack_index(1)
                if n is not None and j is not None and n >= 0 and j >= 0:
                    if j + n <= len(stack):
                        del stack[j:j + n]
                    else:
                        invalidate_stack()
                else:
                    invalidate_stack()
                return True

            # BLKPUSH n,i: push n copies of s(i)
            if upper == "BLKPUSH":
                n = get_stack_index(0)
                i_val = get_stack_index(1)
                if n is not None and i_val is not None and n >= 0 and 0 <= i_val < len(stack):
                    elem = stack[i_val]
                    for _ in range(n):
                        stack.insert(0, elem)
                else:
                    invalidate_stack()
                return True

            # TUPLE n: pop n values, push 1 tuple
            if upper == "TUPLE":
                n = get_stack_index(0)
                if n is not None and n >= 0:
                    if n <= len(stack):
                        del stack[:n]
                    else:
                        invalidate_stack()
                    push_value(1)
                else:
                    invalidate_stack()
                return True

            # UNTUPLE n: pop 1 tuple, push n values
            if upper == "UNTUPLE":
                n = get_stack_index(0)
                if n is not None and n >= 0:
                    pop_value(1)
                    push_value(n)
                else:
                    invalidate_stack()
                return True

            # NULLSWAPIFNOT / NULLSWAPIFNOT2: conditionally insert null(s)
            # Conservative: always model as inserting (over-approximate)
            if upper == "NULLSWAPIFNOT":
                # If TOS is false/null: pushes null below → net +1
                # If TOS is true: no change → net 0
                # Conservative: assume +1 (may over-count stack depth, but preserves tracking)
                push_value(1)
                return True
            if upper == "NULLSWAPIFNOT2":
                # Pushes up to 2 nulls conditionally
                push_value(2)
                return True

            # PUXC s(i),s(j-1): push s(i), then xchg s0,s(j)
            if upper == "PUXC":
                i_val = get_stack_index(0)
                j_val = get_stack_index(1)
                if i_val is not None and 0 <= i_val < len(stack):
                    stack.insert(0, stack[i_val])
                    # Now xchg s0 with s(j)
                    if j_val is not None:
                        j_adj = j_val + 1  # adjust for the push
                        if 0 <= j_adj < len(stack):
                            stack[0], stack[j_adj] = stack[j_adj], stack[0]
                        else:
                            invalidate_stack()
                    else:
                        invalidate_stack()
                else:
                    invalidate_stack()
                return True

            # PUSH2 s(i),s(j): push s(i) then push s(j+1)
            if upper == "PUSH2":
                i_val = get_stack_index(0)
                j_val = get_stack_index(1)
                if i_val is not None and 0 <= i_val < len(stack):
                    stack.insert(0, stack[i_val])
                    if j_val is not None:
                        j_adj = j_val + 1
                        if 0 <= j_adj < len(stack):
                            stack.insert(0, stack[j_adj])
                        else:
                            push_value(1)
                    else:
                        push_value(1)
                else:
                    push_value(2)
                return True

            # PUSH_LONG s(i): same as PUSH but with longer encoding
            if upper == "PUSH_LONG":
                i_val = get_stack_index(0)
                if i_val is not None and 0 <= i_val < len(stack):
                    stack.insert(0, stack[i_val])
                    return True
                push_value(1)
                return True

            if upper in {
                "PUSHNAN",
                "PUSHNULL",
                "PUSHPOW2",
                "PUSHPOW2DEC",
                "PUSHNEGPOW2",
                "PUSHREF",
                "PUSHREFSLICE",
                "PUSHSLICE",
                "PUSHSLICE_LONG",
                "PUSHSLICE_REFS",
                "PREVBLOCKSINFOTUPLE",
                "UNPACKEDCONFIGTUPLE",
            }:
                push_value(1)
                return True

            if upper == "ISTUPLE":
                # ISTUPLE keeps stack depth and normalizes TOS to boolean value.
                if stack:
                    stack[0] = (STACK_VALUE, None)
                elif unknown_below:
                    stack.insert(0, (STACK_VALUE, None))
                else:
                    invalidate_stack()
                return True

            if upper == "POP_LONG":
                # POP_LONG i: pop old s0 and store into old s[i], reducing depth by 1.
                i_val = get_stack_index(0)
                if i_val is None or i_val < 0:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "invalid_pop_long_index")
                    return True
                token = pop_token()
                if token is None or token[0] == STACK_UNKNOWN:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "pop_long_unknown_top")
                    return True
                if i_val == 0:
                    return True
                replace_idx = i_val - 1
                if 0 <= replace_idx < len(stack):
                    stack[replace_idx] = token
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "pop_long_depth_out_of_range")
                return True

            if upper == "XCHGX":
                # XCHGX: pop dynamic index i, then swap s0 and s[i].
                i_val = pop_stack_index(0)
                if i_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_xchgx_index")
                    return True
                if 0 <= i_val < len(stack):
                    if i_val > 0:
                        stack[0], stack[i_val] = stack[i_val], stack[0]
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "xchgx_depth_out_of_range")
                return True

            if upper == "ROLL":
                # ROLL: pop i, then rotate top i+1 items left by one.
                i_val = pop_stack_index(0)
                if i_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_roll_index")
                    return True
                width = i_val + 1
                if width <= 1:
                    return True
                if width <= len(stack):
                    segment = stack[:width]
                    stack[:width] = segment[1:] + segment[:1]
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "roll_width_out_of_range")
                return True

            if upper == "ROLLREV":
                # ROLLREV: pop i, then rotate top i+1 items right by one.
                i_val = pop_stack_index(0)
                if i_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_rollrev_index")
                    return True
                width = i_val + 1
                if width <= 1:
                    return True
                if width <= len(stack):
                    segment = stack[:width]
                    stack[:width] = segment[-1:] + segment[:-1]
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "rollrev_width_out_of_range")
                return True

            if upper == "BLKSWX":
                # BLKSWX: pop dynamic i,j then perform BLKSWAP i,j.
                i_val = pop_stack_index(0)
                j_val = pop_stack_index(0)
                if i_val is None or j_val is None:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "dynamic_blkswx_indices")
                    return True
                total = i_val + j_val
                if total <= len(stack):
                    top_i = stack[:i_val]
                    mid_j = stack[i_val:total]
                    rest = stack[total:]
                    stack[:] = mid_j + top_i + rest
                else:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "blkswx_width_out_of_range")
                return True

            if upper == "TUPLEVAR":
                # TUPLEVAR: x_1..x_n n -> tuple
                n_val = pop_stack_index(0)
                if n_val is None:
                    invalidate_stack("dynamic_effect", fact.index if fact else -1, upper, "unknown_tuplevar_arity")
                    push_value(1)
                    return True
                pop_value(n_val)
                push_value(1)
                return True

            if upper == "UNTUPLEVAR":
                # UNTUPLEVAR: tuple n -> x_1..x_n
                n_val = pop_stack_index(0)
                tuple_token = pop_token()
                if n_val is None or tuple_token is None:
                    invalidate_stack("dynamic_effect", fact.index if fact else -1, upper, "unknown_untuplevar_arity")
                    return True
                if tuple_token[0] == STACK_UNKNOWN:
                    unknown_below = True
                push_value(n_val)
                return True

            if upper == "PUXC2":
                # PUXC2 i,j,k: equivalent to s[i] PUSH; s2 XCHG0; s[j] s[k] XCHG2.
                i_val = get_stack_index(0)
                j_val = get_stack_index(1)
                k_val = get_stack_index(2)
                if (
                    i_val is None
                    or j_val is None
                    or k_val is None
                    or i_val < 0
                    or i_val >= len(stack)
                ):
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "invalid_puxc2_indices")
                    return True
                stack.insert(0, stack[i_val])
                if len(stack) < 3:
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "puxc2_insufficient_stack")
                    return True
                stack[0], stack[2] = stack[2], stack[0]
                j_adj = j_val + 1
                k_adj = k_val + 1
                if not (len(stack) >= 2 and 1 <= j_adj < len(stack) and 0 <= k_adj < len(stack)):
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "invalid_puxc2_depth")
                    return True
                stack[1], stack[j_adj] = stack[j_adj], stack[1]
                stack[0], stack[k_adj] = stack[k_adj], stack[0]
                return True

            if upper == "PUXCPU":
                # PUXCPU i,j,k: equivalent to s[i] s[j-1] PUXC; s[k] PUSH.
                i_val = get_stack_index(0)
                j_val = get_stack_index(1)
                k_val = get_stack_index(2)
                if (
                    i_val is None
                    or j_val is None
                    or k_val is None
                    or i_val < 0
                    or i_val >= len(stack)
                ):
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "invalid_puxcpu_indices")
                    return True
                stack.insert(0, stack[i_val])
                j_adj = j_val + 1
                if not (0 <= j_adj < len(stack)):
                    invalidate_stack("stack_invalidated", fact.index if fact else -1, upper, "invalid_puxcpu_depth")
                    return True
                stack[0], stack[j_adj] = stack[j_adj], stack[0]
                k_adj = k_val + 1
                if 0 <= k_adj < len(stack):
                    stack.insert(0, stack[k_adj])
                else:
                    push_value(1)
                return True

            return False

        def is_unmodeled_shuffle(opcode: str) -> bool:
            """
            Check if opcode is an unmodeled stack shuffle operation.

            Certain complex stack reordering operations (ROLL variants)
            have dynamic effects that depend on arguments and cannot be precisely modeled.
            When encountered, the stack state should be marked as uncertain.

            Extended support for: Extended modeled operations to include:
            - POP, 2DUP (aliases)
            - PUSH, PUSH_LONG, PUSH2 (parametric copy from depth)
            - XCHG_0I, XCHG_0I_LONG, XCHG_1I, XCHG_IJ (parametric exchanges)
            - REVERSE (element reversal)

            Additional stack-modeling cases: Now also models:
            - BLKSWAP, BLKDROP, BLKDROP2, BLKPUSH (block operations with args)
            - TUPLE, UNTUPLE (aggregate operations)
            - NULLSWAPIFNOT, NULLSWAPIFNOT2 (conditional null insertion)
            - PUXC, PUSH2 (compound push/exchange)

            Note: ROT and ROTREV (-ROT) are modeled. The ROTR prefix in shuffle_prefixes
            catches ROLL/ROTR* variants that are unmodeled dynamic rotations.

            Args:
                opcode: The instruction opcode to check

            Returns:
                True if this is an unmodeled shuffle that should clear stack state
            """
            upper = opcode.upper()
            modeled_ops = {
                "DUP", "DUP2", "2DUP", "OVER", "OVER2",
                "SWAP", "XCHG", "XCHG_0I", "XCHG_0I_LONG", "XCHG_1I", "XCHG_IJ",
                "XCHG2", "XCHG3", "XCHG3_ALT",
                "ROT", "-ROT", "ROTREV", "NIP", "TUCK",
                "SWAP2", "2SWAP",
                "DROP", "DROP2", "POP",
                "DROPX",
                "PUSH", "PUSH_LONG", "PUSH2", "PUSH3", "REVERSE",
                "PICK", "ONLYTOPX", "ONLYX",
                "BLKSWAP", "BLKDROP", "BLKDROP2", "BLKPUSH", "BLKSWX",
                "TUPLE", "UNTUPLE", "TUPLEVAR", "UNTUPLEVAR", "ISTUPLE",
                "NULLSWAPIFNOT", "NULLSWAPIFNOT2",
                "PUXC", "PUXC2", "PUXCPU",
                "POP_LONG",
                "XCHGX", "ROLL", "ROLLREV",
                "PUSHNAN", "PUSHNULL", "PUSHPOW2", "PUSHPOW2DEC", "PUSHNEGPOW2",
                "PUSHREF", "PUSHREFSLICE", "PUSHSLICE", "PUSHSLICE_LONG", "PUSHSLICE_REFS",
                "PREVBLOCKSINFOTUPLE", "UNPACKEDCONFIGTUPLE",
            }
            if upper in modeled_ops:
                return False
            # These prefixes indicate unmodeled shuffles that invalidate stack tracking
            # ROLL: dynamic rotation requiring runtime index
            # ROTR: right rotation (different from ROT)
            shuffle_prefixes = ("ROLL", "ROTR")
            return upper.startswith(shuffle_prefixes)

        # ========== SECTION 3: MAIN LOOP ==========
        for i, fact in enumerate(facts):
            idx = fact.index
            opcode = fact.opcode.upper()

            # --- Phase A: Handle PUSHCONT instructions ---
            # When we see a PUSHCONT, push the continuation ID onto our simulated stack
            cont_ids = pushcont_to_cont_ids.get(idx)
            if cont_ids:
                for cont_id in cont_ids:
                    push_cont(cont_id)
                continue

            # Push known integer constants (used for VARARGS resolution)
            if opcode.startswith("PUSHINT"):
                const_val = _int_arg(fact, 0)
                push_value(1, const_val)
                continue

            if opcode == "PREPAREDICT":
                # PREPAREDICT n is equivalent to pushing immediate n and c3.
                # Stack top becomes continuation from c3.
                push_value(1, _int_arg(fact, 0))
                reg_val = ctrl_regs.get(3)
                if isinstance(reg_val, str) and not is_unknown_cont(reg_val):
                    push_cont(reg_val)
                else:
                    push_value(1)
                continue

            # Continuation constructors that return a continuation object on stack.
            if opcode in {"BLESS", "BLESSARGS", "BLESSVARARGS"}:
                effect = get_stack_effect(opcode)
                min_inputs = effect.min_inputs if effect and effect.min_inputs is not None else 1
                pop_value(max(min_inputs, 1))
                if effect and (
                    effect.is_dynamic
                    or effect.max_inputs != effect.min_inputs
                    or effect.max_outputs != effect.min_outputs
                ):
                    unknown_below = True
                # The created continuation target is not statically known here.
                push_cont(make_unknown_cont(idx, opcode))
                continue

            # Phase A.2: Handle PSEUDO_PUSHREF — push inline continuation and
            # propagate current stack as its entry_shape for worklist resolution.
            # PSEUDO_PUSHREF is a disassembler pseudo-op that pushes a cell-ref
            # continuation onto the stack.  The continuation is extracted into its
            # own analysis context but never consumed by a branch in the parent,
            # so the worklist has no entry_shape link.  Recording the parent's
            # current stack gives the inlined context the caller-visible state it
            # would see when eventually executed.
            if opcode == "PSEUDO_PUSHREF":
                inline_conts = inline_cont_map.get(idx)
                if inline_conts:
                    # Record current stack as entry_shape for the inlined
                    # continuation.  This is the stack the continuation will
                    # observe when it is eventually invoked (approximated by
                    # the parent's state at push time).
                    if cont_entry_shapes is not None:
                        for cont_id in inline_conts:
                            cont_entry_shapes.setdefault(cont_id, []).append(
                                (list(stack), unknown_below)
                            )
                    # Push the continuation onto the parent's simulated stack
                    # so callers of the parent can track it.
                    for cont_id in inline_conts:
                        push_cont(cont_id)
                else:
                    # No inline continuation found — treat as generic push.
                    push_value(1)
                continue

            # Control register loads/stores
            if opcode == "PUSHCTR":
                reg_idx = _int_arg(fact, 0)
                if reg_idx is None or not _is_valid_ctrl_reg_idx(reg_idx):
                    push_value(1)
                else:
                    reg_val = ctrl_regs.get(reg_idx)
                    if isinstance(reg_val, str) and not is_unknown_cont(reg_val):
                        push_cont(reg_val)
                    else:
                        push_value(1)
                continue

            if opcode == "POPCTR":
                reg_idx = _int_arg(fact, 0)
                token = pop_token()
                if reg_idx is None or not _is_valid_ctrl_reg_idx(reg_idx):
                    unknown_reg = make_unknown_cont(idx, opcode)
                    ctrl_regs.update({0: unknown_reg, 1: unknown_reg, 2: unknown_reg, 3: unknown_reg})
                else:
                    if token and token[0] == STACK_CONT and token[1] is not None:
                        ctrl_regs[reg_idx] = token[1]
                    else:
                        ctrl_regs[reg_idx] = make_unknown_cont(idx, f"POPCTR_c{reg_idx}")
                continue

            if opcode == "PUSHCTRX":
                # Dynamic register index - resolve precise index when statically known.
                idx_token = pop_token()
                reg_idx = None
                if (
                    idx_token
                    and idx_token[0] == STACK_VALUE
                    and isinstance(idx_token[1], int)
                    and _is_valid_ctrl_reg_idx(idx_token[1])
                ):
                    reg_idx = idx_token[1]
                if reg_idx is None:
                    push_value(1)
                else:
                    reg_val = ctrl_regs.get(reg_idx)
                    if isinstance(reg_val, str) and not is_unknown_cont(reg_val):
                        push_cont(reg_val)
                    else:
                        push_value(1)
                continue

            if opcode == "POPCTRX":
                # Stack layout: x i - ; top is i.
                idx_token = pop_token()
                value_token = pop_token()
                reg_idx = None
                if (
                    idx_token
                    and idx_token[0] == STACK_VALUE
                    and isinstance(idx_token[1], int)
                    and _is_valid_ctrl_reg_idx(idx_token[1])
                ):
                    reg_idx = idx_token[1]
                if reg_idx is None:
                    unknown_reg = make_unknown_cont(idx, opcode)
                    ctrl_regs.update({0: unknown_reg, 1: unknown_reg, 2: unknown_reg, 3: unknown_reg})
                else:
                    if value_token and value_token[0] == STACK_CONT and value_token[1] is not None:
                        ctrl_regs[reg_idx] = value_token[1]
                    else:
                        ctrl_regs[reg_idx] = make_unknown_cont(idx, f"POPCTRX_c{reg_idx}")
                continue

            if opcode == "POPSAVE":
                # POPSAVE writes the popped value to c[i] (like POPCTR) and also mutates c0 savelist.
                reg_idx = _int_arg(fact, 0)
                token = pop_token()
                if reg_idx is None or not _is_valid_ctrl_reg_idx(reg_idx):
                    unknown_reg = make_unknown_cont(idx, opcode)
                    ctrl_regs.update({0: unknown_reg, 1: unknown_reg, 2: unknown_reg, 3: unknown_reg})
                else:
                    if token and token[0] == STACK_CONT and token[1] is not None:
                        ctrl_regs[reg_idx] = token[1]
                    else:
                        ctrl_regs[reg_idx] = make_unknown_cont(idx, f"POPSAVE_c{reg_idx}")
                    # POPSAVE c[i] updates c0's savelist for i != 0; resulting c0 identity is unknown.
                    if reg_idx != 0:
                        ctrl_regs[0] = make_unknown_cont(idx, "POPSAVE_c0")
                continue

            if opcode == "SETRETCTR":
                reg_idx = _int_arg(fact, 0)
                if reg_idx is None:
                    # Backward-compatible fallback for synthetic tests that omit c[i].
                    token = pop_token()
                    if token and token[0] == STACK_CONT and token[1] is not None:
                        ctrl_regs[0] = token[1]
                    else:
                        ctrl_regs[0] = make_unknown_cont(idx, opcode)
                else:
                    # Real SETRETCTR c[i] mutates c0 internals via continuation composition.
                    # Resulting c0 cannot be mapped to a known extracted continuation ID.
                    pop_value(1)
                    ctrl_regs[0] = make_unknown_cont(idx, opcode)
                continue

            if opcode == "SETALTCTR":
                reg_idx = _int_arg(fact, 0)
                if reg_idx is None:
                    # Backward-compatible fallback for synthetic tests that omit c[i].
                    token = pop_token()
                    if token and token[0] == STACK_CONT and token[1] is not None:
                        ctrl_regs[1] = token[1]
                    else:
                        ctrl_regs[1] = make_unknown_cont(idx, opcode)
                else:
                    # Real SETALTCTR c[i] mutates c1 internals via continuation composition.
                    pop_value(1)
                    ctrl_regs[1] = make_unknown_cont(idx, opcode)
                continue

            if opcode == "SETCONTCTR":
                # x c -> c' : updates continuation internals, returns new continuation.
                pop_value(2)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode == "SETCONTCTRX":
                # Dynamic register index variant; continuation internals become unknown.
                pop_value(3)
                unknown_below = True
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode == "SETCONTCTRMANY":
                pop_value(1)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode == "SETCONTCTRMANYX":
                pop_value(2)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode in {"SETCONTARGS", "SETCONTARGS_N"}:
                copy_count = _int_arg(fact, 0)
                if copy_count is not None and copy_count >= 0:
                    pop_value(copy_count + 1)
                else:
                    pop_value(1)
                    unknown_below = True
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode == "SETCONTVARARGS":
                # Stack: x_1..x_r c r n -> c'
                r_val: Optional[int] = None
                r_token = peek_token(1)
                if (
                    r_token
                    and r_token[0] == STACK_VALUE
                    and isinstance(r_token[1], int)
                    and r_token[1] >= 0
                ):
                    r_val = r_token[1]
                if r_val is None:
                    pop_value(3)  # minimum: c, r, n
                    unknown_below = True
                else:
                    pop_value(r_val + 3)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode == "SETNUMVARARGS":
                pop_value(2)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode in {"COMPOS", "COMPOSALT", "COMPOSBOTH", "BOOLAND", "BOOLOR"}:
                pop_value(2)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode in {"THENRET", "THENRETALT"}:
                pop_value(1)
                push_cont(make_unknown_cont(idx, opcode))
                continue

            if opcode == "ATEXIT":
                pop_value(1)
                # c0 is rewritten through continuation composition.
                ctrl_regs[0] = make_unknown_cont(idx, opcode)
                continue

            if opcode == "ATEXITALT":
                pop_value(1)
                # c1 is rewritten through continuation composition.
                ctrl_regs[1] = make_unknown_cont(idx, opcode)
                continue

            if opcode == "SAMEALT":
                # SAMEALT copies c0 into c1.
                ctrl_regs[1] = ctrl_regs.get(0, make_unknown_cont(idx, opcode))
                continue

            if opcode == "SAMEALTSAVE":
                # SAMEALTSAVE stores old c1 into c0's savelist, then copies c0 into c1.
                # c0 identity is no longer a plain extracted continuation ID.
                ctrl_regs[1] = ctrl_regs.get(0, make_unknown_cont(idx, opcode))
                ctrl_regs[0] = make_unknown_cont(idx, opcode)
                continue

            if opcode == "SETEXITALT":
                # SETEXITALT pops a continuation and composes a new c1 from
                # (popped continuation, c0, previous c1). The resulting target
                # cannot be mapped to an extracted continuation ID.
                pop_value(1)
                ctrl_regs[1] = make_unknown_cont(idx, opcode)
                continue

            if opcode == "INVERT":
                # Swap c0 and c1; missing register values are treated as unknown.
                c0_val = ctrl_regs.get(0, make_unknown_cont(idx, "INVERT_c0"))
                c1_val = ctrl_regs.get(1, make_unknown_cont(idx, "INVERT_c1"))
                ctrl_regs[0] = c1_val
                ctrl_regs[1] = c0_val
                continue

            # --- Phase B: Classify instruction type ---
            is_cond = opcode in CONDITIONAL_BRANCHES
            is_tail = opcode in UNCONDITIONAL_BRANCHES
            is_call = _is_call_cont_opcode(opcode)
            returns_to_caller = (
                opcode in RETURNING_CONDITIONALS
                or (is_call and opcode not in LOOP_CALL_OPCODES)
            )
            if opcode in DICT_DISPATCH_OPCODES:
                # Dictionary dispatch transfers control to runtime-loaded continuations
                # on success; treat as unknown successor in CFG.
                add_uncertain_reason(
                    idx,
                    UncertainInfo("ctrl_reg_unknown", idx, opcode, "dict_dispatch"),
                )
            if opcode in DICT_DISPATCH_EXEC_OPCODES:
                # EXEC variants behave like conditional calls: on success they execute
                # continuation and may return to the next instruction.
                returning_branches.add(idx)

            # --- Phase C: Resolve continuation consumption for branches ---
            consumed_conts: List[str] = []
            inline_conts = inline_cont_map.get(idx)
            inline_list = list(inline_conts) if inline_conts else []

            stack_cont_arity = BRANCH_CONT_ARITY.get(opcode, 0) if (is_cond or is_tail or is_call) else 0
            if stack_cont_arity == 0 and _suffix_arg_count(opcode) is not None:
                stack_cont_arity = 1
            # *END loop variants take loop body from extract_cc(0) in current context.
            # ProgramAnalyzer materializes that body as an inline continuation.
            cc_body_arity = 1 if opcode in END_LOOP_OPCODES else 0
            entry_pop_count: Optional[int] = None
            entry_shape_override: Optional[Tuple[List[Tuple[str, Optional[str]]], bool]] = None
            entry_ctrl_regs_override: Optional[Dict[int, Optional[str]]] = None
            try_handler_cont: Optional[str] = None
            if stack_cont_arity > 0 or cc_body_arity > 0:
                resolved_stack_conts = 0
                if inline_list:
                    if cc_body_arity > 0:
                        consumed_conts.extend(inline_list[:cc_body_arity])
                    else:
                        # Non-*END opcodes: inline continuations satisfy branch arity.
                        inline_stack_conts = inline_list[:stack_cont_arity]
                        consumed_conts.extend(inline_stack_conts)
                        resolved_stack_conts = len(inline_stack_conts)
                if opcode in {"TRY", "TRYARGS"} and len(inline_list) >= 2:
                    # TRY* inline continuation order: [body, handler]
                    try_handler_cont = inline_list[1]

                unresolved = False
                ctrl_reg_unresolved = False
                condition_type_mismatch = False
                condition_mismatch_detail = ""
                start_depth: Optional[int] = None

                if (
                    opcode in CALL_CTRL_REG_OPCODES
                    and stack_cont_arity > resolved_stack_conts
                ):
                    reg_idx = CALL_CTRL_REG_INDEX.get(opcode)
                    reg_val = ctrl_regs.get(reg_idx) if reg_idx is not None else None
                    if isinstance(reg_val, str) and not is_unknown_cont(reg_val):
                        consumed_conts.append(reg_val)
                        resolved_stack_conts += 1
                    else:
                        # Dynamic CALL/CALLDICT targets via c3 are often external method
                        # dispatch and do not map to extracted continuation IDs.
                        ctrl_reg_unresolved = True

                stack_needed = max(stack_cont_arity - resolved_stack_conts, 0)
                consume_from_stack = (
                    stack_needed > 0
                    and opcode not in CALL_CTRL_REG_OPCODES
                    and opcode not in CALL_NO_STACK_OPCODES
                )

                if consume_from_stack:
                    if is_cond:
                        # Conditional branches: condition is on TOS, continuations below it.
                        # Peek continuations below the condition; actual pops are applied in Phase E.
                        start_depth = 1
                    elif opcode in {"TRY", "TRYARGS"}:
                        # TRY/TRYARGS direct control transfer uses body continuation.
                        # VM pops handler first, then body; body is therefore at depth 1.
                        start_depth = 1
                        if try_handler_cont is None:
                            handler_token = peek_token(0)
                            if (
                                handler_token
                                and handler_token[0] == STACK_CONT
                                and handler_token[1] is not None
                            ):
                                try_handler_cont = str(handler_token[1])
                    elif opcode in {"CALLXVARARGS", "CALLCCVARARGS"}:
                        if enable_varargs_depth_fix:
                            start_depth = _resolve_call_varargs_cont_depth()
                            if start_depth is None:
                                start_depth = _resolve_call_varargs_unknown_tail_cont_depth()
                        else:
                            start_depth = None
                    elif opcode == "JMPXVARARGS":
                        if enable_varargs_depth_fix:
                            start_depth = _resolve_jmpx_varargs_cont_depth()
                            if start_depth is None:
                                start_depth = _resolve_jmpx_varargs_unknown_tail_cont_depth()
                        else:
                            start_depth = None
                    elif opcode in CALL_ARGS_IMM_OPCODES or _suffix_arg_count(opcode) is not None:
                        # CALLXARGS/CALLCCARGS/JMPXARGS variants pop continuation from TOS.
                        start_depth = 0
                    else:
                        # Default: continuation on TOS
                        start_depth = 0

                    if start_depth is None:
                        unresolved = True
                    else:
                        stack_consumed_conts: List[str] = []
                        for offset in range(stack_needed):
                            token = peek_token(start_depth + offset)
                            if token and token[0] == STACK_CONT and token[1] is not None:
                                stack_consumed_conts.append(token[1])
                            else:
                                unresolved = True
                                break

                        # If unknown data exists below the known prefix, strict depth-based
                        # lookup can miss PUSHCONT-tracked continuations that are still
                        # present in the known slice (e.g. continuation above unknown tail).
                        if enable_known_prefix_fallback and unresolved and unknown_below:
                            fallback_conts = _resolve_known_prefix_conts(stack_needed)
                            if fallback_conts is not None:
                                stack_consumed_conts = fallback_conts
                                unresolved = False

                        # VM IF*/IFELSE* condition slot must be bool-compatible.
                        # A known continuation at condition depth is definitely ill-typed.
                        if not unresolved and is_cond:
                            cond_state = _condition_token_compat(0)
                            if cond_state == COND_INCOMPATIBLE:
                                unresolved = True
                                condition_type_mismatch = True
                                condition_mismatch_detail = "condition_slot_default"

                        # Some conditional branch forms place continuations on top of stack
                        # and keep condition below them. If the default depth fails, try that
                        # alternate layout for known conditional opcodes.
                        if (
                            unresolved
                            and is_cond
                            and opcode in ALT_TOP_CONT_CONDITIONALS
                            and stack_needed > 0
                        ):
                            alt_conts: List[str] = []
                            alt_ok = True
                            for offset in range(stack_needed):
                                token = peek_token(offset)
                                if token and token[0] == STACK_CONT and token[1] is not None:
                                    alt_conts.append(token[1])
                                else:
                                    alt_ok = False
                                    break
                            if alt_ok:
                                cond_state = _condition_token_compat(stack_needed)
                                if cond_state == COND_INCOMPATIBLE:
                                    condition_type_mismatch = True
                                    condition_mismatch_detail = "condition_slot_alt_top_cont"
                                else:
                                    stack_consumed_conts = alt_conts
                                    unresolved = False

                        if not unresolved:
                            consumed_conts.extend(stack_consumed_conts)
                elif stack_needed > 0:
                    # Control-register targets that are unresolved should still produce
                    # an unknown edge for tail jumps (e.g., JMPDICT via unknown c3).
                    if ctrl_reg_unresolved and opcode in UNCONDITIONAL_BRANCHES:
                        unresolved = True
                    elif not ctrl_reg_unresolved:
                        unresolved = True

                # Dynamic EXECUTE targets (e.g., loaded via unknown PUSHCTR) cannot be
                # mapped to extracted continuation IDs; let CFG fallback to call_return.
                if unresolved and opcode == "EXECUTE" and not consumed_conts:
                    unresolved = False

                if not unresolved:
                    if is_cond:
                        # Condition + stack-consumed continuations (inline ones don't occupy stack)
                        entry_pop_count = 1 + stack_needed
                    elif is_tail:
                        # Tail jump: only stack-consumed continuations
                        entry_pop_count = stack_needed
                    elif is_call and consume_from_stack and start_depth is not None:
                        # Continuation may be below call arguments.
                        entry_pop_count = start_depth + stack_needed
                    if opcode == "TRYARGS":
                        entry_shape_override = _project_tryargs_body_entry_shape(_int_arg(fact, 0))
                    if opcode in {"TRY", "TRYARGS"}:
                        # TRY* rewrites c0 to extract_cc(7, ...) and c2 to handler before jumping to body.
                        entry_ctrl_regs_override = dict(ctrl_regs)
                        entry_ctrl_regs_override[0] = make_unknown_cont(idx, f"{opcode}_cc")
                        if try_handler_cont is not None:
                            entry_ctrl_regs_override[2] = try_handler_cont
                        else:
                            entry_ctrl_regs_override[2] = make_unknown_cont(idx, f"{opcode}_handler")

                if unresolved:
                    # Stack state is uncertain - mark this branch for conservative handling
                    reasons: List[UncertainInfo] = []
                    if condition_type_mismatch:
                        reasons.append(
                            UncertainInfo(
                                "type_mismatch",
                                idx,
                                opcode,
                                condition_mismatch_detail or "condition_slot",
                            )
                        )

                    for token_type, token_value in stack:
                        if token_type != STACK_CONT or not isinstance(token_value, str):
                            continue
                        if not is_unknown_cont(token_value):
                            continue
                        parsed = parse_unknown_cont(token_value)
                        if parsed is None:
                            reasons.append(UncertainInfo("unresolved", -1, "legacy_unknown_cont"))
                            continue
                        src_idx, src_opcode = parsed
                        reasons.append(
                            UncertainInfo(
                                source_reason_from_opcode(src_opcode),
                                src_idx,
                                src_opcode,
                            )
                        )

                    if opcode in CALL_CTRL_REG_OPCODES:
                        reg_idx = CALL_CTRL_REG_INDEX.get(opcode)
                        reg_val = ctrl_regs.get(reg_idx) if reg_idx is not None else None
                        if is_unknown_cont(reg_val):
                            detail = f"c{reg_idx}" if reg_idx is not None else "unknown_reg"
                            parsed = parse_unknown_cont(reg_val or "")
                            if parsed is None:
                                reasons.append(
                                    UncertainInfo("ctrl_reg_unknown", -1, "legacy_unknown_cont", detail)
                                )
                            else:
                                src_idx, src_opcode = parsed
                                reasons.append(
                                    UncertainInfo("ctrl_reg_unknown", src_idx, src_opcode, detail)
                                )

                    if not reasons and last_invalidate_reason is not None:
                        reasons.append(last_invalidate_reason)

                    if not reasons:
                        reasons.append(UncertainInfo("unresolved", idx, opcode))

                    for info in reasons:
                        add_uncertain_reason(idx, info)

            # --- Phase D: Update return target mapping for returning branches ---
                if consumed_conts:
                    branch_cont_map[idx] = consumed_conts
                    if idx not in uncertain_branches:
                        _record_entry_shapes(consumed_conts, entry_pop_count, entry_shape_override)
                        if opcode in {"CALLCC", "CALLCCARGS", "CALLCCVARARGS"}:
                            _prepend_entry_shape_token(
                                consumed_conts,
                                (STACK_CONT, make_unknown_cont(idx, f"{opcode}_cc")),
                            )
                        _record_entry_ctrl_regs(consumed_conts, entry_ctrl_regs_override)
                    if opcode in LOOP_CALL_OPCODES:
                        # Loop bodies return to the loop header for the next iteration.
                        for cont_id in consumed_conts:
                            cont_return_targets.setdefault(cont_id, []).append(idx)
                        # Loops may also return to the post-loop instruction
                        # (e.g., completion, break/RETALT path).
                        if i + 1 < len(facts):
                            return_target = facts[i + 1].index
                            for cont_id in consumed_conts:
                                cont_return_targets.setdefault(cont_id, []).append(return_target)
                    elif returns_to_caller:
                        returning_branches.add(idx)
                        if i + 1 < len(facts):
                            return_target = facts[i + 1].index
                            for cont_id in consumed_conts:
                                cont_return_targets.setdefault(cont_id, []).append(return_target)

            # --- Phase E: Apply stack effects ---
            # E.1: Handle modeled stack transforms (DUP, SWAP, etc.)
            if apply_stack_transform(opcode, fact):
                continue

            # E.2: Handle unmodeled shuffles - invalidate stack tracking
            if is_unmodeled_shuffle(opcode):
                invalidate_stack("stack_invalidated", idx, opcode, "unmodeled_shuffle")
                continue

            # E.3: Apply standard stack delta for other instructions
            # Use StackEffectDatabase directly as single source of truth
            if is_call:
                # Calls: skip Phase E delta. Smart stack reset below handles precisely.
                pass
            else:
                effect = get_stack_effect(opcode)
                if effect is not None:
                    min_net, max_net = effect.net_effect_range()
                    if min_net is None:
                        min_net = effect.net_effect
                    # Use exact delta if deterministic, else use min_net
                    if max_net is not None and min_net == max_net and not effect.is_dynamic:
                        delta = min_net
                        known = True
                    else:
                        delta = min_net
                        known = (max_net is not None and min_net == max_net)
                else:
                    # Truly unknown opcode - try prefix rules
                    upper_op = opcode.upper()
                    if upper_op.startswith("PUSH"):
                        delta, known = 1, True
                    elif upper_op.startswith("POP") or upper_op.startswith("DROP"):
                        delta, known = -1, True
                    elif upper_op.startswith("XCHG") or upper_op.startswith("SWAP"):
                        delta, known = 0, True
                    else:
                        delta, known = 0, False

                if is_cond or is_tail:
                    if not known:
                        invalidate_stack("dynamic_effect", idx, opcode, "unknown_stack_delta")
                    else:
                        if delta < 0:
                            pop_value(-delta)
                        elif delta > 0:
                            push_value(delta)
                else:
                    if not known:
                        invalidate_stack("dynamic_effect", idx, opcode, "unknown_stack_delta")
                    elif delta < 0:
                        pop_value(-delta)
                    elif delta > 0:
                        push_value(delta)

            # Smart Stack Reset: precisely model stack consumption at control-flow
            # boundaries instead of conservatively clearing the entire stack.
            #
            # TVM semantics: branch/call instructions consume specific items from
            # the stack (condition + continuations for branches, arguments for
            # calls), but elements BELOW the consumed items are preserved.
            if opcode in TERMINATORS:
                # Terminators (RET, THROW, JMP*): execution doesn't continue here.
                # Stack state is irrelevant for subsequent instructions.
                invalidate_stack("terminator", idx, opcode)
            elif opcode in RETURNING_CONDITIONALS:
                # RETURNING_CONDITIONALS (IF, IFNOT, IFELSE, IFREF*, etc.):
                # These execute a continuation and then RETURN to the next
                # instruction.  They consume: 1 condition (TOS) + N continuations
                # (from BRANCH_CONT_ARITY).  The stack delta at Phase E already
                # popped these items, so the remaining stack is preserved.
                #
                # Nothing extra to do -- the pop_value() above was sufficient.
                pass
            elif is_call:
                # Call instructions: consume continuation + arguments, callee
                # returns values. Phase E skips calls entirely (line 795-797),
                # so stack consumption must be modeled here.
                return_count: Optional[int] = 1
                if opcode in {"CALLDICT", "CALLDICT_LONG", "CALL", "CALLREF"}:
                    # Continuation target is not consumed from tracked stack.
                    # Keep default return_count=1.
                    pass
                elif opcode in {"CALLX", "EXECUTE", "CALLCC"}:
                    # Consume 1 continuation from TOS.
                    pop_value(1)
                elif opcode.startswith(("CALLXARGS", "CALLCCARGS")):
                    # Immediate/suffixed arg count: consume continuation + p args.
                    p = _suffix_arg_count(opcode)
                    if p is None:
                        p = _int_arg(fact, 0)
                    r = _int_arg(fact, 1)
                    if p is not None and p >= 0:
                        pop_value(1 + p)
                    else:
                        # Keep stack shape partially precise instead of full invalidation.
                        effect = get_stack_effect(opcode)
                        min_inputs = effect.min_inputs if effect and effect.min_inputs is not None else 1
                        pop_value(max(min_inputs, 1))
                        unknown_below = True
                    if r is not None and r >= 0:
                        return_count = r
                    else:
                        # Variable/unknown return arity (e.g., r=-1) shifts all
                        # remaining pre-call stack values by an unknown amount.
                        # Drop precise depth tracking to avoid false target resolution.
                        return_count = 1
                        invalidate_stack("dynamic_effect", idx, opcode, "unknown_return_arity")
                elif opcode in {"CALLXVARARGS", "CALLCCVARARGS"}:
                    p = _resolve_call_varargs_p()
                    r_val: Optional[int] = None
                    r_token = peek_token(0)
                    if (
                        r_token
                        and r_token[0] == STACK_VALUE
                        and isinstance(r_token[1], int)
                    ):
                        r_val = r_token[1]
                    if p is not None:
                        if p >= 0:
                            pop_value(p + 3)
                        else:
                            # p = -1 passes the entire current stack as call args.
                            # Caller-side stack remainder becomes empty before returns.
                            stack.clear()
                            unknown_below = False
                    else:
                        invalidate_stack("dynamic_effect", idx, opcode, "unknown_varargs_params")
                    if r_val is not None:
                        if r_val >= 0:
                            return_count = r_val
                        else:
                            return_count = 1
                            unknown_below = True
                elif opcode == "TRY":
                    # TRY consumes (body_cont, handler_cont) and returns no stack values.
                    pop_value(2)
                    return_count = 0
                elif opcode == "TRYARGS":
                    # TRYARGS is TRY + CALLXARGS-style argument passing.
                    # Consume p arguments plus two continuations (body + handler).
                    p = _int_arg(fact, 0)
                    if p is not None and p >= 0:
                        pop_value(p + 2)
                    else:
                        pop_value(2)
                        unknown_below = True
                    r = _int_arg(fact, 1)
                    if r is not None:
                        if r >= 0:
                            return_count = r
                        else:
                            return_count = 0
                            unknown_below = True
                    else:
                        return_count = 0
                else:
                    # Fallback for less-common call aliases.
                    effect = get_stack_effect(opcode)
                    if effect is None:
                        invalidate_stack("dynamic_effect", idx, opcode, "unknown_call_effect")
                        continue
                    min_inputs = effect.min_inputs if effect.min_inputs is not None else effect.inputs
                    min_outputs = effect.min_outputs if effect.min_outputs is not None else effect.outputs
                    pop_value(max(min_inputs, 0))
                    return_count = max(min_outputs, 0)
                    if (
                        effect.is_dynamic
                        or effect.max_inputs != effect.min_inputs
                        or effect.max_outputs != effect.min_outputs
                    ):
                        unknown_below = True

                if return_count is not None and return_count > 0:
                    push_value(return_count)

        return (
            branch_cont_map,
            returning_branches,
            cont_return_targets,
            uncertain_branches,
            dict(ctrl_regs),
        )
