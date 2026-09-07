"""
Taint propagation for TVM data flow analysis.

Contains methods for propagating taint through instructions and detecting
when tainted data reaches sensitive operations.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from ...config import (
    MESSAGE_CONTEXT_STACK_DEPTH,
    DEFAULT_SENSITIVE_CHECK_DEPTH,
)
from ..stack_effects import get_stack_effect
from ...analyzer.utils import extract_int_arg

from .types import DataFlowValue, ValueSource


class TaintPropagationMixin:
    """
    Mixin class providing taint propagation methods for DataFlowAnalyzer.

    This mixin implements taint tracking logic for detecting when untrusted
    data flows to sensitive operations without proper validation.
    """

    # Type hints for attributes provided by DataFlowAnalyzer
    current_state: Any
    graph: Any
    TAINT_SOURCES: Dict[str, ValueSource]
    CONDITIONAL_TAINT_SOURCES: Dict[str, ValueSource]
    MESSAGE_SLICE_LOADERS: Set[str]
    CONTEXT_DEPENDENT_LOADERS: Set[str]
    SENSITIVE_OPCODES: Set[str]
    DYNAMIC_CALL_OPCODES: Set[str]

    # _update_stack_for_opcode is inherited from StackSimulatorMixin via MRO
    # No need to declare stub here - resolved through Python's MRO

    @staticmethod
    def _taint_sources_from_value(value: Optional[DataFlowValue]) -> Set[str]:
        if not value or not value.tainted:
            return set()
        meta = value.metadata or {}
        sources = set(meta.get("taint_sources", []) or [])
        if not sources and isinstance(value.source, ValueSource):
            sources.add(value.source.value)
        return sources

    @staticmethod
    def _taint_origins_from_value(value: Optional[DataFlowValue]) -> Set[int]:
        """Get taint origins from a DataFlowValue."""
        if not value or not value.tainted:
            return set()
        meta = value.metadata or {}
        origins = set(meta.get("taint_origins", []) or [])
        if not origins:
            origins.add(value.definition_site)
        return origins

    @staticmethod
    def _const_from_value(value: Optional[DataFlowValue]) -> Optional[int]:
        if not value:
            return None
        const_val = value.metadata.get("const")
        return const_val if isinstance(const_val, int) else None

    @staticmethod
    def _try_resolve_constant_index(value: Optional[DataFlowValue]) -> Optional[int]:
        """Attempt to resolve a constant integer index from a DataFlowValue.

        Used to optimize dynamic stack operations (XCHGX, PICK, etc.) by resolving
        the index at analysis time when possible, avoiding full stack invalidation.

        Args:
            value: The DataFlowValue potentially containing a constant index

        Returns:
            The constant integer index if resolvable, None otherwise
        """
        if not value:
            return None
        const_val = value.metadata.get("const")
        if isinstance(const_val, int) and const_val >= 0:
            return const_val
        return None

    @staticmethod
    def _get_first_int_arg(inst) -> Optional[int]:
        """Extract first integer argument from instruction.

        Uses shared extract_int_arg utility for consistency with
        ProgramAnalyzer and ContinuationResolver.
        """
        return extract_int_arg(inst.arguments, 0)

    def _merge_taint_metadata(self, values: List[Optional[DataFlowValue]]) -> Tuple[Set[str], Set[int]]:
        sources: Set[str] = set()
        origins: Set[int] = set()
        for val in values:
            sources.update(self._taint_sources_from_value(val))
            origins.update(self._taint_origins_from_value(val))
        return sources, origins

    def _build_taint_metadata(
        self, input_values: List[Optional[DataFlowValue]]
    ) -> Tuple[List[str], List[int]]:
        """Extract and flatten taint sources and origins from input values.

        Args:
            input_values: List of input DataFlowValue objects

        Returns:
            Tuple of (taint_sources, taint_origins) as lists
        """
        sources, origins = self._merge_taint_metadata(input_values)
        return list(sources), list(origins)

    def _resolve_varargs_count(self) -> Optional[int]:
        """
        Attempt to resolve varargs count p from stack.

        CALL*VARARGS stack layout (top first):
          args (p), cont, p, r
        We detect p when a constant integer is located at depth p+1.
        """
        for depth, val in enumerate(self.current_state.stack):
            const_val = self._const_from_value(val)
            if const_val is None or const_val < 0:
                continue
            if depth != const_val + 1:
                continue
            if len(self.current_state.stack) <= depth + 1:
                continue  # r not present
            # stack[const_val] is the continuation at position p (= depth - 1)
            # where p is the number of arguments for CALLXVARARGS/CALLCCVARARGS
            cont_val = self.current_state.stack[const_val] if const_val < len(self.current_state.stack) else None
            if cont_val is None:
                continue
            if cont_val.metadata.get("is_continuation") is False:
                continue
            return const_val
        return None

    def _is_in_message_context(self) -> bool:
        """
        Check if current context involves message-derived data (context-aware taint handling).

        Returns True if stack or registers contain values that originate from a message.
        Checks multiple stack positions and registers for comprehensive detection.

        This helps distinguish LDU/LDI from message body vs from storage/constants.
        """
        MESSAGE_SOURCES = {
            ValueSource.MESSAGE_BODY,
            ValueSource.MESSAGE_SENDER,
            ValueSource.MESSAGE_VALUE,
            ValueSource.MESSAGE_FLAGS,
        }

        for val in self.current_state.stack[:MESSAGE_CONTEXT_STACK_DEPTH]:
            if val is None:
                continue
            # Check if value is tainted from message source
            if val.tainted and val.source in MESSAGE_SOURCES:
                return True
            # Check if value has message-related metadata
            if val.metadata.get("from_message_slice", False):
                return True

        for reg_val in self.current_state.registers.values():
            if reg_val is None:
                continue
            if reg_val.tainted and reg_val.source in MESSAGE_SOURCES:
                return True
            if reg_val.metadata.get("from_message_slice", False):
                return True

        return False

    def _handle_taint_source(self, opcode: str, idx: int, args) -> bool:
        """
        Handle taint source opcodes that always introduce tainted data.

        Returns True if the opcode was handled, False otherwise.
        """
        if opcode not in self.TAINT_SOURCES:
            return False

        source = self.TAINT_SOURCES[opcode]
        self._update_stack_for_opcode(
            opcode,
            idx,
            args,
            force_taint=True,
            force_source=source,
            force_metadata={
                "taint_sources": [source.value],
                "taint_origins": [idx],
            },
        )
        return True

    def _handle_conditional_taint_source(self, opcode: str, idx: int, args) -> bool:
        """
        Handle context-dependent taint sources (LDU/LDI).

        Returns True if the opcode was handled, False otherwise.
        """
        if opcode not in self.CONDITIONAL_TAINT_SOURCES:
            return False

        # Check if we're loading from a message-related slice
        is_message_context = self._is_in_message_context()

        if is_message_context:
            source = self.CONDITIONAL_TAINT_SOURCES[opcode]
            self._update_stack_for_opcode(
                opcode,
                idx,
                args,
                force_taint=True,
                force_source=source,
                force_metadata={
                    "taint_sources": [source.value],
                    "taint_origins": [idx],
                },
            )
        else:
            # Not from message - could be from storage or constant
            # Use clean unknowns so missing inputs don't auto-taint outputs
            self._update_stack_for_opcode(
                opcode,
                idx,
                args,
                unknown_taint=False,
            )
        return True

    def _handle_message_slice_loader(self, opcode: str, idx: int, args) -> bool:
        """
        Handle message slice loaders that mark values as from message context.

        Returns True if the opcode was handled, False otherwise.
        """
        if opcode not in self.MESSAGE_SLICE_LOADERS:
            return False

        # Create a value marked as from message context
        self._update_stack_for_opcode(
            opcode,
            idx,
            args,
            force_taint=True,
            force_source=ValueSource.MESSAGE_BODY,
            force_metadata={
                "from_message_slice": True,
                "taint_sources": [ValueSource.MESSAGE_BODY.value],
                "taint_origins": [idx],
            },
        )
        return True

    def _handle_context_dependent_loader(self, opcode: str, idx: int) -> bool:
        """
        Handle context-dependent loaders like CTOS.

        Returns True if the opcode was handled, False otherwise.
        """
        if opcode not in self.CONTEXT_DEPENDENT_LOADERS:
            return False

        # CTOS consumes one Cell from stack top (after any stack manipulation)
        # We need to check if the Cell being consumed is from message context
        # The Cell could be at stack top OR could have been moved there via SWAP/ROT

        # Check if the input (Cell) comes from message context
        # This handles cases where Cell was moved via stack operations like SWAP, ROT, etc.
        is_from_message = False
        consumed_cell = None

        if self.current_state.stack:
            # First check stack top (most common case)
            top_cell = self.current_state.stack[0]
            if top_cell:
                consumed_cell = top_cell
                is_from_message = top_cell.tainted and top_cell.source in {
                    ValueSource.MESSAGE_BODY,
                    ValueSource.MESSAGE_SENDER,
                    ValueSource.MESSAGE_VALUE,
                }

            # Conservative fallback: If stack top is not message-derived,
            # check if ANY stack position has from_message_slice metadata.
            # This handles cases where the Cell was moved via SWAP/ROT operations
            # (e.g., LDREF -> SWAP -> ... -> CTOS).
            #
            # Trade-off: This is a conservative over-approximation that may produce
            # false positives, but ensures we don't miss real vulnerabilities where
            # message-derived Cells are consumed after stack manipulation.
            if not is_from_message:
                for stack_val in self.current_state.stack:
                    if stack_val and stack_val.metadata.get("from_message_slice", False):
                        # Found a message-derived Cell somewhere in the stack.
                        # Conservatively assume it might be the one consumed by CTOS.
                        is_from_message = True
                        if consumed_cell is None:
                            consumed_cell = stack_val
                        break

        if is_from_message:
            # Cell is from message, so Slice should be tainted
            value = DataFlowValue(
                source=ValueSource.MESSAGE_BODY,
                definition_site=idx,
                tainted=True,
                metadata={
                    "from_message_slice": True,
                    "ctos_tainted": True,
                    "taint_sources": [ValueSource.MESSAGE_BODY.value],
                    "taint_origins": [idx],
                }
            )
        else:
            # Cell is from storage/code, Slice is not tainted
            value = DataFlowValue(
                source=ValueSource.COMPUTATION,
                definition_site=idx,
                tainted=False,
                metadata={"ctos_clean": True}
            )

        # CTOS pops Cell, pushes Slice
        # First pop the Cell from stack top
        if self.current_state.stack:
            self.current_state.stack.pop(0)
        # Then push the new Slice
        self.current_state.stack.insert(0, value)
        # Record in graph
        self.graph.values.setdefault(idx, []).append(value)
        return True

    def _handle_sensitive_operation(self, opcode: str, idx: int) -> None:
        """
        Handle sensitive operations and check for unchecked tainted data.
        """
        if opcode not in self.SENSITIVE_OPCODES:
            return

        # Check if we're using tainted unchecked data
        if not self.current_state.stack:
            return

        # Determine check depth based on opcode's actual stack inputs
        effect = get_stack_effect(opcode)
        if effect:
            if effect.max_inputs is not None:
                check_depth = min(effect.max_inputs, len(self.current_state.stack))
            else:
                check_depth = len(self.current_state.stack)
        else:
            check_depth = DEFAULT_SENSITIVE_CHECK_DEPTH

        seen_pairs: Set[Tuple[int, int]] = set()
        for stack_val in self.current_state.stack[:check_depth]:
            if stack_val and stack_val.tainted and not stack_val.checked:
                # Found tainted unchecked value used in sensitive op!
                origins = self._taint_origins_from_value(stack_val)
                if not origins:
                    origins = {stack_val.definition_site}
                for origin in origins:
                    pair = (origin, idx)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if pair not in self.graph._taint_dedup:
                        self.graph._taint_dedup.add(pair)
                        self.graph.tainted_propagation.append(pair)

    def _handle_dynamic_call(self, opcode: str, idx: int) -> None:
        """
        Handle dynamic call operations (Approach A: precise taint tracking).

        Dynamic calls take the call target from stack top - if tainted, this is a security issue.
        """
        if opcode not in self.DYNAMIC_CALL_OPCODES:
            return

        if not self.current_state.stack:
            return

        # Check stack top (call target position) for tainted unchecked values
        top_value = self.current_state.stack[0]
        if top_value and top_value.tainted and not top_value.checked:
            # Tainted unchecked value used as dynamic call target!
            origins = self._taint_origins_from_value(top_value)
            if not origins:
                origins = {top_value.definition_site}
            for origin in origins:
                pair = (origin, idx)
                if pair not in self.graph._taint_dedup:
                    self.graph._taint_dedup.add(pair)
                    self.graph.tainted_propagation.append(pair)
