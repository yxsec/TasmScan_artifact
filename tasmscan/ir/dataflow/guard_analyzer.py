"""
Guard analysis for TVM data flow analysis.

Contains methods for detecting and processing guard operations that
validate values before use (IF, THROW, THROWIF, etc.).
"""
from typing import Any, Optional, Set

from ...config import MAX_GUARDED_VALUES

from .types import DataFlowValue


class GuardAnalyzerMixin:
    """
    Mixin class providing guard analysis methods for DataFlowAnalyzer.

    This mixin implements guard detection and marking logic for tracking
    which values have been validated before use in sensitive operations.
    """

    # Type hints for attributes provided by DataFlowAnalyzer
    current_state: Any
    _guard_access_counter: int

    # Methods are inherited from TaintPropagationMixin via MRO
    # No need to declare stubs here - they are resolved through Python's MRO

    def _add_guarded_value(self, definition_site: int) -> None:
        """
        Add a definition site to guarded_values with LRU tracking.

    Improved handling: Uses access counter for LRU-based pruning instead of
        numeric value sorting, ensuring early critical guards (like
        entry-point sender checks) are not deleted prematurely.
        """
        self._guard_access_counter += 1
        self.current_state.guarded_values[definition_site] = self._guard_access_counter

    def _mark_guarded(self, value: Optional[DataFlowValue], confidence: str = "high") -> None:
        """
        Mark a value as guarded (checked).

        Args:
            value: The value to mark as guarded
            confidence: Guard confidence level ("high", "medium", "low")
                - "high": guard checks a comparison result involving tainted values
                - "medium": guard checks a tainted value directly
                - "low": guard checks a non-tainted value (likely unrelated check)
        """
        if not value:
            return

        value.checked = True
        value.metadata["guard_confidence"] = confidence

        # If this is a comparison result, trace back to mark original operands as checked
        # This handles the pattern: LDMSGADDR -> EQUAL -> THROWIFNOT
        # where THROWIFNOT checks the EQUAL result, but we need to mark the sender as checked
        comparison_operands = value.metadata.get("comparison_operands", [])
        if comparison_operands:
            for stack_val in self.current_state.stack:
                if stack_val and stack_val.definition_site in comparison_operands:
                    stack_val.checked = True
                    stack_val.metadata["guard_confidence"] = confidence
            for operand in comparison_operands:
                self._add_guarded_value(operand)

        # Handle taint_origins for propagated taint.
        # Restrict origin-chain guarding to high-confidence guards only:
        # medium-confidence checks (guarding a tainted value directly) may be
        # unrelated to specific upstream taint origins.
        origins = self._taint_origins_from_value(value)
        if confidence == "high" and origins:
            for origin in origins:
                self._add_guarded_value(origin)
            for stack_val in self.current_state.stack:
                if stack_val and stack_val.definition_site in origins:
                    stack_val.checked = True
                    stack_val.metadata["guard_confidence"] = confidence
        else:
            self._add_guarded_value(value.definition_site)
            # Preserve guarding semantics for multi-output instructions (e.g. LDMSGADDR)
            # where sibling outputs share the same definition site.
            for stack_val in self.current_state.stack:
                if stack_val and stack_val.definition_site == value.definition_site:
                    stack_val.checked = True
                    stack_val.metadata["guard_confidence"] = confidence

        # instead of numeric value sorting which could delete early critical guards
        if len(self.current_state.guarded_values) > MAX_GUARDED_VALUES:
            # Sort by access_count (ascending), keep top half with highest access counts
            sorted_by_access = sorted(
                self.current_state.guarded_values.items(),
                key=lambda x: x[1]  # Sort by access_count
            )
            # Keep the more recently accessed half
            keep_count = len(sorted_by_access) // 2
            self.current_state.guarded_values = dict(sorted_by_access[keep_count:])

    def _process_guard_instruction(
        self,
        opcode: str,
        guard_opcodes: Set[str],
    ) -> None:
        """
        Process a guard instruction and mark appropriate values as checked.

        Args:
            opcode: The opcode being processed
            guard_opcodes: Set of opcodes that act as guards
        """
        if opcode not in guard_opcodes:
            return

        # Only mark values as checked when there's evidence they participated in the guard condition
        if self.current_state.stack:
            top_value = self.current_state.stack[0]
            if top_value:
                # Check if this guard is checking a comparison result
                comparison_operands = top_value.metadata.get("comparison_operands", [])
                if comparison_operands:
                    # High confidence: guard is checking a comparison involving these values
                    self._mark_guarded(top_value, confidence="high")
                elif top_value.tainted:
                    # Medium confidence: guard checks tainted value directly
                    self._mark_guarded(top_value, confidence="medium")
                else:
                    # Low confidence: guard checks non-tainted value
                    pass
