"""
Stack simulation for TVM data flow analysis.

Contains methods for simulating TVM stack operations including:
- Basic stack operations (PUSH, DUP, SWAP, ROT, etc.)
- Dynamic stack shuffles (XCHGX, PICK, etc.)
- Stack effect-based generic handling
"""
import copy
from typing import Any, Dict, List, Optional, Set

from ...analyzer.utils import extract_int_arg
from ..stack_effects import get_stack_effect
from .types import DataFlowValue, ValueSource


class StackSimulatorMixin:
    """
    Mixin class providing stack simulation methods for DataFlowAnalyzer.

    This mixin implements the `_update_stack_for_opcode` method and related
    helper methods for simulating TVM stack operations.
    """

    # Type hints for attributes provided by DataFlowAnalyzer
    current_state: Any
    graph: Any
    CONTINUATION_PUSH_OPCODES: Set[str]
    CALL_VARARGS_OPCODES: Set[str]
    DYNAMIC_SHUFFLE_OPCODES: Set[str]
    MULTI_OUTPUT_TAINT_RULES: Dict[str, List[str]]

    # Methods are inherited from TaintPropagationMixin via MRO
    # No need to declare stubs here - they are resolved through Python's MRO

    def _update_stack_for_opcode(
        self,
        opcode: str,
        idx: int,
        args=None,
        *,
        force_taint: Optional[bool] = None,
        force_source: Optional[ValueSource] = None,
        force_metadata: Optional[Dict[str, Any]] = None,
        unknown_taint: bool = True,
    ):
        """
        Update stack state based on opcode.

        This is a simplified simulation - full TVM stack behavior is complex.
        """
        args = args or []

        def _int_arg(pos: int) -> Optional[int]:
            return extract_int_arg(args, pos)

        # Constant integer pushes (used for VARARGS resolution)
        if opcode.upper().startswith("PUSHINT"):
            const_val = _int_arg(0)
            metadata: Dict[str, Any] = {}
            if const_val is not None:
                metadata["const"] = const_val
            value = DataFlowValue(
                source=ValueSource.CONSTANT,
                definition_site=idx,
                tainted=False,
                metadata=metadata,
            )
            self.current_state.stack.insert(0, value)
            self.graph.values.setdefault(idx, []).append(value)
            return

        # Continuation pushes (code, not data)
        if opcode.upper() in self.CONTINUATION_PUSH_OPCODES:
            value = DataFlowValue(
                source=ValueSource.COMPUTATION,
                definition_site=idx,
                tainted=False,
                metadata={"is_continuation": True},
            )
            self.current_state.stack.insert(0, value)
            self.graph.values.setdefault(idx, []).append(value)
            return

        # Varargs calls: trim stack to argument list when p is resolvable
        if opcode.upper() in self.CALL_VARARGS_OPCODES:
            p = self._resolve_varargs_count()
            if p is None:
                self.graph.analysis_metadata.setdefault("varargs_unresolved", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                })
            else:
                self.current_state.stack = self.current_state.stack[:p]
            return

        def _warn_stack(required: int) -> None:
            self.graph.analysis_metadata.setdefault("stack_depth_warnings", []).append({
                "instruction": idx,
                "opcode": opcode,
                "required": required,
                "actual": len(self.current_state.stack),
            })

        def _warn_stack_args(reason: str) -> None:
            self.graph.analysis_metadata.setdefault("stack_arg_warnings", []).append({
                "instruction": idx,
                "opcode": opcode,
                "reason": reason,
            })

        def _unknown_value(reason: str, *, tainted: Optional[bool] = None) -> DataFlowValue:
            value_tainted = unknown_taint if tainted is None else tainted
            value = DataFlowValue(
                source=ValueSource.UNKNOWN,
                definition_site=idx,
                tainted=value_tainted,
                metadata={"unknown_reason": reason},
            )
            return value

        def _invalidate_stack(reason: str) -> None:
            if not self.current_state.stack:
                return
            for i in range(len(self.current_state.stack)):
                self.current_state.stack[i] = _unknown_value(reason)

        def _ensure_depth(required: int, reason: str) -> None:
            if required <= 0:
                return
            if len(self.current_state.stack) < required:
                _warn_stack(required)
                missing = required - len(self.current_state.stack)
                for _ in range(missing):
                    # Append to bottom so existing top-of-stack order is preserved.
                    self.current_state.stack.append(_unknown_value(reason))

        def _merge_forced_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            merged = dict(metadata) if metadata else {}
            if force_metadata:
                for key, value in force_metadata.items():
                    # Prefer forced metadata when present
                    merged[key] = value
            return merged

        def _swap(i: int, j: int) -> bool:
            if i < 0 or j < 0:
                _warn_stack_args("negative stack index")
                return False
            max_idx = max(i, j)
            if max_idx >= len(self.current_state.stack):
                _ensure_depth(max_idx + 1, "swap")
            if i == j:
                return True
            self.current_state.stack[i], self.current_state.stack[j] = (
                self.current_state.stack[j],
                self.current_state.stack[i],
            )
            return True

        def _copy_from(i: int) -> bool:
            if i < 0:
                _warn_stack_args("negative stack index")
                return False
            if i >= len(self.current_state.stack):
                _ensure_depth(i + 1, "copy_from")
            original = self.current_state.stack[i] if i < len(self.current_state.stack) else None
            if original:
                copied = original.copy()
            else:
                copied = _unknown_value("copy_from")
            self.current_state.stack.insert(0, copied)
            return True

        # Control register loads/stores
        if opcode == "PUSHCTR":
            reg_idx = _int_arg(0)
            if reg_idx is None:
                value = _unknown_value("pushctr_missing_register_index")
            else:
                reg_value = self.current_state.registers.get(reg_idx)
                value = reg_value.copy() if reg_value is not None else _unknown_value(
                    "pushctr_uninitialized_register"
                )
            self.current_state.stack.insert(0, value)
            self.graph.values.setdefault(idx, []).append(value)
            return

        if opcode == "POPCTR":
            reg_idx = _int_arg(0)
            _ensure_depth(1, "popctr")
            popped = self.current_state.stack.pop(0) if self.current_state.stack else _unknown_value("popctr")
            if reg_idx is None:
                # Unknown destination register index: tracked register state becomes uncertain.
                if self.current_state.registers:
                    for tracked_reg in list(self.current_state.registers.keys()):
                        self.current_state.registers[tracked_reg] = _unknown_value("popctr_dynamic_register")
                self.graph.analysis_metadata.setdefault("register_warnings", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "reason": "missing register index for POPCTR",
                })
            else:
                self.current_state.registers[reg_idx] = popped.copy() if popped is not None else None
            return

        if opcode == "PUSHCTRX":
            _ensure_depth(1, "pushctrx")
            # Dynamic register index pop
            if self.current_state.stack:
                self.current_state.stack.pop(0)
            any_tainted_register = any(
                v is not None and v.tainted for v in self.current_state.registers.values()
            )
            value = _unknown_value(
                "pushctrx_dynamic_register",
                tainted=any_tainted_register,
            )
            self.current_state.stack.insert(0, value)
            self.graph.values.setdefault(idx, []).append(value)
            self.graph.analysis_metadata.setdefault("register_warnings", []).append({
                "instruction": idx,
                "opcode": opcode,
                "reason": "dynamic register index for PUSHCTRX",
            })
            return

        if opcode == "POPCTRX":
            _ensure_depth(2, "popctrx")
            # Pops value and dynamic register index
            if self.current_state.stack:
                self.current_state.stack.pop(0)
            if self.current_state.stack:
                self.current_state.stack.pop(0)
            if self.current_state.registers:
                for tracked_reg in list(self.current_state.registers.keys()):
                    self.current_state.registers[tracked_reg] = _unknown_value("popctrx_dynamic_register")
            self.graph.analysis_metadata.setdefault("register_warnings", []).append({
                "instruction": idx,
                "opcode": opcode,
                "reason": "dynamic register index for POPCTRX",
            })
            return

        if opcode == "SETRETCTR":
            _ensure_depth(1, "setretctr")
            popped = self.current_state.stack.pop(0) if self.current_state.stack else _unknown_value("setretctr")
            self.current_state.registers[0] = popped.copy() if popped is not None else None
            return

        if opcode == "SETALTCTR":
            _ensure_depth(1, "setaltctr")
            popped = self.current_state.stack.pop(0) if self.current_state.stack else _unknown_value("setaltctr")
            self.current_state.registers[1] = popped.copy() if popped is not None else None
            return

        # Common patterns
        if opcode == "PUSH":
            # PUSH s[i]: duplicate stack item at index i to top (s0)
            push_idx = _int_arg(0)
            if push_idx is None:
                push_idx = 0
            _copy_from(push_idx)

        elif opcode == "DUP":
            _copy_from(0)

        elif opcode in {"DUP2", "2DUP"}:
            # Duplicate top two stack elements: a b -> a b a b
            _ensure_depth(2, "dup2")
            first = self.current_state.stack[0] if len(self.current_state.stack) > 0 else None
            second = self.current_state.stack[1] if len(self.current_state.stack) > 1 else None
            first_copy = first.copy() if first else _unknown_value("dup2")
            second_copy = second.copy() if second else _unknown_value("dup2")
            self.current_state.stack.insert(0, second_copy)
            self.current_state.stack.insert(0, first_copy)

        elif opcode == "OVER":
            # Copy s1 to top of stack
            _copy_from(1)

        elif opcode == "OVER2":
            # Copy second pair: a b c d -> a b c d a b (a,b copied to top)
            _ensure_depth(4, "over2")
            third = self.current_state.stack[2] if len(self.current_state.stack) > 2 else None
            fourth = self.current_state.stack[3] if len(self.current_state.stack) > 3 else None
            third_copy = third.copy() if third else _unknown_value("over2")
            fourth_copy = fourth.copy() if fourth else _unknown_value("over2")
            self.current_state.stack.insert(0, fourth_copy)
            self.current_state.stack.insert(0, third_copy)

        elif opcode == "POP":
            if len(self.current_state.stack) > 0:
                self.current_state.stack.pop(0)

        elif opcode == "DROP":
            if len(self.current_state.stack) > 0:
                self.current_state.stack.pop(0)

        elif opcode == "NIP":
            # Remove second item (s1), keep top
            _ensure_depth(2, "nip")
            self.current_state.stack.pop(1)

        elif opcode == "TUCK":
            # Duplicate top and insert below second element: a b -> a b a
            _ensure_depth(2, "tuck")
            original = self.current_state.stack[0] if self.current_state.stack else None
            copied = original.copy() if original else _unknown_value("tuck")
            self.current_state.stack.insert(2, copied)

        elif opcode in {"SWAP", "XCHG"}:
            # Swap stack items
            _swap(0, 1)

        elif opcode == "ROT":
            # Rotate left: a b c -> b c a
            _ensure_depth(3, "rot")
            self.current_state.stack[0], self.current_state.stack[1], self.current_state.stack[2] = (
                self.current_state.stack[1],
                self.current_state.stack[2],
                self.current_state.stack[0],
            )

        elif opcode in {"ROTREV", "-ROT"}:
            # Reverse rotation: a b c -> c a b
            _ensure_depth(3, "rotrev")
            self.current_state.stack[0], self.current_state.stack[1], self.current_state.stack[2] = (
                self.current_state.stack[2],
                self.current_state.stack[0],
                self.current_state.stack[1],
            )

        elif opcode == "REVERSE":
            # Reverse top n elements (default 2)
            n = _int_arg(0)
            if n is None:
                n = 2
            if n < 0:
                _warn_stack_args("negative REVERSE length")
            else:
                _ensure_depth(n, "reverse")
                self.current_state.stack[:n] = list(reversed(self.current_state.stack[:n]))

        elif opcode in {"XCHG_0I", "XCHG_0I_LONG"}:
            swap_idx = _int_arg(0)
            if swap_idx is None:
                _warn_stack_args("missing stack index for XCHG_0I")
            else:
                _swap(0, swap_idx)

        elif opcode == "XCHG_1I":
            # XCHG_1I swaps s1 with s[i], where i is the first argument
            swap_idx = _int_arg(0)
            if swap_idx is None:
                _warn_stack_args("missing stack index for XCHG_1I")
            else:
                _swap(1, swap_idx)

        elif opcode == "XCHG_IJ":
            i = _int_arg(0)
            j = _int_arg(1)
            if i is None or j is None:
                _warn_stack_args("missing stack indices for XCHG_IJ")
            else:
                _swap(i, j)

        elif opcode == "XCHG2":
            i = _int_arg(0)
            j = _int_arg(1)
            if i is None or j is None:
                _warn_stack_args("missing stack indices for XCHG2")
            else:
                _swap(1, i)
                _swap(0, j)

        elif opcode in {"XCHG3", "XCHG3_ALT"}:
            i = _int_arg(0)
            j = _int_arg(1)
            k = _int_arg(2)
            if i is None or j is None or k is None:
                _warn_stack_args("missing stack indices for XCHG3")
            else:
                _swap(2, i)
                _swap(1, j)
                _swap(0, k)

        elif opcode == "PICK":
            # Dynamic copy: pop index, then copy s[i] to top
            # Optimization: try to resolve constant index to preserve taint info
            _ensure_depth(1, "pick")
            index_val = self.current_state.stack.pop(0)

            # Try to resolve the dynamic index from constant tracking
            resolved_index = self._try_resolve_constant_index(index_val)

            if resolved_index is not None and resolved_index < len(self.current_state.stack):
                # Successfully resolved: copy the value at resolved_index to top
                source_val = self.current_state.stack[resolved_index]
                if source_val:
                    # Copy the value preserving taint info
                    copied_val = source_val.copy()
                    copied_val.definition_site = idx  # Update definition site
                    self.current_state.stack.insert(0, copied_val)
                else:
                    self.current_state.stack.insert(0, _unknown_value("pick"))
                self.graph.analysis_metadata.setdefault("dynamic_stack_copies", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "resolved_index": resolved_index,
                    "reason": "dynamic stack index resolved from constant",
                })
            else:
                # Cannot resolve index: push unknown value
                self.current_state.stack.insert(0, _unknown_value("pick"))
                self.graph.analysis_metadata.setdefault("dynamic_stack_copies", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "reason": "dynamic stack index for PICK not resolved",
                })

        elif opcode == "XCHGX":
            # Dynamic index: pop i, then swap s0 with s[i]
            # Optimization: try to resolve constant index to preserve taint info
            _ensure_depth(1, "xchgx")
            index_val = self.current_state.stack.pop(0)

            # Try to resolve the dynamic index from constant tracking
            resolved_index = self._try_resolve_constant_index(index_val)

            if resolved_index is not None and resolved_index < len(self.current_state.stack):
                # Successfully resolved: perform precise swap, preserving taint info
                if resolved_index > 0:
                    _swap(0, resolved_index)
                # If resolved_index == 0, swap s0 with s0 is a no-op
                self.graph.analysis_metadata.setdefault("dynamic_stack_shuffles", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "resolved_index": resolved_index,
                    "reason": "dynamic stack index resolved from constant",
                })
            else:
                # Cannot resolve index: conservative invalidation
                self.graph.analysis_metadata.setdefault("dynamic_stack_shuffles", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "reason": "dynamic stack index for XCHGX not resolved",
                })
                _invalidate_stack("xchgx")

        elif opcode in self.DYNAMIC_SHUFFLE_OPCODES:
            # Dynamic shuffle: pop indices if any, then invalidate stack ordering.
            effect = get_stack_effect(opcode)
            pop_count = effect.min_inputs if effect and effect.min_inputs is not None else 0
            if pop_count:
                _ensure_depth(pop_count, "dynamic_shuffle")
                for _ in range(pop_count):
                    self.current_state.stack.pop(0)
            self.graph.analysis_metadata.setdefault("dynamic_stack_shuffles", []).append({
                "instruction": idx,
                "opcode": opcode,
                "reason": "dynamic stack shuffle not resolved",
            })
            _invalidate_stack("dynamic_shuffle")

        elif opcode in {"ADD", "SUB", "MUL", "DIV", "EQUAL", "LESS", "GREATER", "LEQ", "GEQ", "NEQ"}:
            # Binary operation: pop 2, push 1 result
            # TVM semantics: operation uses s0 and s1, result pushed to s0
            _ensure_depth(2, "binary_op")
            # Pop in order: first get s0, then get s1 (which becomes s0 after first pop)
            val_s0 = self.current_state.stack.pop(0) or _unknown_value("binary_op")
            val_s1 = self.current_state.stack.pop(0) or _unknown_value("binary_op")

            # Taint propagation: result is tainted if ANY input is tainted (OR logic)
            result_tainted = (val_s0 and val_s0.tainted) or (val_s1 and val_s1.tainted)

            # Validation tracking: Conservative approach
            # If result is tainted, it's only considered checked if:
            # 1. BOTH inputs are checked (if they exist), OR
            # 2. Only the tainted input is checked (if only one is tainted)
            # This ensures we don't lose track of unchecked tainted data
            result_checked = False
            if result_tainted:
                # Check which inputs contributed to the taint
                s0_tainted = val_s0 and val_s0.tainted
                s1_tainted = val_s1 and val_s1.tainted

                if s0_tainted and s1_tainted:
                    # Both tainted: both must be checked
                    result_checked = (val_s0.checked and val_s1.checked)
                elif s0_tainted:
                    # Only s0 tainted: it must be checked
                    result_checked = val_s0.checked
                elif s1_tainted:
                    # Only s1 tainted: it must be checked
                    result_checked = val_s1.checked

            metadata = {}
            if result_tainted:
                taint_sources, taint_origins = self._build_taint_metadata([val_s0, val_s1])
                if taint_sources:
                    metadata["taint_sources"] = taint_sources
                if taint_origins:
                    metadata["taint_origins"] = taint_origins

            # For comparison operations, save original operands' definition_sites
            # This allows guard operations to trace back and mark the original
            # tainted values as checked when the comparison result is guarded
            if opcode in {"EQUAL", "LESS", "GREATER", "LEQ", "GEQ", "NEQ"}:
                comparison_operands = []
                if val_s0:
                    comparison_operands.append(val_s0.definition_site)
                if val_s1:
                    comparison_operands.append(val_s1.definition_site)
                if comparison_operands:
                    metadata["comparison_operands"] = comparison_operands

            result = DataFlowValue(
                source=ValueSource.COMPUTATION,
                definition_site=idx,
                tainted=result_tainted,
                checked=result_checked,
                metadata=metadata,
            )
            self.current_state.stack.insert(0, result)
            self.graph.values.setdefault(idx, []).append(result)

        else:
            # Generic fallback: use cp0.json stack effects for comprehensive coverage
            # Use a conservative fallback for an instruction without a precise
            # stack model.
            self._handle_generic_stack_effect(
                opcode, idx, args,
                force_taint=force_taint,
                force_source=force_source,
                force_metadata=force_metadata,
                unknown_taint=unknown_taint,
                _unknown_value=_unknown_value,
                _ensure_depth=_ensure_depth,
                _merge_forced_metadata=_merge_forced_metadata,
            )

    def _handle_generic_stack_effect(
        self,
        opcode: str,
        idx: int,
        args,
        *,
        force_taint: Optional[bool],
        force_source: Optional[ValueSource],
        force_metadata: Optional[Dict[str, Any]],
        unknown_taint: bool,
        _unknown_value,
        _ensure_depth,
        _merge_forced_metadata,
    ):
        """Handle opcodes using generic stack effects from cp0.json."""
        effect = get_stack_effect(opcode)
        if effect:
            if effect.min_inputs > len(self.current_state.stack):
                _ensure_depth(effect.min_inputs, "stack_effects")
            pre_stack = list(self.current_state.stack)
            # Pop minimum guaranteed inputs from stack
            popped_values: List[Optional[DataFlowValue]] = []
            pop_count = effect.min_inputs
            for _ in range(pop_count):
                popped_values.append(self.current_state.stack.pop(0) or _unknown_value("stack_effects"))

            # Determine default taint and checked status for outputs
            # If ANY relevant input is tainted, outputs are tainted (conservative)
            if effect.is_dynamic:
                if effect.max_inputs is None:
                    taint_candidates = [v for v in pre_stack if v]
                else:
                    taint_candidates = [v for v in pre_stack[:effect.max_inputs] if v]
            else:
                taint_candidates = [v for v in popped_values if v]

            result_tainted = any(v.tainted for v in taint_candidates)

            # For checked status: use conservative approach
            result_checked = False
            if result_tainted:
                # All tainted inputs must be checked
                tainted_inputs = [v for v in taint_candidates if v.tainted]
                if tainted_inputs:
                    result_checked = all(v.checked for v in tainted_inputs)

            metadata: Dict[str, Any] = {}
            if result_tainted:
                taint_sources, taint_origins = self._build_taint_metadata(taint_candidates)
                if taint_sources:
                    metadata["taint_sources"] = taint_sources
                if taint_origins:
                    metadata["taint_origins"] = taint_origins

            taint_rules = self.MULTI_OUTPUT_TAINT_RULES.get(opcode)

            extra_outputs = 0
            if effect.is_dynamic and effect.max_outputs is None and effect.min_outputs == 0:
                extra_outputs = 1

            # Push minimum guaranteed outputs to stack (plus one unknown if outputs are unbounded)
            # Use output-specific taint rules if available, otherwise use conservative approach
            total_outputs = effect.min_outputs + extra_outputs
            for output_idx in range(total_outputs):
                if output_idx >= effect.min_outputs:
                    output_value = _unknown_value("dynamic_output")
                    if force_taint is True:
                        output_value.tainted = True
                        output_value.checked = False
                        if force_source is not None:
                            output_value.source = force_source
                        output_value.metadata = _merge_forced_metadata(output_value.metadata)
                    self.current_state.stack.insert(0, output_value)
                    if output_idx == 0:
                        self.graph.values.setdefault(idx, []).append(output_value)
                    continue
                # Determine taint for this specific output
                output_tainted = result_tainted
                output_checked = result_checked
                # Use deep copy to prevent shared mutable state between outputs
                # (metadata may contain sets like taint_sources, taint_origins)
                output_metadata = copy.deepcopy(metadata) if metadata else {}

                if taint_rules and output_idx < len(taint_rules):
                    rule = taint_rules[output_idx]

                    if rule == "inherit_first_input":
                        # Taint from first input (slice being read)
                        first_input = popped_values[0] if popped_values else None
                        output_tainted = first_input.tainted if first_input else False
                        output_checked = first_input.checked if first_input else False
                        if first_input and first_input.tainted:
                            output_metadata = {
                                "taint_sources": list(self._taint_sources_from_value(first_input)),
                                "taint_origins": list(self._taint_origins_from_value(first_input)),
                            }

                    elif rule == "preserve_slice":
                        # Remaining slice preserves original slice's taint/metadata
                        first_input = popped_values[0] if popped_values else None
                        if first_input:
                            output_tainted = first_input.tainted
                            output_checked = first_input.checked
                            # Preserve all metadata including from_message_slice flag
                            output_metadata = copy.deepcopy(first_input.metadata) if first_input.metadata else {}

                    elif rule == "no_taint":
                        output_tainted = False
                        output_checked = False
                        output_metadata = {}

                    # "inherit_all" uses the default (result_tainted/result_checked)

                if force_taint is not None:
                    output_tainted = force_taint
                    output_checked = False
                    output_metadata = _merge_forced_metadata(output_metadata)
                output_source = force_source if force_source is not None else ValueSource.COMPUTATION
                output_value = DataFlowValue(
                    source=output_source,
                    definition_site=idx,
                    tainted=output_tainted,
                    checked=output_checked,
                    metadata=output_metadata,
                )
                self.current_state.stack.insert(0, output_value)

                # Only record first output in graph to avoid clutter
                if output_idx == 0:
                    self.graph.values.setdefault(idx, []).append(output_value)

            if effect.is_dynamic and (
                effect.max_inputs != effect.min_inputs or effect.max_outputs != effect.min_outputs
            ):
                self.graph.analysis_metadata.setdefault("dynamic_stack_effects", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "min_inputs": effect.min_inputs,
                    "max_inputs": effect.max_inputs,
                    "min_outputs": effect.min_outputs,
                    "max_outputs": effect.max_outputs,
                })
            if extra_outputs:
                self.graph.analysis_metadata.setdefault("dynamic_stack_outputs", []).append({
                    "instruction": idx,
                    "opcode": opcode,
                    "min_outputs": effect.min_outputs,
                    "max_outputs": effect.max_outputs,
                })
        else:
            # Unknown instruction - record for debugging/analysis completeness
            self.graph.analysis_metadata.setdefault("unknown_instructions", []).append({
                "index": idx,
                "opcode": opcode
            })
            if force_taint is True or unknown_taint is False:
                output_tainted = force_taint is True
                output_metadata = _merge_forced_metadata({})
                output_value = DataFlowValue(
                    source=force_source if force_source is not None else ValueSource.UNKNOWN,
                    definition_site=idx,
                    tainted=output_tainted,
                    checked=False,
                    metadata=output_metadata,
                )
                self.current_state.stack.insert(0, output_value)
                self.graph.values.setdefault(idx, []).append(output_value)
