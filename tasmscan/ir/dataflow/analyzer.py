"""
Data Flow Analysis for TVM bytecode.

Tracks how values flow through the program:
- Stack values
- Register values
- Constants
- Message fields

This helps detect:
- Unvalidated data usage
- Improper value propagation
- Missing checks before critical operations

Note: This module shares several constants and classes with path_sensitive_dataflow.py:
- ValueSource, DataFlowValue, DataFlowState, DataFlowGraph are defined here and imported there
- TAINT_SOURCES, GUARD_OPCODES, SENSITIVE_OPCODES are defined in taint_registry.py and re-exported as class attributes on DataFlowAnalyzer
- PathSensitiveDataFlowAnalyzer reuses DataFlowAnalyzer internally for instruction analysis
This deliberate design avoids circular imports while maintaining code reuse.

Known limitations - cross-continuation taint propagation:
=================================================================
TVM supports continuations (PUSHCONT, CALLX, EXECUTE, JMPX, etc.) which are
first-class callable objects that capture code fragments. The analyzer now
builds a continuation-aware CFG and can propagate taint across resolved
continuation calls and returns (path-sensitive mode).

Remaining limitations:
1. Dynamic continuation targets (CALLREF/CALLDICT/JMPDICT, computed continuations)
   may not be resolvable. These are modeled conservatively with unknown targets.

2. Variable-arity call/jump instructions (e.g., *VARARGS) have runtime-dependent
   argument counts. The analysis approximates argument passing and may over-taint.

3. Control-register manipulation (PUSHCTR/POPCTR) is tracked only in simple cases;
   complex save/restore patterns can still lose precision.

Impact: False positives may occur due to conservative state merging.
        False negatives may occur when dynamic targets are unresolved.

Mitigation: For continuation-heavy code:
- Prefer path-sensitive analysis (uses continuation-aware CFG)
- Review dynamic continuation dispatch paths manually
"""
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from ...analyzer.facts import AnalysisFacts
from ...config import (
    INMSGPARAM_SENDER_INDEX,
    GUARD_OPCODES as CONFIG_GUARD_OPCODES,
    TAINT_STACK_DEPTH,
)
from ..tasir_types import SaveList, TVMModule

from .types import (
    ValueSource,
    DataFlowValue,
    DataFlowState,
    DataFlowGraph,
)
from .taint_registry import (
    CONTINUATION_OPCODES,
    MULTI_OUTPUT_TAINT_RULES,
    TAINT_SOURCES,
    CONDITIONAL_TAINT_SOURCES,
    MESSAGE_SLICE_LOADERS,
    CONTEXT_DEPENDENT_LOADERS,
    SENSITIVE_OPCODES,
    DYNAMIC_CALL_OPCODES,
    CALL_VARARGS_OPCODES,
    CONTINUATION_PUSH_OPCODES,
    DYNAMIC_SHUFFLE_OPCODES,
)
from .stack_simulator import StackSimulatorMixin
from .guard_analyzer import GuardAnalyzerMixin
from .taint_propagation import TaintPropagationMixin
from .program_view import ProgramBlock, ProgramInstruction, ProgramView, ensure_program_view


class DataFlowAnalyzer(StackSimulatorMixin, GuardAnalyzerMixin, TaintPropagationMixin):
    """
    Performs inter-procedural data flow analysis on TVM bytecode.

    Tracks:
    1. Taint analysis - which values come from untrusted sources (messages)
    2. Guard analysis - which values are checked before use
    3. Value propagation - how values flow through stack/registers
    """

    # Class-level constants for backward compatibility
    CONTINUATION_OPCODES = CONTINUATION_OPCODES
    MULTI_OUTPUT_TAINT_RULES = MULTI_OUTPUT_TAINT_RULES
    TAINT_SOURCES = TAINT_SOURCES
    CONDITIONAL_TAINT_SOURCES = CONDITIONAL_TAINT_SOURCES
    MESSAGE_SLICE_LOADERS = MESSAGE_SLICE_LOADERS
    CONTEXT_DEPENDENT_LOADERS = CONTEXT_DEPENDENT_LOADERS

    # Opcodes that validate values (make them "checked")
    GUARD_OPCODES = CONFIG_GUARD_OPCODES

    SENSITIVE_OPCODES = SENSITIVE_OPCODES
    DYNAMIC_CALL_OPCODES = DYNAMIC_CALL_OPCODES
    CALL_VARARGS_OPCODES = CALL_VARARGS_OPCODES
    CONTINUATION_PUSH_OPCODES = CONTINUATION_PUSH_OPCODES
    DYNAMIC_SHUFFLE_OPCODES = DYNAMIC_SHUFFLE_OPCODES

    def __init__(self):
        self.current_state = DataFlowState(stack=[], registers={}, guarded_values={})
        self.graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[])
        self._guard_access_counter = 0

    def analyze(
        self,
        program: AnalysisFacts | ProgramView | Any,
        path_sensitive: bool = False,
        path_config: Optional[Dict[str, int]] = None,
    ) -> DataFlowGraph:
        """
        Perform data flow analysis on the program.

        Args:
            program: AnalysisFacts, TASIR TVMModule, or ProgramView
            path_sensitive: If True, use path-sensitive analysis (path-sensitive analysis)
                          If False, use path-insensitive analysis
                          (CFG-aware fixpoint when blocks are available)
            path_config: Optional overrides for path-sensitive analysis parameters

        Returns a DataFlowGraph containing:
        - All value definitions
        - Dependencies between instructions
        - Taint propagation paths

        Note: Path-sensitive analysis requires CFG information and may be slower
        but produces more accurate results with fewer false positives.
        """
        view = ensure_program_view(program)

        if path_sensitive:
            from ..path_sensitive_dataflow import PathSensitiveDataFlowAnalyzer
            path_config = path_config or {}
            ps_analyzer = PathSensitiveDataFlowAnalyzer(
                max_paths=path_config.get("max_paths"),
                max_loop_unroll=path_config.get("max_loop_unroll"),
                max_analysis_depth=path_config.get("max_analysis_depth"),
                max_block_visits=path_config.get("max_block_visits"),
                max_worklist_size=path_config.get("max_worklist_size"),
            )
            graph = ps_analyzer.analyze(view)
            graph.analysis_type = "path_sensitive"
            if ps_analyzer.analysis_incomplete:
                graph.analysis_metadata["analysis_incomplete"] = True
                graph.analysis_metadata["truncated_paths"] = len(ps_analyzer.truncated_paths)
                graph.analysis_metadata["truncated_loop_paths"] = len(ps_analyzer.truncated_loop_paths)
            if any(inst.opcode in self.CONTINUATION_OPCODES for inst in view.instructions):
                graph.analysis_metadata["continuation_calls_detected"] = True
            if view.metadata.get("continuation_extraction_failed"):
                graph.analysis_metadata["continuation_extraction_failed"] = True
                graph.analysis_metadata["continuation_extraction_error"] = view.metadata.get(
                    "continuation_extraction_error"
                )
                graph.analysis_metadata["analysis_incomplete"] = True
            if view.metadata.get("analysis_incomplete"):
                graph.analysis_metadata["analysis_incomplete"] = True
                reasons = view.metadata.get("analysis_incomplete_reasons", [])
                if reasons:
                    existing = graph.analysis_metadata.get("analysis_incomplete_reasons", [])
                    graph.analysis_metadata["analysis_incomplete_reasons"] = list(
                        dict.fromkeys(list(existing) + list(reasons))
                    )
            self._mark_unknown_successors_incomplete(graph, view)
            return graph

        # Default: path-insensitive analysis
        self.graph = DataFlowGraph(
            values={}, edges=[], tainted_propagation=[],
            analysis_type="path_insensitive"
        )
        self.current_state = DataFlowState(stack=[], registers={}, guarded_values={})
        self._guard_access_counter = 0

        # Path-insensitive CFG-aware fixpoint when block structure exists.
        # Falls back to linear single pass for instruction-only inputs.
        if view.basic_blocks:
            self._analyze_cfg_path_insensitive(view)
        else:
            for inst in view.instructions:
                self._analyze_instruction(inst)

        if any(inst.opcode in self.CONTINUATION_OPCODES for inst in view.instructions):
            self.graph.analysis_metadata["continuation_calls_detected"] = True
        if view.metadata.get("continuation_extraction_failed"):
            self.graph.analysis_metadata["continuation_extraction_failed"] = True
            self.graph.analysis_metadata["continuation_extraction_error"] = view.metadata.get(
                "continuation_extraction_error"
            )
            self.graph.analysis_metadata["analysis_incomplete"] = True
        if view.metadata.get("analysis_incomplete"):
            self.graph.analysis_metadata["analysis_incomplete"] = True
            reasons = view.metadata.get("analysis_incomplete_reasons", [])
            if reasons:
                existing = self.graph.analysis_metadata.get("analysis_incomplete_reasons", [])
                self.graph.analysis_metadata["analysis_incomplete_reasons"] = list(
                    dict.fromkeys(list(existing) + list(reasons))
                )
        self._mark_unknown_successors_incomplete(self.graph, view)

        return self.graph

    @staticmethod
    def _mark_unknown_successors_incomplete(graph: DataFlowGraph, view: ProgramView) -> None:
        """Flag analysis as incomplete when CFG has unresolved successors."""
        unknown_blocks = sorted(
            block.id for block in view.basic_blocks if getattr(block, "has_unknown_successor", False)
        )
        if not unknown_blocks:
            return
        graph.analysis_metadata["analysis_incomplete"] = True
        graph.analysis_metadata["unknown_successor_blocks"] = unknown_blocks
        graph.analysis_metadata["unknown_successor_count"] = len(unknown_blocks)
        existing_reasons = graph.analysis_metadata.get("analysis_incomplete_reasons", [])
        graph.analysis_metadata["analysis_incomplete_reasons"] = list(
            dict.fromkeys(list(existing_reasons) + ["cfg_unknown_successor"])
        )

    def _analyze_cfg_path_insensitive(self, view: ProgramView) -> None:
        """Run path-insensitive dataflow using block-level CFG fixpoint."""
        block_map: Dict[int, ProgramBlock] = {block.id: block for block in view.basic_blocks}
        if not block_map:
            for inst in view.instructions:
                self._analyze_instruction(inst)
            return

        inst_map: Dict[int, ProgramInstruction] = {
            inst.index: inst for inst in view.instructions
        }
        predecessors: Dict[int, Set[int]] = {block_id: set() for block_id in block_map}
        for block in block_map.values():
            for succ in block.successors:
                if succ in predecessors:
                    predecessors[succ].add(block.id)

        entry_ids = [bid for bid in view.entry_block_ids() if bid in block_map]
        if not entry_ids:
            entry_ids = [min(block_map)]
        reachable = self._collect_reachable_block_ids(block_map, entry_ids)
        if not reachable:
            reachable = set(entry_ids)

        block_in: Dict[int, DataFlowState] = {}
        block_out: Dict[int, DataFlowState] = {}
        worklist: Deque[int] = deque(sorted(reachable))
        queued: Set[int] = set(worklist)
        max_iterations = max(64, len(reachable) * 32)
        iterations = 0
        stack_depth_cap = TAINT_STACK_DEPTH

        # Fixpoint phase computes stable block in/out states without polluting
        # output graph with duplicate values from repeated block visits.
        saved_graph = self.graph

        while worklist and iterations < max_iterations:
            iterations += 1
            block_id = worklist.popleft()
            queued.discard(block_id)
            block = block_map.get(block_id)
            if block is None:
                continue

            pred_states = [
                block_out[pred_id]
                for pred_id in sorted(predecessors.get(block_id, set()))
                if pred_id in block_out
            ]
            if pred_states:
                merged_in = self._merge_dataflow_states(pred_states)
            else:
                merged_in = DataFlowState(stack=[], registers={}, guarded_values={})

            if len(merged_in.stack) > stack_depth_cap:
                merged_in.stack = merged_in.stack[:stack_depth_cap]

            if block_id not in block_in or not self._states_converged(block_in.get(block_id), merged_in):
                block_in[block_id] = merged_in.copy()

            self.graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[])
            self.current_state = merged_in.copy()
            self._guard_access_counter = max(self.current_state.guarded_values.values(), default=0)
            for inst_idx in block.instruction_indices:
                inst = inst_map.get(inst_idx)
                if inst is None:
                    continue
                self._analyze_instruction(inst)
            new_out = self.current_state.copy()

            # guarded_values counter drift (only compares key sets)
            if block_id not in block_out or not self._states_converged(block_out[block_id], new_out):
                block_out[block_id] = new_out
                for succ_id in block.successors:
                    if succ_id in reachable and succ_id not in queued:
                        worklist.append(succ_id)
                        queued.add(succ_id)

        self.graph = saved_graph
        if worklist:
            self.graph.analysis_metadata["analysis_incomplete"] = True
            reasons = self.graph.analysis_metadata.get("analysis_incomplete_reasons", [])
            self.graph.analysis_metadata["analysis_incomplete_reasons"] = list(
                dict.fromkeys(list(reasons) + ["path_insensitive_fixpoint_cap"])
            )

        # Replay phase emits graph values once from converged block-entry states.
        for block_id in self._ordered_reachable_blocks(block_map, entry_ids):
            if block_id not in reachable:
                continue
            block = block_map[block_id]
            entry_state = block_in.get(
                block_id,
                DataFlowState(stack=[], registers={}, guarded_values={}),
            )
            self.current_state = entry_state.copy()
            self._guard_access_counter = max(self.current_state.guarded_values.values(), default=0)
            for inst_idx in block.instruction_indices:
                inst = inst_map.get(inst_idx)
                if inst is None:
                    continue
                self._analyze_instruction(inst)

        terminal_states = []
        for block_id in sorted(reachable):
            block = block_map.get(block_id)
            if block is None:
                continue
            reachable_succs = [succ for succ in block.successors if succ in reachable]
            if not reachable_succs and block_id in block_out:
                terminal_states.append(block_out[block_id])
        if not terminal_states:
            terminal_states = [block_out[bid] for bid in sorted(block_out) if bid in reachable]
        self.current_state = (
            self._merge_dataflow_states(terminal_states)
            if terminal_states
            else DataFlowState(stack=[], registers={}, guarded_values={})
        )
        self._guard_access_counter = max(self.current_state.guarded_values.values(), default=0)
        self.graph.analysis_metadata["path_insensitive_cfg_fixpoint"] = True
        self.graph.analysis_metadata["path_insensitive_cfg_iterations"] = iterations
        self.graph.analysis_metadata["path_insensitive_cfg_blocks"] = len(reachable)
        self.graph.analysis_metadata["path_insensitive_cfg_entry_blocks"] = sorted(entry_ids)

    @staticmethod
    def _collect_reachable_block_ids(
        block_map: Dict[int, ProgramBlock],
        entry_ids: List[int],
    ) -> Set[int]:
        """Collect blocks reachable from entries."""
        reachable: Set[int] = set()
        queue: Deque[int] = deque(entry_ids)
        while queue:
            block_id = queue.popleft()
            if block_id in reachable or block_id not in block_map:
                continue
            reachable.add(block_id)
            for succ_id in block_map[block_id].successors:
                if succ_id not in reachable:
                    queue.append(succ_id)
        return reachable

    @staticmethod
    def _ordered_reachable_blocks(
        block_map: Dict[int, ProgramBlock],
        entry_ids: List[int],
    ) -> List[int]:
        """Return deterministic BFS order over reachable blocks."""
        visited: Set[int] = set()
        order: List[int] = []
        queue: Deque[int] = deque(sorted(entry_ids))
        while queue:
            block_id = queue.popleft()
            if block_id in visited or block_id not in block_map:
                continue
            visited.add(block_id)
            order.append(block_id)
            for succ_id in sorted(block_map[block_id].successors):
                if succ_id not in visited:
                    queue.append(succ_id)
        for block_id in sorted(block_map):
            if block_id not in visited:
                order.append(block_id)
        return order

    @staticmethod
    def _states_converged(old: Optional[DataFlowState], new: DataFlowState) -> bool:
        """Semantic convergence check for the path-insensitive fixpoint.

        Compares two DataFlowStates while ignoring the LRU counter values
        in ``guarded_values`` (only the key set matters for convergence).
        This prevents the monotonically-increasing guard access counter
        from causing spurious state inequality and fixpoint divergence.
        """
        if old is None:
            return False

        if set(old.guarded_values.keys()) != set(new.guarded_values.keys()):
            return False

        # Compare stack length and values
        if len(old.stack) != len(new.stack):
            return False
        for ov, nv in zip(old.stack, new.stack):
            if ov is None and nv is None:
                continue
            if ov is None or nv is None:
                return False
            if (ov.tainted, ov.checked, ov.definition_site) != (nv.tainted, nv.checked, nv.definition_site):
                return False

        # Compare registers
        if old.registers.keys() != new.registers.keys():
            return False
        for k in old.registers:
            ov, nv = old.registers[k], new.registers.get(k)
            if ov is None and nv is None:
                continue
            if ov is None or nv is None:
                return False
            if (ov.tainted, ov.checked, ov.definition_site) != (nv.tainted, nv.checked, nv.definition_site):
                return False

        return True

    def _merge_dataflow_states(self, states: List[DataFlowState]) -> DataFlowState:
        """Conservatively merge multiple dataflow states."""
        if not states:
            return DataFlowState(stack=[], registers={}, guarded_values={})
        if len(states) == 1:
            return states[0].copy()

        max_stack_size = max(len(state.stack) for state in states)
        merged_stack: List[Optional[DataFlowValue]] = []
        for depth in range(max_stack_size):
            values = [
                state.stack[depth] if depth < len(state.stack) else None
                for state in states
            ]
            merged_stack.append(self._merge_dataflow_values(values))

        merged_registers: Dict[int, Optional[DataFlowValue]] = {}
        reg_indices: Set[int] = set()
        for state in states:
            reg_indices.update(state.registers.keys())
        for reg_idx in sorted(reg_indices):
            values = [state.registers.get(reg_idx) for state in states]
            merged_value = self._merge_dataflow_values(values)
            if merged_value is not None:
                merged_registers[reg_idx] = merged_value

        merged_guarded: Dict[int, int] = {}
        for state in states:
            for def_site, access_count in state.guarded_values.items():
                if (
                    def_site not in merged_guarded
                    or access_count > merged_guarded[def_site]
                ):
                    merged_guarded[def_site] = access_count

        return DataFlowState(
            stack=merged_stack,
            registers=merged_registers,
            guarded_values=merged_guarded,
        )

    def _merge_dataflow_values(
        self,
        values: List[Optional[DataFlowValue]],
    ) -> Optional[DataFlowValue]:
        """Merge value candidates from multiple predecessors."""
        candidates = [value for value in values if value is not None]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0].copy()

        tainted = any(value.tainted for value in candidates)
        checked = False
        metadata: Dict[str, Any] = {}
        if tainted:
            tainted_values = [value for value in candidates if value.tainted]
            if tainted_values:
                checked = all(value.checked for value in tainted_values)
            sources, origins = self._merge_taint_metadata(candidates)
            if sources:
                # convergence comparison (list(set(...)) is non-deterministic)
                metadata["taint_sources"] = sorted(sources)
            if origins:
                metadata["taint_origins"] = sorted(origins)

        comparison_ops = self._merge_comparison_operands(candidates)
        if comparison_ops:
            metadata["comparison_operands"] = comparison_ops

        template = candidates[0]
        return DataFlowValue(
            source=template.source,
            definition_site=template.definition_site,
            tainted=tainted,
            checked=checked,
            metadata=metadata,
        )

    @staticmethod
    def _merge_comparison_operands(values: List[DataFlowValue]) -> Optional[List[Any]]:
        """Merge comparison_operands metadata from multiple values."""
        merged: List[Any] = []
        for value in values:
            ops = value.metadata.get("comparison_operands", [])
            for op in ops:
                if op not in merged:
                    merged.append(op)
        return merged if merged else None

    def _analyze_instruction(self, inst: ProgramInstruction):
        """Analyze a single instruction and update state"""
        opcode = inst.opcode
        idx = inst.index

        # Check if this instruction introduces tainted data via INMSGPARAM
        if opcode == "INMSGPARAM":
            arg_value = self._get_first_int_arg(inst)
            if arg_value == INMSGPARAM_SENDER_INDEX:
                source = ValueSource.MESSAGE_SENDER
                value = DataFlowValue(
                    source=source,
                    definition_site=idx,
                    tainted=True,
                    metadata={
                        "taint_sources": [source.value],
                        "taint_origins": [idx],
                    },
                )
                self.current_state.stack.insert(0, value)
                self.graph.values.setdefault(idx, []).append(value)
                return

        # Handle taint sources (always taint)
        if self._handle_taint_source(opcode, idx, inst.arguments):
            return

        # Handle context-dependent taint sources (LDU/LDI)
        if self._handle_conditional_taint_source(opcode, idx, inst.arguments):
            return

        # Handle guard operations
        if opcode in self.GUARD_OPCODES:
            self._process_guard_instruction(opcode, self.GUARD_OPCODES)

        # Handle sensitive operations
        self._handle_sensitive_operation(opcode, idx)

        # Handle dynamic calls
        self._handle_dynamic_call(opcode, idx)

        # Handle message slice loaders
        if self._handle_message_slice_loader(opcode, idx, inst.arguments):
            return

        # Handle context-dependent loaders (CTOS)
        if self._handle_context_dependent_loader(opcode, idx):
            return

        # Stack simulation (simplified)
        self._update_stack_for_opcode(opcode, idx, inst.arguments)

    def get_tainted_paths(self) -> List[Tuple[int, int]]:
        """Get all paths where tainted data reaches sensitive operations"""
        return self.graph.tainted_propagation

    def get_unguarded_tainted_values(self) -> List[DataFlowValue]:
        """Get all tainted values that are never checked"""
        unguarded = []
        for values in self.graph.values.values():
            for value in values:
                if value.tainted and not value.checked:
                    if value.definition_site not in self.current_state.guarded_values:
                        unguarded.append(value)
        return unguarded

    def format_report(self) -> str:
        """Format analysis results as readable report"""
        lines = ["Data Flow Analysis Report", "=" * 50, ""]

        # Tainted paths
        if self.graph.tainted_propagation:
            lines.append(f"[WARNING] Found {len(self.graph.tainted_propagation)} tainted data flows:")
            for from_idx, to_idx in self.graph.tainted_propagation:
                lines.append(
                    f"   Tainted value from instruction #{from_idx} "
                    f"reaches sensitive operation at #{to_idx}"
                )
            lines.append("")

        # Unguarded tainted values
        unguarded = self.get_unguarded_tainted_values()
        if unguarded:
            lines.append(f"[WARNING] Found {len(unguarded)} unguarded tainted values:")
            for value in unguarded[:10]:  # Show first 10
                lines.append(f"   {value}")
            if len(unguarded) > 10:
                lines.append(f"   ... and {len(unguarded) - 10} more")
            lines.append("")

        # Summary
        total_values = sum(len(vals) for vals in self.graph.values.values())
        tainted_values = sum(
            1 for vals in self.graph.values.values() for v in vals if v.tainted
        )

        lines.append("Summary:")
        lines.append(f"  Total values tracked: {total_values}")
        lines.append(f"  Tainted values: {tainted_values}")
        lines.append(f"  Guarded values: {len(self.current_state.guarded_values)}")

        return "\n".join(lines)

    # =========================================================================
    # TASIR Support Methods (TASIR support)
    # =========================================================================

    def analyze_tasir(
        self,
        module: TVMModule,
        path_sensitive: bool = False,
        path_config: Optional[Dict[str, int]] = None,
    ) -> DataFlowGraph:
        """
        Perform data flow analysis on TASIR representation.

        This method leverages the TASIR's ContinuationDescriptor and SaveList
        to achieve more precise cross-continuation taint tracking, addressing
        the cross-continuation limitation.

        Args:
            module: TASIR module from IRBuilder.build_tasir()
            path_sensitive: Whether to use path-sensitive analysis
            path_config: Optional config overrides

        Returns:
            DataFlowGraph with enhanced continuation tracking
        """
        if module is None:
            # Return empty graph if module is None
            return DataFlowGraph(
                values={},
                edges=[],
                tainted_propagation=[],
                analysis_type="path_insensitive",
                analysis_metadata={"error": "TVMModule is None"},
            )

        # Direct module adapter path (no TASIR->Facts roundtrip).
        program_view = ensure_program_view(module)

        # Run standard analysis
        graph = self.analyze(program_view, path_sensitive, path_config)

        # Enhance with SaveList information
        self._enhance_with_savelists(graph, module)

        # Mark analysis as TASIR-enhanced
        graph.analysis_metadata["tasir_enhanced"] = True
        graph.analysis_metadata["savelist_edges_added"] = graph.analysis_metadata.get(
            "savelist_edges_added", 0
        )
        dynamic_target_count = int(module.analysis_metadata.get("dynamic_target_count", 0) or 0)
        if dynamic_target_count > 0:
            graph.analysis_metadata["dynamic_target_count"] = dynamic_target_count
            graph.analysis_metadata["dynamic_target_instructions"] = list(
                module.analysis_metadata.get("dynamic_target_instructions", [])
            )
            graph.analysis_metadata["analysis_incomplete"] = True
        if "dynamic_target_solver_stats" in module.analysis_metadata:
            graph.analysis_metadata["dynamic_target_solver_stats"] = dict(
                module.analysis_metadata.get("dynamic_target_solver_stats", {})
            )
        if "dynamic_target_solver_results" in module.analysis_metadata:
            graph.analysis_metadata["dynamic_target_solver_results"] = list(
                module.analysis_metadata.get("dynamic_target_solver_results", [])
            )
        if "dynamic_target_solver_obligations" in module.analysis_metadata:
            graph.analysis_metadata["dynamic_target_solver_obligations"] = list(
                module.analysis_metadata.get("dynamic_target_solver_obligations", [])
            )
        if module.analysis_metadata.get("analysis_incomplete"):
            graph.analysis_metadata["analysis_incomplete"] = True
            reasons = module.analysis_metadata.get("analysis_incomplete_reasons", [])
            if reasons:
                existing = graph.analysis_metadata.get("analysis_incomplete_reasons", [])
                graph.analysis_metadata["analysis_incomplete_reasons"] = list(
                    dict.fromkeys(list(existing) + list(reasons))
                )

        return graph

    def _enhance_with_savelists(
        self,
        graph: DataFlowGraph,
        module: TVMModule,
    ) -> None:
        """
        Enhance dataflow graph with SaveList information.

        This is the core cross-continuation propagation: when we encounter CONT_CALL or CONT_JUMP
        instructions, we check if the target continuation has a SaveList with
        saved register values and propagate their taint state accordingly.

        Args:
            graph: DataFlowGraph to enhance
            module: TVMModule with continuation descriptors
        """
        from ..tasir_types import InstructionKind

        savelist_edges_added = 0

        # Find all continuation call/jump sites
        for inst in module.all_instructions():
            if inst.kind in {InstructionKind.CONT_CALL, InstructionKind.CONT_JUMP}:
                # Check each continuation reference
                for cont_ref in inst.continuation_refs:
                    # Look up descriptor in cross-function continuations
                    descriptor = module.cross_function_continuations.get(cont_ref)

                    # Also check function-local continuations
                    if descriptor is None:
                        for func in module.functions.values():
                            if cont_ref in func.continuations:
                                descriptor = func.continuations[cont_ref]
                                break

                    if descriptor and descriptor.savelist.has_saved_values():
                        # Propagate taint from saved registers
                        edges_added = self._propagate_savelist_taint(
                            graph, inst.index, descriptor.savelist
                        )
                        savelist_edges_added += edges_added

        graph.analysis_metadata["savelist_edges_added"] = savelist_edges_added

    def _propagate_savelist_taint(
        self,
        graph: DataFlowGraph,
        call_site: int,
        savelist: SaveList,
    ) -> int:
        """
        Propagate taint through SaveList to call site.

        When a continuation is called and it has saved register values,
        we need to track whether those saved values were tainted at their
        definition site and propagate that taint to the call site.

        Note: SaveList TVMAbstractValues are created during the linking phase
        (before taint analysis runs), so their `tainted` field is typically
        stale (always False). To bridge this phase gap, we also look up the
        taint status from the DataFlowGraph, which contains results from the
        completed analysis pass.

        Args:
            graph: DataFlowGraph to update
            call_site: Instruction index of the call/jump
            savelist: SaveList with saved register values

        Returns:
            Number of taint propagation edges added
        """
        edges_added = 0

        for reg_idx, saved_values in savelist.iter_saved_values():
            for saved_value in saved_values:
                # Look up taint status from completed analysis
                definition_site = saved_value.definition_site
                is_tainted_in_graph = (
                    definition_site is not None
                    and definition_site in graph.values
                    and any(v.tainted for v in graph.values[definition_site])
                )
                if is_tainted_in_graph or saved_value.tainted:
                    # Add taint propagation edge from definition site to call site
                    if saved_value.definition_site is not None:
                        edge = (saved_value.definition_site, call_site)
                        if edge not in graph._taint_dedup:
                            graph._taint_dedup.add(edge)
                            graph.tainted_propagation.append(edge)
                            edges_added += 1

                            # Also add to graph values if not already present
                            if call_site not in graph.values:
                                graph.values[call_site] = []

                            # Record the savelist taint source
                            savelist_value = DataFlowValue(
                                source=ValueSource.UNKNOWN,
                                definition_site=call_site,
                                tainted=True,
                                checked=saved_value.checked,
                                metadata={
                                    "savelist_source": True,
                                    "saved_register": reg_idx,
                                    "original_definition": saved_value.definition_site,
                                    "taint_origins": [saved_value.definition_site],
                                    "savelist_disjunctive": len(saved_values) > 1,
                                },
                            )
                            graph.values[call_site].append(savelist_value)

        return edges_added
