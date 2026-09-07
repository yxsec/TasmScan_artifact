"""
Control Flow Graph (CFG) builder for TVM program analysis.

This module handles:
- Building CFG edges from instruction sequences
- Handling continuation call/return edges
- Building basic blocks from CFG
"""
import logging
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .constants import (
    CALL_ARGS_SUFFIX_PREFIXES,
    CALL_CONT_OPCODES,
    CALL_OPCODES,
    CONDITIONAL_BRANCHES,
    CONDITIONAL_GUARDS,
    CONDITIONAL_RETURNS,
    DICT_DISPATCH_EXEC_OPCODES,
    DICT_DISPATCH_OPCODES,
    INFINITE_LOOP_OPCODES,
    LOOP_CALL_OPCODES,
    MAIN_CONTEXT,
    TERMINATORS,
    UNCONDITIONAL_BRANCHES,
)
from .continuation_resolver import UncertainInfo, is_unknown_cont
from .facts import BasicBlock, ControlFlowEdge, InstructionFact

logger = logging.getLogger(__name__)


class CFGBuilder:
    """Builds Control Flow Graphs from TVM instruction sequences."""

    _RETURN_OPCODES = {
        "RET", "RETALT", "RETBOOL", "RETDATA", "RETVARARGS",
    }

    def __init__(self, continuation_resolver):
        """Initialize with a ContinuationResolver for mapping continuations.

        Args:
            continuation_resolver: ContinuationResolver instance
        """
        self._continuation_resolver = continuation_resolver
        self.last_build_metadata: Dict[str, Any] = {}

    def build_cfg_with_continuations(
        self,
        all_facts: Sequence[InstructionFact],
        pushcont_to_cont_ids: Dict[int, List[str]],
        inline_cont_map: Dict[int, List[str]],
        continuations: Dict[str, Any],
        cont_fact_map: Dict[Tuple[str, int], int],
        *,
        enable_known_prefix_fallback: bool = True,
        enable_entry_shape_propagation: bool = True,
        enable_ctrl_reg_propagation: bool = True,
        enable_varargs_depth_fix: bool = True,
        enable_reachability_pruning: bool = True,
    ) -> List[ControlFlowEdge]:
        """
        Build CFG with continuation support.

        Creates:
        - Normal edges (fallthrough, branch, jump) for each context
        - call_cont edges: from IF/JMP to continuation entry
        - return_cont edges: from continuation end back to caller
        """
        self.last_build_metadata = {}
        edges: List[ControlFlowEdge] = []

        main_facts = [f for f in all_facts if f.continuation_id is None]
        cont_facts_by_id: Dict[str, List[InstructionFact]] = {}
        for cont_id in continuations:
            cont_facts_by_id[cont_id] = [
                f for f in all_facts if f.continuation_id == cont_id
            ]

        context_facts: Dict[str, Sequence[InstructionFact]] = {MAIN_CONTEXT: main_facts}
        context_facts.update(cont_facts_by_id)
        empty_cont_ids: Set[str] = {
            cont_id for cont_id, cont in continuations.items() if not getattr(cont, "instructions", None)
        }

        # Interprocedural entry-shape propagation:
        # analyze contexts in a worklist and propagate caller-derived stack shape
        # into callee continuations until shapes converge.
        context_entry_shapes: Dict[str, Tuple[List[Tuple[str, Optional[str]]], bool]] = {
            cont_id: ([], True) for cont_id in cont_facts_by_id
        }
        # Track c1 (alt return), c2 (exception handler), and c3 (method dictionary)
        # across context calls to improve RETALT/RETBOOL/TRY precision.
        propagated_ctrl_reg_indices: Set[int] = {1, 2, 3} if enable_ctrl_reg_propagation else set()
        context_ctrl_regs: Dict[str, Dict[int, Optional[str]]] = {}
        context_results: Dict[
            str,
            Tuple[Dict[int, List[str]], Set[int], Dict[str, List[int]], Dict[int, List[UncertainInfo]]]
        ] = {}
        caller_outputs: Dict[str, Dict[str, Tuple[List[Tuple[str, Optional[str]]], bool]]] = {}
        callee_inputs: Dict[str, Dict[str, Tuple[List[Tuple[str, Optional[str]]], bool]]] = {}
        caller_ctrl_outputs: Dict[str, Dict[str, Dict[int, Optional[str]]]] = {}
        callee_ctrl_inputs: Dict[str, Dict[str, Dict[int, Optional[str]]]] = {}

        queue = deque(context_facts.keys())
        queued: Set[str] = set(context_facts.keys())
        active_entry_shape_propagation = enable_entry_shape_propagation
        active_ctrl_reg_propagation = enable_ctrl_reg_propagation
        # Without entry-shape propagation, each context is analyzed exactly once.
        max_iterations = max(1, len(context_facts) * 8) if enable_entry_shape_propagation else len(context_facts)
        # After widening, run one conservative sweep without further inter-context
        # propagation to avoid non-terminating refinement cycles.
        post_widening_max_iterations = (
            len(context_facts) if enable_entry_shape_propagation else len(context_facts)
        )
        phase_max_iterations = max_iterations
        phase_iterations = 0
        total_iterations = 0
        widened = False
        widened_contexts: List[str] = []
        self.last_build_metadata = {
            "entry_shape_max_iterations": max_iterations,
            "entry_shape_post_widening_max_iterations": post_widening_max_iterations,
            "entry_shape_iterations": 0,
            "entry_shape_converged": True,
            "entry_shape_pending_contexts": [],
            "entry_shape_widened": False,
            "entry_shape_widened_contexts": [],
        }

        while queue:
            if phase_iterations >= phase_max_iterations:
                if enable_entry_shape_propagation and not widened:
                    widened = True
                    widened_contexts = sorted(cont_facts_by_id.keys())
                    # Conservative widening: restart pending continuation contexts from
                    # top-unknown stack/control-register seeds, then rerun propagation.
                    for ctx_id in widened_contexts:
                        context_entry_shapes[ctx_id] = ([], True)
                        if enable_ctrl_reg_propagation:
                            context_ctrl_regs[ctx_id] = {}
                    caller_outputs.clear()
                    callee_inputs.clear()
                    caller_ctrl_outputs.clear()
                    callee_ctrl_inputs.clear()
                    queue = deque(context_facts.keys())
                    queued = set(context_facts.keys())
                    phase_iterations = 0
                    phase_max_iterations = post_widening_max_iterations
                    active_entry_shape_propagation = False
                    active_ctrl_reg_propagation = False
                    self.last_build_metadata["entry_shape_widened"] = True
                    self.last_build_metadata["entry_shape_widened_contexts"] = widened_contexts
                    continue
                break

            ctx_id = queue.popleft()
            queued.discard(ctx_id)
            phase_iterations += 1
            total_iterations += 1
            facts_in_ctx = context_facts.get(ctx_id, [])

            emitted_entry_shapes: Dict[str, List[Tuple[List[Tuple[str, Optional[str]]], bool]]] = {}
            emitted_entry_ctrl_regs: Dict[str, List[Dict[int, Optional[str]]]] = {}
            branch_map, returning_branches, nested_targets, uncertain_branches, _ = \
                self._continuation_resolver.map_continuations(
                    facts_in_ctx,
                    pushcont_to_cont_ids,
                    inline_cont_map,
                    entry_shape=context_entry_shapes.get(ctx_id) if active_entry_shape_propagation else None,
                    cont_entry_shapes=emitted_entry_shapes if active_entry_shape_propagation else None,
                    cont_entry_ctrl_regs=emitted_entry_ctrl_regs if active_ctrl_reg_propagation else None,
                    initial_ctrl_regs=context_ctrl_regs.get(ctx_id),
                    enable_known_prefix_fallback=enable_known_prefix_fallback,
                    enable_varargs_depth_fix=enable_varargs_depth_fix,
                )
            context_results[ctx_id] = (
                branch_map,
                returning_branches,
                nested_targets,
                uncertain_branches,
            )

            new_outputs = {
                callee_id: self._merge_entry_shapes(shape_list)
                for callee_id, shape_list in emitted_entry_shapes.items()
                if shape_list
            }
            new_ctrl_outputs: Dict[str, Dict[int, Optional[str]]] = {}
            for callee_id, reg_sets in emitted_entry_ctrl_regs.items():
                if callee_id not in cont_facts_by_id or not reg_sets:
                    continue
                filtered_reg_sets = [
                    self._filter_ctrl_regs(regs, propagated_ctrl_reg_indices)
                    for regs in reg_sets
                ]
                # Keep empty snapshots in the merge so "missing c3 at some callsites"
                # is preserved as unknown instead of being incorrectly dropped.
                merged_regs = self._merge_ctrl_regs(filtered_reg_sets) if filtered_reg_sets else {}
                new_ctrl_outputs[callee_id] = merged_regs
            previous_outputs = caller_outputs.get(ctx_id, {})
            caller_outputs[ctx_id] = new_outputs
            previous_ctrl_outputs = caller_ctrl_outputs.get(ctx_id, {})
            caller_ctrl_outputs[ctx_id] = new_ctrl_outputs

            if active_entry_shape_propagation:
                affected_callees = set(previous_outputs) | set(new_outputs)
                for callee_id in affected_callees:
                    if callee_id not in cont_facts_by_id:
                        continue

                    inputs_by_caller = callee_inputs.setdefault(callee_id, {})
                    new_shape = new_outputs.get(callee_id)
                    if new_shape is None:
                        inputs_by_caller.pop(ctx_id, None)
                    else:
                        inputs_by_caller[ctx_id] = new_shape

                    merged_shape = (
                        self._merge_entry_shapes(list(inputs_by_caller.values()))
                        if inputs_by_caller
                        else ([], True)
                    )
                    if context_entry_shapes.get(callee_id) != merged_shape:
                        context_entry_shapes[callee_id] = merged_shape
                        if callee_id not in queued:
                            queue.append(callee_id)
                            queued.add(callee_id)

            if active_ctrl_reg_propagation:
                affected_ctrl_callees = set(previous_ctrl_outputs) | set(new_ctrl_outputs)
                for callee_id in affected_ctrl_callees:
                    if callee_id not in cont_facts_by_id:
                        continue

                    ctrl_inputs_by_caller = callee_ctrl_inputs.setdefault(callee_id, {})
                    new_regs = new_ctrl_outputs.get(callee_id)
                    if new_regs is None:
                        ctrl_inputs_by_caller.pop(ctx_id, None)
                    else:
                        ctrl_inputs_by_caller[ctx_id] = dict(new_regs)

                    merged_regs = (
                        self._merge_ctrl_regs(list(ctrl_inputs_by_caller.values()))
                        if ctrl_inputs_by_caller
                        else {}
                    )
                    if context_ctrl_regs.get(callee_id) != merged_regs:
                        context_ctrl_regs[callee_id] = merged_regs
                        if callee_id not in queued:
                            queue.append(callee_id)
                            queued.add(callee_id)

        if queue:
            if widened:
                logger.warning(
                    "Interprocedural entry-shape propagation did not converge after %d + %d iterations",
                    max_iterations,
                    post_widening_max_iterations,
                )
            else:
                logger.warning(
                    "Interprocedural entry-shape propagation did not converge after %d iterations",
                    max_iterations,
                )
            self.last_build_metadata["entry_shape_converged"] = False
            self.last_build_metadata["entry_shape_pending_contexts"] = sorted(queue)
        self.last_build_metadata["entry_shape_iterations"] = total_iterations

        cont_return_targets: Dict[str, List[int]] = {}
        for _ctx_id, (_branch_map, _returning, nested_targets, _uncertain) in context_results.items():
            for cont_id, targets in nested_targets.items():
                cont_return_targets.setdefault(cont_id, []).extend(targets)
        for cont_id, targets in list(cont_return_targets.items()):
            cont_return_targets[cont_id] = sorted(set(targets))
        alt_return_targets_by_context: Dict[str, List[int]] = {}
        alt_return_unknown_by_context: Set[str] = set()
        throw_targets_by_context: Dict[str, List[int]] = {}
        throw_target_unknown_by_context: Set[str] = set()

        for cont_id in cont_facts_by_id:
            ctrl_inputs = list(callee_ctrl_inputs.get(cont_id, {}).values())
            if not ctrl_inputs and cont_id in context_ctrl_regs:
                ctrl_inputs = [context_ctrl_regs[cont_id]]
            if not ctrl_inputs:
                continue

            alt_targets: Set[int] = set()
            has_unknown_alt = False
            throw_targets: Set[int] = set()
            has_unknown_throw = False
            for regs in ctrl_inputs:
                if 1 not in regs:
                    has_unknown_alt = True
                else:
                    c1_value = regs.get(1)
                    if not isinstance(c1_value, str) or not c1_value or is_unknown_cont(c1_value):
                        has_unknown_alt = True
                    else:
                        target_idx = cont_fact_map.get((c1_value, 0))
                        if target_idx is None:
                            has_unknown_alt = True
                        else:
                            alt_targets.add(target_idx)

                if 2 not in regs:
                    has_unknown_throw = True
                else:
                    c2_value = regs.get(2)
                    if not isinstance(c2_value, str) or not c2_value or is_unknown_cont(c2_value):
                        has_unknown_throw = True
                    else:
                        throw_target_idx = cont_fact_map.get((c2_value, 0))
                        if throw_target_idx is None:
                            has_unknown_throw = True
                        else:
                            throw_targets.add(throw_target_idx)

            if alt_targets:
                alt_return_targets_by_context[cont_id] = sorted(alt_targets)
            if has_unknown_alt:
                alt_return_unknown_by_context.add(cont_id)
            if throw_targets:
                throw_targets_by_context[cont_id] = sorted(throw_targets)
            if has_unknown_throw:
                throw_target_unknown_by_context.add(cont_id)

        infinite_loop_continue_targets: Dict[str, List[int]] = {}
        infinite_loop_exit_targets: Dict[str, List[int]] = {}
        for ctx_id, facts_in_ctx in context_facts.items():
            branch_map_ctx, _ret_ctx, _nested_ctx, _uncertain_ctx = context_results.get(
                ctx_id, ({}, set(), {}, {})
            )
            for pos, fact in enumerate(facts_in_ctx):
                if fact.opcode.upper() not in INFINITE_LOOP_OPCODES:
                    continue
                cont_ids = branch_map_ctx.get(fact.index, [])
                if not cont_ids:
                    continue
                exit_target = facts_in_ctx[pos + 1].index if pos + 1 < len(facts_in_ctx) else None
                for cont_id in cont_ids:
                    infinite_loop_continue_targets.setdefault(cont_id, []).append(fact.index)
                    if exit_target is not None:
                        infinite_loop_exit_targets.setdefault(cont_id, []).append(exit_target)
        for cont_id, targets in list(infinite_loop_continue_targets.items()):
            infinite_loop_continue_targets[cont_id] = sorted(set(targets))
        for cont_id, targets in list(infinite_loop_exit_targets.items()):
            infinite_loop_exit_targets[cont_id] = sorted(set(targets))
        method_dispatch_targets = sorted(
            cont_fact_map[(cont_id, 0)]
            for cont_id, cont in continuations.items()
            if getattr(cont, "kind", None) == "method" and (cont_id, 0) in cont_fact_map
        )

        main_branch_map, main_returning, _main_nested, main_uncertain = context_results.get(
            MAIN_CONTEXT, ({}, set(), {}, {})
        )
        edges.extend(
            self._build_cfg_for_context(
                main_facts,
                context_id=MAIN_CONTEXT,
                cont_fact_map=cont_fact_map,
                branch_cont_map=main_branch_map,
                returning_branches=main_returning,
                return_targets=None,
                uncertain_branches=main_uncertain,
                empty_cont_ids=empty_cont_ids,
                method_dispatch_targets=method_dispatch_targets,
                infinite_loop_continue_targets=None,
                infinite_loop_exit_targets=None,
                guard_throw_targets=None,
                guard_throw_target_unknown=False,
            )
        )

        for cont_id, cont_facts in cont_facts_by_id.items():
            cont_branch_map, cont_returning, _nested, cont_uncertain = context_results.get(
                cont_id, ({}, set(), {}, {})
            )
            edges.extend(
                self._build_cfg_for_context(
                    cont_facts,
                    context_id=cont_id,
                    cont_fact_map=cont_fact_map,
                    branch_cont_map=cont_branch_map,
                    returning_branches=cont_returning,
                    return_targets=cont_return_targets.get(cont_id, []),
                    alt_return_targets=alt_return_targets_by_context.get(cont_id, []),
                    alt_return_target_unknown=cont_id in alt_return_unknown_by_context,
                    uncertain_branches=cont_uncertain,
                    empty_cont_ids=empty_cont_ids,
                    method_dispatch_targets=method_dispatch_targets,
                    infinite_loop_continue_targets=infinite_loop_continue_targets.get(cont_id, []),
                    infinite_loop_exit_targets=infinite_loop_exit_targets.get(cont_id, []),
                    guard_throw_targets=throw_targets_by_context.get(cont_id, []),
                    guard_throw_target_unknown=cont_id in throw_target_unknown_by_context,
                )
            )

        reachable_indices = (
            self._compute_reachable_instruction_indices(
                all_facts=all_facts,
                edges=edges,
                continuations=continuations,
                cont_fact_map=cont_fact_map,
            )
            if enable_reachability_pruning
            else set()
        )
        if reachable_indices:
            known_components = self._compute_known_edge_components(all_facts, edges)
            reachable_unknown_edges = [
                edge for edge in edges
                if (
                    edge.target is None
                    and edge.kind in {"branch", "jump", "call"}
                    and edge.source in reachable_indices
                )
            ]
            self.last_build_metadata["reachable_unknown_edge_count"] = len(reachable_unknown_edges)

            protected_component_ids: Set[int] = set()
            unbounded_reachable_unknown_edge_count = 0
            for edge in reachable_unknown_edges:
                src_component = known_components.get(edge.source)
                if src_component is not None:
                    protected_component_ids.add(src_component)

                candidate_cont_ids = self._extract_candidate_cont_ids(edge.metadata)
                if not candidate_cont_ids:
                    unbounded_reachable_unknown_edge_count += 1
                    continue

                for cont_id in candidate_cont_ids:
                    target_idx = cont_fact_map.get((cont_id, 0))
                    if target_idx is None:
                        continue
                    target_component = known_components.get(target_idx)
                    if target_component is not None:
                        protected_component_ids.add(target_component)

            self.last_build_metadata["protected_unknown_component_count"] = len(protected_component_ids)
            if unbounded_reachable_unknown_edge_count:
                self.last_build_metadata["unbounded_reachable_unknown_edge_count"] = (
                    unbounded_reachable_unknown_edge_count
                )
                # Preserve backward-compatible key for downstream tooling; pruning now
                # continues with component-local protection instead of global skip.
                self.last_build_metadata["pruning_skipped_due_reachable_unknown"] = False

            pruned_unreachable_unresolved = 0
            filtered_edges: List[ControlFlowEdge] = []
            for edge in edges:
                if (
                    edge.target is None
                    and edge.kind in {"branch", "jump", "call"}
                    and edge.source not in reachable_indices
                ):
                    source_component = known_components.get(edge.source)
                    if (
                        source_component is not None
                        and source_component in protected_component_ids
                    ):
                        filtered_edges.append(edge)
                        continue
                    pruned_unreachable_unresolved += 1
                    continue
                filtered_edges.append(edge)
            if pruned_unreachable_unresolved:
                self.last_build_metadata["pruned_unreachable_unresolved_edges"] = (
                    pruned_unreachable_unresolved
                )
            edges = filtered_edges

        self.last_build_metadata["uncertain_branches"] = self._merge_uncertain_branches(
            context_results
        )

        return edges

    @staticmethod
    def _merge_entry_shapes(
        shapes: Sequence[Tuple[List[Tuple[str, Optional[str]]], bool]]
    ) -> Tuple[List[Tuple[str, Optional[str]]], bool]:
        """Merge multiple entry shapes conservatively using common known prefix."""
        if not shapes:
            return ([], True)

        merged_stack = list(shapes[0][0])
        merged_unknown = bool(shapes[0][1])
        for stack_tokens, unknown_below in shapes[1:]:
            common_len = 0
            limit = min(len(merged_stack), len(stack_tokens))
            while common_len < limit and merged_stack[common_len] == stack_tokens[common_len]:
                common_len += 1
            if common_len < len(merged_stack) or common_len < len(stack_tokens):
                merged_unknown = True
            merged_stack = merged_stack[:common_len]
            merged_unknown = merged_unknown or bool(unknown_below)
        return merged_stack, merged_unknown

    @staticmethod
    def _merge_ctrl_regs(
        reg_sets: Sequence[Dict[int, Optional[str]]]
    ) -> Dict[int, Optional[str]]:
        """Conservatively merge control-register snapshots from multiple callers."""
        if not reg_sets:
            return {}
        merged: Dict[int, Optional[str]] = {}
        all_keys: Set[int] = set()
        for regs in reg_sets:
            all_keys.update(regs.keys())
        for key in all_keys:
            if any(key not in regs for regs in reg_sets):
                continue
            first_val = reg_sets[0].get(key)
            if all(regs.get(key) == first_val for regs in reg_sets[1:]):
                merged[key] = first_val
        return merged

    @staticmethod
    def _filter_ctrl_regs(
        regs: Dict[int, Optional[str]],
        allowed_indices: Set[int],
    ) -> Dict[int, Optional[str]]:
        """Keep only modeled inter-context control registers."""
        return {
            idx: value
            for idx, value in regs.items()
            if idx in allowed_indices
        }

    @staticmethod
    def _merge_uncertain_branches(
        context_results: Dict[
            str,
            Tuple[
                Dict[int, List[str]],
                Set[int],
                Dict[str, List[int]],
                Dict[int, List[UncertainInfo]],
            ],
        ],
    ) -> Dict[int, List[UncertainInfo]]:
        """Merge per-context uncertain branch reasons into one instruction-index map."""
        merged: Dict[int, List[UncertainInfo]] = {}
        for _ctx_id, (_branch_map, _returning, _nested, uncertain_map) in context_results.items():
            for inst_idx, reasons in uncertain_map.items():
                bucket = merged.setdefault(inst_idx, [])
                for reason in reasons:
                    if reason not in bucket:
                        bucket.append(reason)
        return merged

    @staticmethod
    def _compute_reachable_instruction_indices(
        all_facts: Sequence[InstructionFact],
        edges: Sequence[ControlFlowEdge],
        continuations: Dict[str, Any],
        cont_fact_map: Dict[Tuple[str, int], int],
    ) -> Set[int]:
        """Compute reachable instruction indices from main/method entries."""
        entry_indices: Set[int] = set()
        main_indices = [f.index for f in all_facts if f.continuation_id is None]
        if main_indices:
            entry_indices.add(min(main_indices))

        for cont_id, cont in continuations.items():
            if getattr(cont, "kind", None) == "method":
                entry_idx = cont_fact_map.get((cont_id, 0))
                if entry_idx is not None:
                    entry_indices.add(entry_idx)

        if not entry_indices:
            return set()

        adjacency: Dict[int, List[int]] = defaultdict(list)
        for edge in edges:
            if edge.target is None:
                continue
            adjacency[edge.source].append(edge.target)

        visited: Set[int] = set()
        stack: List[int] = list(entry_indices)
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            for nxt in adjacency.get(idx, []):
                if nxt not in visited:
                    stack.append(nxt)
        return visited

    @staticmethod
    def _compute_known_edge_components(
        all_facts: Sequence[InstructionFact],
        edges: Sequence[ControlFlowEdge],
    ) -> Dict[int, int]:
        """Compute weakly-connected component IDs using only resolved CFG edges."""
        indices: Set[int] = {fact.index for fact in all_facts}
        adjacency: Dict[int, Set[int]] = {idx: set() for idx in indices}
        for edge in edges:
            if edge.target is None:
                continue
            if edge.source not in adjacency or edge.target not in adjacency:
                continue
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

        component_by_index: Dict[int, int] = {}
        component_id = 0
        for idx in sorted(indices):
            if idx in component_by_index:
                continue
            stack = [idx]
            component_by_index[idx] = component_id
            while stack:
                current = stack.pop()
                for nxt in adjacency.get(current, set()):
                    if nxt in component_by_index:
                        continue
                    component_by_index[nxt] = component_id
                    stack.append(nxt)
            component_id += 1
        return component_by_index

    @staticmethod
    def _extract_candidate_cont_ids(metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Extract concrete continuation candidate IDs from unresolved edge metadata."""
        if not metadata:
            return []
        raw_candidates = metadata.get("candidate_targets")
        if not isinstance(raw_candidates, (list, tuple, set)):
            return []
        candidates: List[str] = []
        for candidate in raw_candidates:
            if not isinstance(candidate, str) or not candidate:
                continue
            if is_unknown_cont(candidate):
                continue
            candidates.append(candidate)
        return candidates

    def _emit_resolved_or_uncertain_edges(
        self,
        *,
        idx: int,
        cont_ids: Sequence[str],
        cont_fact_map: Dict[Tuple[str, int], int],
        uncertain_branches: Dict[int, List[UncertainInfo]],
        edge_kind: str,
        unresolved_edge_kind: str,
        overapprox_targets: Optional[List[int]],
        overapprox_condition: bool,
        filter_unknown_placeholders: bool,
        metadata_extras: Optional[Dict[str, Any]],
        empty_cont_ids: Set[str],
        unresolved_nonempty_ids: Callable[[Sequence[str]], List[str]],
    ) -> Tuple[List[ControlFlowEdge], bool]:
        """
        Emit resolved and unresolved CFG edges for continuation-based branching.

        Returns:
            (edges, added_unknown_edge)
        """
        emitted_edges: List[ControlFlowEdge] = []
        resolved_targets: List[int] = []
        for cont_id in cont_ids:
            if (cont_id, 0) in cont_fact_map:
                resolved_targets.append(cont_fact_map[(cont_id, 0)])

        used_overapprox = False
        if not resolved_targets and overapprox_condition and overapprox_targets:
            resolved_targets = list(overapprox_targets)
            used_overapprox = True

        resolved_metadata_extras = metadata_extras or {}
        added_unknown_edge = False

        def unknown_edge_metadata(
            candidate_targets: Sequence[str],
            stack_uncertain: bool,
        ) -> Dict[str, Any]:
            metadata: Dict[str, Any] = {
                "stack_uncertain": stack_uncertain,
                "candidate_targets": candidate_targets,
            }
            if unresolved_edge_kind == "branch":
                metadata["target_unknown"] = True
            else:
                metadata["continuation_resolved"] = False
            return metadata

        if resolved_targets:
            for target in resolved_targets:
                metadata: Dict[str, Any] = {
                    "continuation_resolved": not used_overapprox,
                }
                metadata.update(resolved_metadata_extras)
                emitted_edges.append(
                    ControlFlowEdge(
                        source=idx,
                        target=target,
                        kind=edge_kind,
                        metadata=metadata,
                    )
                )

            # Dict-dispatch over-approximation should suppress the extra
            # fallback unknown edge when stack is marked uncertain.
            if used_overapprox and unresolved_edge_kind == "branch":
                added_unknown_edge = True

            if cont_ids and len(resolved_targets) < len(cont_ids):
                unresolved_ids = unresolved_nonempty_ids(cont_ids)
                if filter_unknown_placeholders:
                    unresolved_ids = [
                        cid for cid in unresolved_ids if not is_unknown_cont(cid)
                    ]
                if unresolved_ids:
                    emitted_edges.append(
                        ControlFlowEdge(
                            source=idx,
                            target=None,
                            kind=unresolved_edge_kind,
                            metadata=unknown_edge_metadata(
                                unresolved_ids,
                                stack_uncertain=idx in uncertain_branches,
                            ),
                        )
                    )
                    added_unknown_edge = True
        else:
            unresolved_ids = [cid for cid in cont_ids if cid not in empty_cont_ids]
            if filter_unknown_placeholders:
                unresolved_ids = [
                    cid for cid in unresolved_ids if not is_unknown_cont(cid)
                ]
            if unresolved_ids or idx in uncertain_branches:
                emitted_edges.append(
                    ControlFlowEdge(
                        source=idx,
                        target=None,
                        kind=unresolved_edge_kind,
                        metadata=unknown_edge_metadata(
                            unresolved_ids,
                            stack_uncertain=idx in uncertain_branches,
                        ),
                    )
                )
                added_unknown_edge = True

        if (idx in uncertain_branches) and not added_unknown_edge:
            emitted_edges.append(
                ControlFlowEdge(
                    source=idx,
                    target=None,
                    kind=unresolved_edge_kind,
                    metadata=unknown_edge_metadata(
                        cont_ids if cont_ids else [],
                        stack_uncertain=True,
                    ),
                )
            )
            added_unknown_edge = True

        return emitted_edges, added_unknown_edge

    def _build_cfg_for_context(
        self,
        facts: Sequence[InstructionFact],
        context_id: Optional[str],
        cont_fact_map: Dict[Tuple[str, int], int],
        branch_cont_map: Dict[int, List[str]],
        returning_branches: Set[int],
        return_targets: Optional[List[int]],
        alt_return_targets: Optional[List[int]] = None,
        alt_return_target_unknown: bool = False,
        uncertain_branches: Optional[Dict[int, List[UncertainInfo]]] = None,
        empty_cont_ids: Optional[Set[str]] = None,
        method_dispatch_targets: Optional[List[int]] = None,
        infinite_loop_continue_targets: Optional[List[int]] = None,
        infinite_loop_exit_targets: Optional[List[int]] = None,
        guard_throw_targets: Optional[List[int]] = None,
        guard_throw_target_unknown: bool = False,
    ) -> List[ControlFlowEdge]:
        """
        Build CFG edges for a single context (main or continuation).

        For main context, creates call_cont edges to continuations.
        For continuation context, creates normal control flow only.
        """
        edges = []
        if not facts:
            return edges

        total = len(facts)
        is_main = (context_id == MAIN_CONTEXT)
        return_targets = return_targets or []
        alt_return_targets = alt_return_targets or []
        branch_cont_map = branch_cont_map or {}
        returning_branches = returning_branches or set()
        uncertain_branches = uncertain_branches or {}
        empty_cont_ids = empty_cont_ids or set()
        method_dispatch_targets = method_dispatch_targets or []
        infinite_loop_continue_targets = infinite_loop_continue_targets or []
        infinite_loop_exit_targets = infinite_loop_exit_targets or []
        guard_throw_targets = guard_throw_targets or []

        def unresolved_nonempty_ids(cont_ids: Sequence[str]) -> List[str]:
            resolved_ids = {cid for cid in cont_ids if (cid, 0) in cont_fact_map}
            return [
                cid for cid in cont_ids
                if cid not in resolved_ids and cid not in empty_cont_ids
            ]

        for i, fact in enumerate(facts):
            idx = fact.index
            opcode = fact.opcode
            upper_opcode = opcode.upper()
            is_call = upper_opcode in CALL_CONT_OPCODES or any(
                upper_opcode.startswith(prefix) for prefix in CALL_ARGS_SUFFIX_PREFIXES
            )
            call_branch_cont_ids = branch_cont_map.get(idx, [])
            call_cont_ids = branch_cont_map.get(idx, []) if is_call else []
            call_cont_targets: List[int] = []
            if is_call and call_cont_ids:
                for cont_id in call_cont_ids:
                    if (cont_id, 0) in cont_fact_map:
                        call_cont_targets.append(cont_fact_map[(cont_id, 0)])
            call_resolved = bool(call_cont_targets)

            # IFELSE variants have no fallthrough because both branches are explicit continuations
            # (true branch and false branch are both taken from stack/inline/ref).
            #
            # Non-loop call-like transfers (CALL*/EXECUTE/TRY*/BOOLEVAL) never execute
            # the next instruction directly, even when targets are unresolved.
            #
            # Loop calls keep existing behavior: unresolved loops preserve conservative
            # fallthrough; resolved loops suppress it except REPEAT/REPEATBRK where c<=0
            # exits immediately.
            call_no_fallthrough = (
                is_call
                and (
                    upper_opcode not in LOOP_CALL_OPCODES
                    or (
                        call_resolved
                        and upper_opcode not in {"REPEAT", "REPEATBRK"}
                    )
                )
            )
            no_fallthrough = (
                upper_opcode in {"IFELSE", "IFREFELSE", "IFELSEREF", "IFREFELSEREF"}
                or call_no_fallthrough
                or upper_opcode in INFINITE_LOOP_OPCODES
                or upper_opcode == "PFXDICTGETEXEC"
            )
            if i + 1 < total and upper_opcode not in TERMINATORS and not no_fallthrough:
                next_fact = facts[i + 1]
                edges.append(
                    ControlFlowEdge(source=idx, target=next_fact.index, kind="fallthrough")
                )

            # Skip CONDITIONAL_RETURNS (IFRET, IFNOTRET, etc.) - they don't consume
            # continuations and are handled separately as guard_return edges below.
            if upper_opcode in CONDITIONAL_BRANCHES and upper_opcode not in CONDITIONAL_RETURNS:
                cont_ids = branch_cont_map.get(idx, [])
                has_concrete_targets = any((cid, 0) in cont_fact_map for cid in cont_ids)
                dict_dispatch_overapprox = (
                    upper_opcode in DICT_DISPATCH_OPCODES
                    and bool(method_dispatch_targets)
                    and not has_concrete_targets
                )
                emitted_edges, _ = self._emit_resolved_or_uncertain_edges(
                    idx=idx,
                    cont_ids=cont_ids,
                    cont_fact_map=cont_fact_map,
                    uncertain_branches=uncertain_branches,
                    edge_kind="call_cont" if idx in returning_branches else "jump",
                    unresolved_edge_kind="branch",
                    overapprox_targets=method_dispatch_targets,
                    overapprox_condition=dict_dispatch_overapprox,
                    filter_unknown_placeholders=False,
                    metadata_extras={
                        "dict_dispatch_overapprox": dict_dispatch_overapprox,
                    },
                    empty_cont_ids=empty_cont_ids,
                    unresolved_nonempty_ids=unresolved_nonempty_ids,
                )
                edges.extend(emitted_edges)

                if upper_opcode in DICT_DISPATCH_EXEC_OPCODES and i + 1 < total:
                    # EXEC-style dictionary dispatch behaves like a conditional call:
                    # on success it returns to the next instruction.
                    next_fact = facts[i + 1]
                    edges.append(
                        ControlFlowEdge(source=idx, target=next_fact.index, kind="call_return")
                    )

            if upper_opcode in UNCONDITIONAL_BRANCHES:
                cont_ids = branch_cont_map.get(idx, [])
                emitted_edges, _ = self._emit_resolved_or_uncertain_edges(
                    idx=idx,
                    cont_ids=cont_ids,
                    cont_fact_map=cont_fact_map,
                    uncertain_branches=uncertain_branches,
                    edge_kind="jump",
                    unresolved_edge_kind="jump",
                    overapprox_targets=None,
                    overapprox_condition=False,
                    filter_unknown_placeholders=False,
                    metadata_extras=None,
                    empty_cont_ids=empty_cont_ids,
                    unresolved_nonempty_ids=unresolved_nonempty_ids,
                )
                edges.extend(emitted_edges)

            if is_call:
                unknown_call_edge_emitted = False
                call_dispatch_overapprox = (
                    not call_cont_targets
                    and bool(call_cont_ids)
                    and bool(method_dispatch_targets)
                    and all(
                        is_unknown_cont(cid) or cid in empty_cont_ids
                        for cid in call_cont_ids
                    )
                )
                emitted_edges, unknown_call_edge_emitted = self._emit_resolved_or_uncertain_edges(
                    idx=idx,
                    cont_ids=call_cont_ids,
                    cont_fact_map=cont_fact_map,
                    uncertain_branches=uncertain_branches,
                    edge_kind="call_cont",
                    unresolved_edge_kind="call",
                    overapprox_targets=method_dispatch_targets,
                    overapprox_condition=call_dispatch_overapprox,
                    filter_unknown_placeholders=True,
                    metadata_extras={
                        "dynamic_dispatch_overapprox": call_dispatch_overapprox,
                    },
                    empty_cont_ids=empty_cont_ids,
                    unresolved_nonempty_ids=unresolved_nonempty_ids,
                )
                edges.extend(emitted_edges)
                call_resolved = any(
                    edge.source == idx and edge.target is not None and edge.kind == "call_cont"
                    for edge in emitted_edges
                )
            else:
                unknown_call_edge_emitted = False

            # Unresolved call: add fallback call_return edge to next instruction
            unresolved_call_like = (
                (
                    upper_opcode in CALL_OPCODES
                    or any(upper_opcode.startswith(prefix) for prefix in CALL_ARGS_SUFFIX_PREFIXES)
                )
                and not call_resolved
                and upper_opcode not in INFINITE_LOOP_OPCODES
            )
            if unresolved_call_like:
                if not unknown_call_edge_emitted:
                    edges.append(
                        ControlFlowEdge(
                            source=idx,
                            target=None,
                            kind="call",
                            metadata={
                                "stack_uncertain": idx in uncertain_branches,
                                "candidate_targets": list(call_branch_cont_ids),
                                "continuation_resolved": False,
                            },
                        )
                    )
                if i + 1 < total:
                    next_fact = facts[i + 1]
                    edges.append(
                        ControlFlowEdge(source=idx, target=next_fact.index, kind="call_return")
                    )

            if upper_opcode in CONDITIONAL_GUARDS:
                for throw_target in guard_throw_targets:
                    edges.append(
                        ControlFlowEdge(
                            source=idx,
                            target=throw_target,
                            kind="guard_throw",
                            metadata={
                                "terminates": False,
                                "via_c2": True,
                                "continuation_resolved": True,
                            },
                        )
                    )

                if guard_throw_target_unknown or not guard_throw_targets:
                    edges.append(
                        ControlFlowEdge(
                            source=idx,
                            target=None,
                            kind="guard_throw",
                            metadata={
                                "terminates": True,
                                "exception_target_unknown": True,
                                "continuation_resolved": False,
                            },
                        )
                    )

            if upper_opcode in CONDITIONAL_RETURNS:
                edges.append(
                    ControlFlowEdge(
                        source=idx,
                        target=None,
                        kind="guard_return",
                        metadata={"terminates": True},
                    )
                )

        if not is_main and return_targets:
            fact_opcode_by_index = {fact.index: fact.opcode.upper() for fact in facts}
            infinite_continue_set = set(infinite_loop_continue_targets)
            infinite_exit_set = set(infinite_loop_exit_targets)
            return_sources = [
                fact.index
                for fact in facts
                if self._is_continuation_return(fact.opcode)
            ]
            if not return_sources and facts:
                last_opcode = facts[-1].opcode
                implicit_ok = (
                    last_opcode not in TERMINATORS
                    and last_opcode not in CONDITIONAL_BRANCHES
                    and last_opcode not in CONDITIONAL_GUARDS
                    and last_opcode not in CALL_OPCODES
                )
                if implicit_ok:
                    return_sources = [facts[-1].index]

            for src in return_sources:
                src_opcode = fact_opcode_by_index.get(src, "")
                emitted_target_set: Set[int] = set()
                emitted_targets: List[int] = []
                for target in return_targets:
                    if target is None:
                        continue
                    if not self._allow_return_target_for_source(
                        src_opcode=src_opcode,
                        target=target,
                        infinite_continue_targets=infinite_continue_set,
                        infinite_exit_targets=infinite_exit_set,
                    ):
                        continue
                    edges.append(
                        ControlFlowEdge(
                            source=src,
                            target=target,
                            kind="return_cont",
                            metadata={"cont_id": context_id}
                        )
                    )
                    emitted_target_set.add(target)
                    emitted_targets.append(target)

                alt_emitted_targets: List[int] = []
                if src_opcode in {"RETALT", "RETBOOL"}:
                    for alt_target in alt_return_targets:
                        if alt_target in emitted_target_set:
                            continue
                        if infinite_continue_set or infinite_exit_set:
                            if not self._allow_return_target_for_source(
                                src_opcode=src_opcode,
                                target=alt_target,
                                infinite_continue_targets=infinite_continue_set,
                                infinite_exit_targets=infinite_exit_set,
                            ):
                                continue
                        edges.append(
                            ControlFlowEdge(
                                source=src,
                                target=alt_target,
                                kind="return_cont",
                                metadata={"cont_id": context_id, "via_c1": True},
                            )
                        )
                        emitted_target_set.add(alt_target)
                        alt_emitted_targets.append(alt_target)

                # RETALT returns via c1; when caller c1 target is not known in this context,
                # preserve soundness with an explicit unknown successor.
                add_unknown_alt_return = False
                if self._is_alt_return_opcode(src_opcode):
                    add_unknown_alt_return = alt_return_target_unknown or not (
                        emitted_targets or alt_emitted_targets
                    )
                elif src_opcode == "RETBOOL":
                    # RETBOOL may return via c0 or c1. Keep unknown c1 path unless both
                    # destinations are explicitly represented (loop continue + exit).
                    if not infinite_continue_set and not infinite_exit_set:
                        add_unknown_alt_return = alt_return_target_unknown or not alt_emitted_targets
                    else:
                        all_emitted = set(emitted_targets) | set(alt_emitted_targets)
                        has_continue = any(t in infinite_continue_set for t in all_emitted)
                        has_exit = any(t in infinite_exit_set for t in all_emitted)
                        add_unknown_alt_return = alt_return_target_unknown or not (has_continue and has_exit)

                if add_unknown_alt_return:
                    edges.append(
                        ControlFlowEdge(
                            source=src,
                            target=None,
                            kind="jump",
                            metadata={
                                "continuation_resolved": False,
                                "return_target_unknown": True,
                            },
                        )
                    )

        return edges

    @staticmethod
    def _is_continuation_return(opcode: str) -> bool:
        upper = opcode.upper()
        return upper in CFGBuilder._RETURN_OPCODES or upper.startswith("RETARGS")

    @staticmethod
    def _is_alt_return_opcode(opcode: str) -> bool:
        upper = opcode.upper()
        return upper == "RETALT"

    @staticmethod
    def _is_normal_return_opcode(opcode: str) -> bool:
        upper = opcode.upper()
        if upper in {"RET", "RETDATA", "RETVARARGS"}:
            return True
        return upper.startswith("RETARGS")

    @staticmethod
    def _allow_return_target_for_source(
        *,
        src_opcode: str,
        target: int,
        infinite_continue_targets: Set[int],
        infinite_exit_targets: Set[int],
    ) -> bool:
        """Filter return edges for infinite-loop calls using return opcode semantics."""
        if not infinite_continue_targets and not infinite_exit_targets:
            # Without loop hints, caller c1 target is generally unknown.
            if CFGBuilder._is_alt_return_opcode(src_opcode):
                return False
            return True

        in_continue = target in infinite_continue_targets
        in_exit = target in infinite_exit_targets
        if not in_continue and not in_exit:
            return True

        if CFGBuilder._is_alt_return_opcode(src_opcode):
            # RETALT/RETFALSE should model loop break for AGAIN*/AGAINBRK*.
            return not (in_continue and not in_exit)

        if CFGBuilder._is_normal_return_opcode(src_opcode):
            # RET/RETTRUE/RETDATA/RETARGS/RETVARARGS should continue infinite loops.
            return not (in_exit and not in_continue)

        # Mixed-return opcodes (e.g., RETBOOL/BRANCH) may go either way.
        return True

    def build_basic_blocks_with_continuations(
        self,
        all_facts: Sequence[InstructionFact],
        cfg_edges: Sequence[ControlFlowEdge],
        continuations: Dict[str, Any]
    ) -> List[BasicBlock]:
        """
        Build basic blocks for all contexts (main + continuations).

        Each context gets its own set of blocks with appropriate context field.
        Blocks are renumbered globally and successors are computed from cfg_edges.
        """
        blocks = []

        # Group facts by context
        main_facts = [f for f in all_facts if f.continuation_id is None]
        cont_facts_by_id = {}
        for cont_id in continuations:
            cont_facts_by_id[cont_id] = [f for f in all_facts if f.continuation_id == cont_id]

        # Build blocks for main context
        main_blocks = self._build_blocks_for_context(main_facts, cfg_edges, MAIN_CONTEXT)
        blocks.extend(main_blocks)

        # Build blocks for each continuation context
        for cont_id, cont_facts in cont_facts_by_id.items():
            cont_blocks = self._build_blocks_for_context(cont_facts, cfg_edges, cont_id)
            blocks.extend(cont_blocks)

        # Step 1: Renumber blocks globally
        for global_id, block in enumerate(blocks):
            block.id = global_id

        # Step 2: Build instruction-to-block mapping
        instr_to_block = {}
        for block in blocks:
            for idx in block.instruction_indices:
                instr_to_block[idx] = block.id

        # Step 3: Add successors from cfg_edges
        # For edges with unknown targets (unresolved continuations/jumps),
        # set has_unknown_successor=True instead of using a magic ID like -1.

        for edge in cfg_edges:
            # Handle edges with known targets
            if edge.target is not None:
                src_block_id = instr_to_block.get(edge.source)
                dst_block_id = instr_to_block.get(edge.target)

                if src_block_id is not None and dst_block_id is not None:
                    # Self-loops are important for detecting infinite loops and analyzing
                    # loop-carried dependencies in data flow analysis
                    if src_block_id != dst_block_id:
                        blocks[src_block_id].successors.append(dst_block_id)
                    elif edge.kind != "fallthrough":
                        # Preserve non-fallthrough self-loops (e.g., branch/jump/call/return
                        # back-edges) inside compact single-block loops.
                        blocks[src_block_id].successors.append(dst_block_id)

            # Handle branch edges with unknown targets
            # These represent unresolved continuation/jump targets
            elif edge.kind in {"branch", "jump", "call", "guard_throw", "guard_return"}:
                src_block_id = instr_to_block.get(edge.source)
                if src_block_id is not None:
                    # Mark the block as having unknown successor(s)
                    # Detectors should check this flag and handle conservatively
                    blocks[src_block_id].has_unknown_successor = True

        # Step 4: Deduplicate and sort successors
        for block in blocks:
            block.successors = sorted(set(block.successors))

        return blocks

    def _build_blocks_for_context(
        self,
        facts: Sequence[InstructionFact],
        cfg_edges: Sequence[ControlFlowEdge],
        context: str
    ) -> List[BasicBlock]:
        """Build basic blocks for a single context."""
        if not facts:
            return []

        blocks = []
        current = []
        context_indices: Set[int] = {fact.index for fact in facts}
        leaders: Set[int] = {facts[0].index}
        for edge in cfg_edges:
            if edge.target is None:
                continue
            if edge.target in context_indices:
                leaders.add(edge.target)

        boundary_opcodes = (
            TERMINATORS
            | CONDITIONAL_BRANCHES
            | CONDITIONAL_GUARDS
            | CONDITIONAL_RETURNS
            | CALL_OPCODES
            | LOOP_CALL_OPCODES
        )

        for i, fact in enumerate(facts):
            if current and fact.index in leaders:
                blocks.append(
                    BasicBlock(
                        id=len(blocks),
                        instruction_indices=current.copy(),
                        context=context,
                    )
                )
                current = []

            current.append(fact.index)
            upper_opcode = fact.opcode.upper()
            is_boundary = (
                upper_opcode in boundary_opcodes
                or any(upper_opcode.startswith(prefix) for prefix in CALL_ARGS_SUFFIX_PREFIXES)
                or i == len(facts) - 1
            )
            if is_boundary:
                if current:
                    block = BasicBlock(
                        id=len(blocks),  # This will be renumbered globally later
                        instruction_indices=current.copy(),
                        context=context
                    )
                    blocks.append(block)
                    current = []

        # Blocks returned with context-local IDs; caller renumbers globally
        return blocks
