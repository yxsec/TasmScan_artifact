"""
Program analyzer that builds facts from disassembled instructions.

This module is the main coordinator for TVM program analysis, delegating to:
- ContinuationResolver: Extracts and maps continuations
- CFGBuilder: Builds control flow graphs and basic blocks
- StackAnalyzer: Analyzes stack heights

Type Annotation Notes (typing):
=============================
This module uses `Dict[str, Any]` for the `continuations` parameter in several
functions. This is intentional due to the dynamic nature of continuation data
structures from the disassembler.

Continuations are nested data structures with varying shapes:
- They can contain instruction lists, nested continuations, metadata
- The exact structure depends on the disassembler version and opcode type
- Using a strict TypedDict would require extensive runtime validation

The `Any` type here represents: Dict with string keys containing either:
- List[Instruction]: The continuation's instruction sequence
- Dict[str, Any]: Nested continuation data
- Optional metadata fields

For type-safe access patterns, see the actual usage sites which perform
appropriate runtime checks before accessing continuation data.
"""
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..config import INMSGPARAM_SENDER_INDEX

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from pytoniq_core import Cell
else:
    Cell = Any  # type: ignore[misc,assignment]

from .constants import (
    CALL_OPCODES,
    DEFAULT_EVENT_MAP,
    END_LOOP_OPCODES,
    MAIN_CONTEXT,
)
from .facts import (
    AnalysisFacts,
    BlockSummary,
    CallGraphEdge,
    CallSite,
    Continuation,
    Event,
    InstructionFact,
)
from .cfg_builder import CFGBuilder
from .continuation_resolver import ContinuationResolver
from .stack_analyzer import StackAnalyzer
from .utils import extract_int_arg

logger = logging.getLogger(__name__)


class ProgramAnalyzer:
    """Analyzes disassembled instructions to build AnalysisFacts."""

    def __init__(self):
        self.event_map = DEFAULT_EVENT_MAP
        self._stack_analyzer = StackAnalyzer()
        self._continuation_resolver = ContinuationResolver()
        self._cfg_builder = CFGBuilder(self._continuation_resolver)

    def analyze(
        self,
        instructions: List,
        cell: Optional[Cell] = None,
        *,
        enable_known_prefix_fallback: bool = True,
        enable_entry_shape_propagation: bool = True,
        enable_ctrl_reg_propagation: bool = True,
        enable_varargs_depth_fix: bool = True,
        enable_reachability_pruning: bool = True,
    ) -> AnalysisFacts:
        """
        Build AnalysisFacts from instruction list with continuation support.

        Args:
            instructions: List of Instruction objects
            cell: Root cell for hash

        Returns:
            AnalysisFacts with all analysis results including continuation subgraphs
        """
        context_id = cell.hash.hex() if cell else "root"

        # Step 1a: Extract continuations : PushMap, InlineMap, and Continuation bodies(methods, if-else branches)
        continuation_extraction_error = None
        analysis_incomplete_reasons: List[str] = []
        try:
            push_map_by_context, inline_map_by_context, continuations = \
                self._continuation_resolver.extract_continuations(instructions) 
        except RecursionError as e:
            # Return partial results with warning metadata when continuation extraction fails
            # This allows analysis to proceed with main context only
            logger.warning(
                f"Continuation extraction failed due to excessive depth: {e}. "
                "Analysis will proceed with main context only."
            )
            continuation_extraction_error = str(e)
            analysis_incomplete_reasons.append("continuation_extraction_failed")
            push_map_by_context = {}
            inline_map_by_context = {}
            continuations = {}

        # Step 1b: Materialize end-loop bodies into separate continuations for better analysis(end-loop bodies)
        instructions = self._materialize_end_loop_bodies(
            list(instructions),
            push_map_by_context=push_map_by_context,
            inline_map_by_context=inline_map_by_context,
            continuations=continuations,
        ) 

        # Step 2a: Build instruction facts for main context
        main_facts = []
        for i, inst in enumerate(instructions):
            fact = InstructionFact(
                instruction=inst,
                index=i,
                offset=0,
                length=0,
                hash=context_id,
                continuation_id=None,
                parent_index=None,
            )
            main_facts.append(fact)

        # Build instruction facts for each continuation
        # These have separate index spaces within their continuation
        cont_facts = []
        cont_fact_map = {}  # Maps (cont_id, local_index) -> global_index in combined list

        # Step 2b: Assign global indices to continuation instructions, following main context
        next_global_index = len(main_facts)

        for cont_id, continuation in continuations.items():
            for local_idx, cont_inst in enumerate(continuation.instructions):
                global_idx = next_global_index
                fact = InstructionFact(
                    instruction=cont_inst,
                    index=global_idx,
                    offset=0,
                    length=0,
                    hash=f"{context_id}:{cont_id}",
                    continuation_id=cont_id,
                    parent_index=None,
                )
                cont_facts.append(fact)
                cont_fact_map[(cont_id, local_idx)] = global_idx
                next_global_index += 1

        # Combine all facts (main + continuations)
        all_facts = main_facts + cont_facts
        cont_facts_by_id: Dict[str, List[InstructionFact]] = {}
        for cont_id in continuations:
            cont_facts_by_id[cont_id] = [
                fact for fact in cont_facts if fact.continuation_id == cont_id
            ]

        # Resolve parent instruction indices for continuations
        pushcont_to_cont_ids: Dict[int, List[str]] = {}
        continuation_parent_branch: Dict[str, int] = {}

        def assign_pushes(ctx_id: str, facts_in_ctx: Sequence[InstructionFact]):
            for local_idx, fact in enumerate(facts_in_ctx):
                local_key = (ctx_id, local_idx)
                cont_ids = push_map_by_context.get(local_key)
                if cont_ids:
                    pushcont_to_cont_ids[fact.index] = list(cont_ids)

        assign_pushes(MAIN_CONTEXT, main_facts)

        inline_cont_map: Dict[int, List[str]] = {}

        def assign_inline(ctx_id: str, facts_in_ctx: Sequence[InstructionFact]):
            for local_idx, fact in enumerate(facts_in_ctx):
                local_key = (ctx_id, local_idx)
                cont_ids = inline_map_by_context.get(local_key)
                if cont_ids:
                    inline_cont_map[fact.index] = list(cont_ids)

        assign_inline(MAIN_CONTEXT, main_facts)

        # Step 3: Map continuations to their parent instruction indices using the resolver's mapping
        branch_map, _, _, _, _ = \
            self._continuation_resolver.map_continuations(
                main_facts,
                pushcont_to_cont_ids,
                inline_cont_map,
            )
        for idx, cont_ids in branch_map.items():
            for cont_id in cont_ids:
                continuation_parent_branch.setdefault(cont_id, idx)
        # Process nested continuations within each continuation body
        for cont_id, cont_facts_list in cont_facts_by_id.items():
            assign_pushes(cont_id, cont_facts_list)
            assign_inline(cont_id, cont_facts_list)
            nested_map, _, _, _, _ = \
                self._continuation_resolver.map_continuations(
                    cont_facts_list, pushcont_to_cont_ids, inline_cont_map,
                )
            for idx, nested_cont_ids in nested_map.items():
                for nested_cont_id in nested_cont_ids:
                    continuation_parent_branch.setdefault(nested_cont_id, idx)

        for cont in continuations.values():
            cont.parent_instruction_index = continuation_parent_branch.get(cont.id)

        for fact in cont_facts:
            cont = continuations[fact.continuation_id]
            fact.parent_index = cont.parent_instruction_index

        events = self._collect_events(all_facts) # Collect semantic events from all facts (main + continuations)

        # Step 4: Build CFG with continuation awareness
        cfg_edges = self._cfg_builder.build_cfg_with_continuations(
            all_facts,
            pushcont_to_cont_ids,
            inline_cont_map,
            continuations,
            cont_fact_map,
            enable_known_prefix_fallback=enable_known_prefix_fallback,
            enable_entry_shape_propagation=enable_entry_shape_propagation,
            enable_ctrl_reg_propagation=enable_ctrl_reg_propagation,
            enable_varargs_depth_fix=enable_varargs_depth_fix,
            enable_reachability_pruning=enable_reachability_pruning,
        )
        cfg_build_meta = dict(getattr(self._cfg_builder, "last_build_metadata", {}) or {})
        uncertain_branches_data = cfg_build_meta.get("uncertain_branches", {})
        if cfg_build_meta and cfg_build_meta.get("entry_shape_widened", False):
            analysis_incomplete_reasons.append("cfg_entry_shape_widened")
        if cfg_build_meta and not cfg_build_meta.get("entry_shape_converged", True):
            analysis_incomplete_reasons.append("cfg_entry_shape_non_converged")

        # Build basic blocks for all contexts
        basic_blocks = self._cfg_builder.build_basic_blocks_with_continuations(
            all_facts,
            cfg_edges,
            continuations
        )

        # Compute entry points (main + method continuations)
        entry_points = []
        instr_to_block = {}
        for block in basic_blocks:
            for idx in block.instruction_indices:
                instr_to_block[idx] = block.id
        incoming_by_index: Dict[int, int] = {}
        for edge in cfg_edges:
            if edge.target is None:
                continue
            incoming_by_index[edge.target] = incoming_by_index.get(edge.target, 0) + 1

        def add_entry(kind: str, context: str, instr_index: int, method_id: Optional[int] = None) -> None:
            block_id = instr_to_block.get(instr_index)
            if block_id is None:
                return
            info = {"kind": kind, "context": context, "block_id": block_id}
            if method_id is not None:
                info["method_id"] = method_id
            entry_points.append(info)

        if main_facts:
            add_entry(MAIN_CONTEXT, MAIN_CONTEXT, min(f.index for f in main_facts))

        for cont_id, cont in continuations.items():
            cont_facts_list = cont_facts_by_id.get(cont_id, [])
            if not cont_facts_list:
                continue
            entry_idx = min(f.index for f in cont_facts_list)
            if cont.kind != "method" and incoming_by_index.get(entry_idx, 0) > 0:
                # Continuation has a concrete caller edge; do not classify as module entry.
                continue
            add_entry(cont.kind, cont_id, entry_idx, method_id=cont.method_id)

        # Analyze stack (needs to handle context boundaries)
        stack_states = self._stack_analyzer.analyze_stack(all_facts, continuations, cfg_edges)
        stack_analysis_meta = dict(getattr(self._stack_analyzer, "last_analysis_metadata", {}) or {})
        if stack_analysis_meta and not stack_analysis_meta.get("global_converged", True):
            analysis_incomplete_reasons.append("stack_analysis_non_converged")

        # Collect call sites
        call_sites = self._collect_call_sites(all_facts)

        # Build context -> method mapping for call graph generation
        context_method_map = self._build_context_method_map(continuations)

        # Build call graph
        call_graph_edges = self._build_call_graph(
            context_id,
            call_sites,
            context_method_map=context_method_map,
        )

        # Build summary
        summary = self._build_summary(
            context_id,
            context_id,
            events,
            call_sites,
            all_facts
        )

        metadata = {
            "block_hash": context_id,
            "instruction_count": len(all_facts),
            "continuation_count": len(continuations),
            "continuations": continuations,
            "uncertain_branches": uncertain_branches_data,
        }
        if cfg_build_meta:
            metadata["cfg_build"] = cfg_build_meta
        if stack_analysis_meta:
            metadata["stack_analysis"] = stack_analysis_meta
        if continuation_extraction_error:
            metadata["continuation_extraction_failed"] = True
            metadata["continuation_extraction_error"] = continuation_extraction_error
        if analysis_incomplete_reasons:
            metadata["analysis_incomplete"] = True
            metadata["analysis_incomplete_reasons"] = list(dict.fromkeys(analysis_incomplete_reasons))

        return AnalysisFacts(
            instructions=all_facts,
            events=events,
            cfg_edges=cfg_edges,
            basic_blocks=basic_blocks,
            stack_states=stack_states,
            call_sites=call_sites,
            call_graph_edges=call_graph_edges,
            entry_points=entry_points,
            metadata=metadata,
            summaries=[summary],
        )

    @staticmethod
    def _next_continuation_counter(continuations: Dict[str, Continuation]) -> int: # Determine the next available continuation counter based on existing continuations
        next_counter = 0
        for cont_id in continuations:
            if not cont_id.startswith("cont_"):
                continue
            suffix = cont_id[5:]
            if suffix.isdigit():
                next_counter = max(next_counter, int(suffix) + 1)
        if next_counter == 0:
            next_counter = len(continuations)
        while f"cont_{next_counter}" in continuations:
            next_counter += 1
        return next_counter

    @staticmethod
    def _allocate_continuation_id(
        cont_counter: int,
        continuations: Dict[str, Continuation],
    ) -> Tuple[str, int]:
        while f"cont_{cont_counter}" in continuations:
            cont_counter += 1
        cont_id = f"cont_{cont_counter}"
        return cont_id, cont_counter + 1

    @staticmethod
    def _remap_moved_instruction_metadata(
        source_context: str,
        target_context: str,
        tail_start_local_index: int,
        push_map_by_context: Dict[Tuple[str, int], List[str]],
        inline_map_by_context: Dict[Tuple[str, int], List[str]],
        continuations: Dict[str, Continuation],
    ) -> None:
        def _rekey_tail_entries(
            mapping: Dict[Tuple[str, int], List[str]],
        ) -> None:
            moved_entries: Dict[Tuple[str, int], List[str]] = {}
            for (ctx, local_idx), cont_ids in list(mapping.items()):
                if ctx != source_context or local_idx < tail_start_local_index:
                    continue
                mapping.pop((ctx, local_idx), None)
                new_key = (target_context, local_idx - tail_start_local_index)
                moved_entries.setdefault(new_key, []).extend(list(cont_ids))

            for key, cont_ids in moved_entries.items():
                existing = mapping.setdefault(key, [])
                for cont_id in cont_ids:
                    if cont_id not in existing:
                        existing.append(cont_id)

        _rekey_tail_entries(push_map_by_context)
        _rekey_tail_entries(inline_map_by_context)

        for continuation in continuations.values():
            if continuation.id == target_context:
                continue
            if continuation.parent_context != source_context:
                continue
            if continuation.parent_local_index < tail_start_local_index:
                continue
            continuation.parent_context = target_context
            continuation.parent_local_index -= tail_start_local_index

    def _split_end_loop_body_once(
        self,
        instructions: List,
        *,
        context_id: str,
        continuations: Dict[str, Continuation],
        push_map_by_context: Dict[Tuple[str, int], List[str]],
        inline_map_by_context: Dict[Tuple[str, int], List[str]],
        cont_counter: int,
    ) -> Tuple[List, int, Optional[str]]:
        for local_idx, inst in enumerate(instructions):
            opcode = getattr(inst, "name", "").upper()
            if opcode not in END_LOOP_OPCODES:
                continue

            loop_body = list(instructions[local_idx + 1:])
            if not loop_body:
                return list(instructions), cont_counter, None

            cont_id, cont_counter = self._allocate_continuation_id(cont_counter, continuations)
            continuations[cont_id] = Continuation(
                id=cont_id,
                instructions=loop_body,
                parent_context=context_id,
                parent_local_index=local_idx,
                parent_arg_index=0,
                entry_index=0,
                kind="end_loop_body",
            )

            inline_map_by_context.setdefault((context_id, local_idx), []).append(cont_id)

            self._remap_moved_instruction_metadata(
                source_context=context_id,
                target_context=cont_id,
                tail_start_local_index=local_idx + 1,
                push_map_by_context=push_map_by_context,
                inline_map_by_context=inline_map_by_context,
                continuations=continuations,
            )

            return list(instructions[: local_idx + 1]), cont_counter, cont_id

        return list(instructions), cont_counter, None

    def _materialize_end_loop_bodies(
        self,
        instructions: List,
        *,
        push_map_by_context: Dict[Tuple[str, int], List[str]],
        inline_map_by_context: Dict[Tuple[str, int], List[str]],
        continuations: Dict[str, Continuation],
    ) -> List:
        next_cont_counter = self._next_continuation_counter(continuations)
        main_instructions = list(instructions)
        contexts_to_process: List[str] = [MAIN_CONTEXT, *list(continuations.keys())]
        cursor = 0
        while cursor < len(contexts_to_process):
            context_id = contexts_to_process[cursor]
            cursor += 1

            if context_id == MAIN_CONTEXT:
                context_instructions = list(main_instructions)
            else:
                continuation = continuations.get(context_id)
                if continuation is None:
                    continue
                context_instructions = list(continuation.instructions)

            truncated, next_cont_counter, created_cont_id = self._split_end_loop_body_once(
                context_instructions,
                context_id=context_id,
                continuations=continuations,
                push_map_by_context=push_map_by_context,
                inline_map_by_context=inline_map_by_context,
                cont_counter=next_cont_counter,
            )

            if context_id == MAIN_CONTEXT:
                main_instructions = truncated
            else:
                continuation = continuations.get(context_id)
                if continuation is not None:
                    continuation.instructions = truncated

            if created_cont_id is not None:
                contexts_to_process.append(created_cont_id)

        return main_instructions

    def _collect_events(
        self,
        instructions: Sequence[InstructionFact]
    ) -> List[Event]:
        """Identify semantic events."""
        events = []
        for fact in instructions:
            opcode = fact.opcode
            for event_type, opcode_set in self.event_map.items():
                if opcode in opcode_set:
                    events.append(Event(type=event_type, instruction=fact))
            if opcode == "INMSGPARAM":
                arg_value = self._get_first_int_arg(fact)
                if arg_value == INMSGPARAM_SENDER_INDEX:
                    events.append(Event(type="sender_read", instruction=fact))
        return events

    @staticmethod
    def _get_first_int_arg(fact: InstructionFact) -> Optional[int]:
        return extract_int_arg(fact.arguments, 0)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _collect_call_sites(
        self,
        instructions: Sequence[InstructionFact]
    ) -> List[CallSite]:
        """Identify call sites."""
        call_sites = []
        for fact in instructions:
            if fact.opcode not in CALL_OPCODES:
                continue

            # Determine target type
            target_type = "unknown"
            target_value = None

            # CALLDICT/CALLDICT_LONG carry an immediate method selector.
            # Be tolerant to varying disassembler argument shapes.
            if fact.opcode in {"CALLDICT", "CALLDICT_LONG"}:
                method_id = None
                for arg in fact.arguments or []:
                    if getattr(arg, "type", None) == "uint":
                        arg_value = getattr(arg, "value", None)
                        if isinstance(arg_value, int):
                            method_id = arg_value
                            break
                if method_id is None:
                    method_id = self._get_first_int_arg(fact)
                if method_id is not None:
                    target_type = "method"
                    target_value = method_id

            call_sites.append(
                CallSite(
                    instruction=fact,
                    target_type=target_type,
                    target=target_value,
                )
            )

        return call_sites

    def _build_call_graph(
        self,
        caller_id: str,
        call_sites: Sequence[CallSite],
        context_method_map: Optional[Dict[str, int]] = None,
    ) -> List[CallGraphEdge]:
        """Build call graph edges."""
        edges = []
        context_method_map = context_method_map or {MAIN_CONTEXT: 0}
        default_method_id = context_method_map.get(MAIN_CONTEXT, 0)
        for site in call_sites:
            caller_context = site.instruction.continuation_id or MAIN_CONTEXT
            caller_method_id = context_method_map.get(caller_context, default_method_id)
            caller_label = (
                f"method:{caller_method_id}"
                if caller_method_id is not None
                else caller_id
            )
            callee = self._format_callee(site)
            edges.append(
                CallGraphEdge(
                    caller=caller_label,
                    callee=callee,
                    target_type=site.target_type,
                    instruction_index=site.instruction.index,
                )
            )
        return edges

    def _build_context_method_map(
        self, continuations: Dict[str, Continuation]
    ) -> Dict[str, int]:
        """Build context->method_id mapping for call graph attribution.

        Non-method continuations inherit the nearest ancestor method ID.
        Main context is mapped to method 0 by default.
        """
        context_method_map: Dict[str, int] = {MAIN_CONTEXT: 0}
        unresolved = set(continuations.keys())

        max_iters = len(continuations) + 1
        iters = 0
        while unresolved:
            iters += 1
            if iters > max_iters:
                logger.warning(
                    "_build_context_method_map did not converge after %d iterations",
                    max_iters,
                )
                break
            progressed = False
            for cont_id in list(unresolved):
                cont = continuations.get(cont_id)
                if cont is None:
                    unresolved.remove(cont_id)
                    progressed = True
                    continue

                if cont.kind == "method":
                    context_method_map[cont_id] = (
                        cont.method_id if cont.method_id is not None else 0
                    )
                    unresolved.remove(cont_id)
                    progressed = True
                    continue

                parent = cont.parent_context
                if parent in (None, MAIN_CONTEXT):
                    context_method_map[cont_id] = context_method_map[MAIN_CONTEXT]
                    unresolved.remove(cont_id)
                    progressed = True
                    continue

                if parent in context_method_map:
                    context_method_map[cont_id] = context_method_map[parent]
                    unresolved.remove(cont_id)
                    progressed = True

            if not progressed:
                # Fallback: attach remaining contexts to main method.
                for cont_id in list(unresolved):
                    context_method_map[cont_id] = context_method_map[MAIN_CONTEXT]
                    unresolved.remove(cont_id)

        return context_method_map

    def _format_callee(self, site: CallSite) -> str:
        """Format callee identifier."""
        if site.target_type == "method" and site.target is not None:
            return f"method:{site.target}"
        return "unknown"

    def _build_summary(
        self,
        context_id: str,
        hash_str: str,
        events: Sequence[Event],
        call_sites: Sequence[CallSite],
        instructions: Sequence[InstructionFact],
    ) -> BlockSummary:
        """Build block summary."""
        event_counts = Counter(event.type for event in events)
        call_targets = [
            self._format_callee(site)
            for site in call_sites
            if site.target is not None
        ]

        return BlockSummary(
            context_id=context_id,
            hash=hash_str,
            event_counts=dict(event_counts),
            call_targets=call_targets,
            register_reads={},
            register_writes={},
            caller_hash=None,
            caller_offset=None,
        )
