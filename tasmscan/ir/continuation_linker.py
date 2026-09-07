"""
Continuation Linker - Links continuation references and builds SaveLists

This module implements the linking pass that resolves continuation references
and constructs SaveLists for precise cross-continuation data flow tracking.

This is the CORE INNOVATION of TASIR:
- Addresses the cross-continuation limitation in path-sensitive analysis
- Enables precise taint tracking across continuation boundaries
- Models TVM's SAVE/SAVEALT/SAVEBOTH semantics

Academic contribution:
The SaveList abstraction captures the register-saving semantics of TVM
continuations, enabling the first precise inter-continuation data flow
analysis for the TON Virtual Machine.
"""

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

from ..analyzer.utils import extract_int_arg
from .tasir_types import (
    TVMModule,
    TVMInstruction,
    SaveList,
    InstructionKind,
    TVMAbstractValue,
    RegisterLocation,
)
from .stack_effects import get_stack_effect

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..solver.dynamic_target_solver import DynamicTargetSolver


@dataclass
class SavePoint:
    """A point where registers are saved to a continuation.

    Attributes:
        instruction_index: Index of the SAVE instruction
        saved_registers: List of register indices (c0, c1, etc.)
        target_continuation: ID of the continuation receiving the save
        target_candidates: Conservative candidate continuation IDs when target is ambiguous
    """
    instruction_index: int
    saved_registers: List[int] = field(default_factory=list)
    target_continuation: Optional[str] = None
    target_candidates: List[str] = field(default_factory=list)


@dataclass
class ContinuationReference:
    """A reference from one continuation to another.

    Used for tracking call/jump relationships between continuations.

    Attributes:
        source_cont: ID of the calling/jumping continuation
        target_cont: ID of the target continuation
        call_site: Instruction index of the call/jump
        is_call: True for calls, False for jumps
        nargs: Number of arguments passed (-1 for varargs)
    """
    source_cont: str
    target_cont: str
    call_site: int
    is_call: bool = True
    nargs: int = -1


class ContinuationLinker:
    """
    Links continuation references and builds SaveLists.

    The linking process:
    1. Find all SAVE/SAVEALT/SAVEBOTH instructions
    2. Track which registers are saved at each point
    3. Build SaveList for each ContinuationDescriptor
    4. Resolve dynamic continuation targets where possible

    This addresses the cross-continuation limitation documented in path_sensitive_dataflow.py:
    - Cross-continuation data flow is tracked via SaveLists
    - Register values at save points are propagated to continuation entry
    - Dynamic targets are conservatively marked for special handling
    """

    # Opcodes that save registers to continuations
    SAVE_OPCODES: Dict[str, Optional[List[int]]] = {
        "SAVE": [0],           # Save c0 to savelist
        "SAVEALT": [1],        # Save c1 to savelist
        "SAVEBOTH": [0, 1],    # Save both c0 and c1
        "SAVECTR": None,       # Save specified register (needs arg)
        "SAVEALTCTR": None,    # Save specified register to c1
        "SAVEBOTHCTR": None,   # Save specified register to both
        "POPSAVE": None,       # Pop and save old register value
        "SAMEALTSAVE": [1],    # Save c1 then set c1 := c0
    }

    # Opcodes that create continuations
    CONT_CREATE_OPCODES: Set[str] = {
        "PUSHCONT", "PUSHREFCONT", "PUSHREF",
        "BLESS", "BLESSARGS", "BLESSVARARGS",
    }

    # Invocation opcode families used for dynamic target modeling
    DICT_TARGET_OPCODES: Set[str] = {
        "CALL", "CALLDICT", "CALLDICT_LONG", "JMPDICT", "JMPDICT_LONG",
    }
    REF_TARGET_OPCODES: Set[str] = {
        "CALLREF", "JMPREF", "JMPREFDATA",
    }
    STACK_TARGET_OPCODES: Set[str] = {
        "EXECUTE",
        "BOOLEVAL",
        "CALLX", "CALLXARGS", "CALLXARGS_VAR", "CALLXVARARGS",
        "CALLCC", "CALLCCARGS", "CALLCCARGS_VAR", "CALLCCVARARGS",
        "JMP", "JMPX", "JMPXARGS", "JMPXVARARGS", "JMPXDATA", "JMP_JMPREF",
    }
    # Opcodes that call/jump to continuations
    CONT_INVOKE_OPCODES: Set[str] = (
        DICT_TARGET_OPCODES | REF_TARGET_OPCODES | STACK_TARGET_OPCODES
    )

    # Opcodes that are calls (vs jumps)
    CONT_CALL_OPCODES: Set[str] = {
        "CALL", "CALLDICT", "CALLDICT_LONG", "CALLREF",
        "EXECUTE", "BOOLEVAL",
        "CALLX", "CALLXARGS", "CALLXARGS_VAR", "CALLXVARARGS",
        "CALLCC", "CALLCCARGS", "CALLCCARGS_VAR", "CALLCCVARARGS",
    }

    DEFAULT_CONT_REF_SEARCH_WINDOW = 2
    DEFAULT_CONT_STACK_TRACKING_DEPTH = 8
    REGISTER_MERGE_STRATEGIES: Set[str] = {"merge", "drop_conflict"}

    def __init__(
        self,
        *,
        cont_ref_search_window: int = DEFAULT_CONT_REF_SEARCH_WINDOW,
        cont_stack_tracking_depth: int = DEFAULT_CONT_STACK_TRACKING_DEPTH,
        register_merge_strategy: str = "merge",
        max_savelist_candidates_per_register: int = 4,
        skip_savelists: bool = False,
    ) -> None:
        """Initialize the linker with empty state."""
        if cont_ref_search_window < 0:
            raise ValueError("cont_ref_search_window must be >= 0")
        if cont_stack_tracking_depth <= 0:
            raise ValueError("cont_stack_tracking_depth must be > 0")
        if register_merge_strategy not in self.REGISTER_MERGE_STRATEGIES:
            raise ValueError(
                "register_merge_strategy must be one of "
                f"{sorted(self.REGISTER_MERGE_STRATEGIES)}"
            )
        if max_savelist_candidates_per_register < 0:
            raise ValueError("max_savelist_candidates_per_register must be >= 0")
        self._save_points: Dict[int, SavePoint] = {}
        self._register_values: Dict[int, Dict[int, TVMAbstractValue]] = {}
        self._continuation_refs: List[ContinuationReference] = []
        self._dynamic_targets: Set[int] = set()  # Instruction indices with dynamic targets
        self._solver_results: List[Dict[str, Any]] = []
        self._solver_obligations: List[Dict[str, Any]] = []
        self._cont_ref_search_window = cont_ref_search_window
        self._cont_stack_tracking_depth = cont_stack_tracking_depth
        self._register_merge_strategy = register_merge_strategy
        self._skip_savelists = skip_savelists
        self._max_savelist_candidates_per_register = max_savelist_candidates_per_register
        self._reachable_instruction_indices: Set[int] = set()
        self._register_merge_conflicts = 0
        self._savelist_placeholder_count = 0
        self._savelist_widened_registers = 0
        self._continuation_stack_in: Dict[int, Tuple[Optional[FrozenSet[str]], ...]] = {}
        self._continuation_cfg_pred_map: Dict[int, List[int]] = {}
        self._continuation_cfg_context_map: Dict[int, str] = {}
        self._cont_stack_fixpoint_capped = False
        self._cont_ref_resolution_stats: Dict[str, int] = {
            "direct_hits": 0,
            "stack_def_use_hits": 0,
            "window_hits": 0,
            "cfg_backward_hits": 0,
            "ambiguous_overapprox_hits": 0,
            "unresolved": 0,
        }

    def link(self, module: TVMModule) -> TVMModule:
        """
        Main entry point: link all continuation references.

        Args:
            module: Raw TVMModule from IRLifter

        Returns:
            Linked TVMModule with populated SaveLists
        """
        logger.info("Starting continuation linking pass")

        self._save_points.clear()
        self._register_values.clear()
        self._continuation_refs.clear()
        self._dynamic_targets.clear()
        self._solver_results.clear()
        self._solver_obligations.clear()
        self._reachable_instruction_indices = self._compute_reachable_instruction_indices(module)
        self._register_merge_conflicts = 0
        self._savelist_placeholder_count = 0
        self._savelist_widened_registers = 0
        self._continuation_stack_in.clear()
        self._continuation_cfg_pred_map.clear()
        self._continuation_cfg_context_map.clear()
        self._cont_stack_fixpoint_capped = False
        self._cont_ref_resolution_stats = {
            "direct_hits": 0,
            "stack_def_use_hits": 0,
            "window_hits": 0,
            "cfg_backward_hits": 0,
            "ambiguous_overapprox_hits": 0,
            "unresolved": 0,
        }

        # Step 0: Build continuation stack reaching definitions for SAVE target
        # resolution. This replaces short-window-only heuristics for long-distance
        # continuation references in a CFG-aware way.
        self._build_continuation_stack_values(module)

        # Step 1: Find all save points
        self._find_save_points(module)
        logger.debug(f"Found {len(self._save_points)} save points")

        # Step 2: Build register value map through forward analysis
        self._build_register_values(module)
        logger.debug(f"Built register values for {len(self._register_values)} instructions")

        # Step 3: Build SaveLists for each continuation
        if not self._skip_savelists:
            self._build_savelists(module)
        else:
            logger.debug("Skipping savelist construction (skip_savelists=True)")

        # Step 4: Resolve dynamic continuation targets
        solver_stats = self._resolve_dynamic_targets(module)
        logger.debug(f"Found {len(self._dynamic_targets)} dynamic targets")

        # Step 5: Propagate SaveLists to call sites
        if not self._skip_savelists:
            self._propagate_savelists_to_calls(module)
        else:
            logger.debug("Skipping savelist propagation (skip_savelists=True)")

        # Step 6: Build continuation reference graph
        self._build_continuation_graph(module)

        # Update metadata
        module.analysis_metadata["linking_complete"] = True
        module.analysis_metadata["save_point_count"] = len(self._save_points)
        module.analysis_metadata["dynamic_target_count"] = len(self._dynamic_targets)
        module.analysis_metadata["dynamic_target_instructions"] = sorted(self._dynamic_targets)
        module.analysis_metadata["continuation_ref_count"] = len(self._continuation_refs)
        module.analysis_metadata["dynamic_target_solver_stats"] = dict(solver_stats)
        module.analysis_metadata["dynamic_target_solver_results"] = list(self._solver_results)
        module.analysis_metadata["dynamic_target_solver_obligations"] = list(self._solver_obligations)
        module.analysis_metadata["register_merge_strategy"] = self._register_merge_strategy
        module.analysis_metadata["register_merge_conflicts"] = self._register_merge_conflicts
        module.analysis_metadata["savelist_placeholder_count"] = self._savelist_placeholder_count
        module.analysis_metadata["savelist_widened_registers"] = self._savelist_widened_registers
        module.analysis_metadata["cont_ref_search_window"] = self._cont_ref_search_window
        module.analysis_metadata["cont_stack_tracking_depth"] = self._cont_stack_tracking_depth
        module.analysis_metadata["cont_stack_fixpoint_capped"] = self._cont_stack_fixpoint_capped
        module.analysis_metadata["cont_ref_resolution"] = dict(self._cont_ref_resolution_stats)
        module.analysis_metadata["max_savelist_candidates_per_register"] = (
            self._max_savelist_candidates_per_register
        )
        module.analysis_metadata["reachable_instruction_count"] = len(self._reachable_instruction_indices)
        if self._dynamic_targets:
            module.analysis_metadata["analysis_incomplete"] = True
            reasons = module.analysis_metadata.setdefault("analysis_incomplete_reasons", [])
            if "dynamic_continuation_targets" not in reasons:
                reasons.append("dynamic_continuation_targets")
        if self._cont_ref_resolution_stats.get("unresolved", 0) > 0:
            module.analysis_metadata["analysis_incomplete"] = True
            reasons = module.analysis_metadata.setdefault("analysis_incomplete_reasons", [])
            if "savelist_target_unresolved" not in reasons:
                reasons.append("savelist_target_unresolved")
        if self._cont_stack_fixpoint_capped:
            module.analysis_metadata["analysis_incomplete"] = True
            reasons = module.analysis_metadata.setdefault("analysis_incomplete_reasons", [])
            if "savelist_fixpoint_capped" not in reasons:
                reasons.append("savelist_fixpoint_capped")

        logger.info(
            f"Linking complete: {len(self._save_points)} save points, "
            f"{len(self._dynamic_targets)} dynamic targets, "
            f"{solver_stats.get('solved_count', 0)} solved by dynamic target solver"
        )

        return module

    def _find_save_points(self, module: TVMModule) -> None:
        """Find all SAVE/SAVEALT/SAVEBOTH instructions.

        Uses visited set to prevent processing the same instruction twice,
        which could happen if continuation references form cycles.
        """
        visited_insts: Set[int] = set()

        all_insts = module.all_instructions()
        inst_pos_map = {inst.index: pos for pos, inst in enumerate(all_insts)}
        inst_context_map = self._build_inst_context_map(module)

        for inst in all_insts:
            if self._reachable_instruction_indices and inst.index not in self._reachable_instruction_indices:
                continue
            # Cycle detection: skip if already processed
            if inst.index in visited_insts:
                logger.warning(
                    f"Cycle detected in _find_save_points: "
                    f"instruction {inst.index} already visited, skipping"
                )
                continue
            visited_insts.add(inst.index)

            upper = inst.opcode.upper()

            if upper in self.SAVE_OPCODES:
                saved_regs = self.SAVE_OPCODES[upper]

                # Handle parametric save (SAVECTR etc.)
                if saved_regs is None and inst.immediates:
                    first_imm = inst.immediates[0]
                    if isinstance(first_imm, int):
                        saved_regs = [first_imm]

                if saved_regs:
                    # Determine target continuation
                    target_cont, target_candidates = self._find_target_continuation(
                        inst=inst,
                        module=module,
                        all_insts=all_insts,
                        inst_pos_map=inst_pos_map,
                        inst_context_map=inst_context_map,
                    )

                    save_point = SavePoint(
                        instruction_index=inst.index,
                        saved_registers=list(saved_regs),
                        target_continuation=target_cont,
                        target_candidates=list(target_candidates),
                    )
                    self._save_points[inst.index] = save_point

                    logger.debug(
                        f"Found save point at {inst.index}: "
                        f"save c{saved_regs} to {target_cont or target_candidates}"
                    )

    def _find_target_continuation(
        self,
        inst: TVMInstruction,
        module: TVMModule,
        all_insts: Optional[List[TVMInstruction]] = None,
        inst_pos_map: Optional[Dict[int, int]] = None,
        inst_context_map: Optional[Dict[int, str]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """Find which continuation receives the saved registers."""
        direct_refs = [ref for ref in inst.continuation_refs if ref and not ref.startswith("__")]
        resolved_direct_refs = [
            ref for ref in direct_refs if self._is_resolved_cont_ref(ref, module)
        ]
        if len(direct_refs) == 1:
            self._cont_ref_resolution_stats["direct_hits"] += 1
            return direct_refs[0], []
        if len(resolved_direct_refs) == 1:
            self._cont_ref_resolution_stats["direct_hits"] += 1
            return resolved_direct_refs[0], []

        ambiguous_candidates = sorted(set(resolved_direct_refs)) if len(resolved_direct_refs) > 1 else []

        stack_target = self._resolve_target_from_continuation_stack(inst, module)
        if stack_target is not None:
            self._cont_ref_resolution_stats["stack_def_use_hits"] += 1
            return stack_target, []

        # Fallback: local window heuristic near SAVE when stack-def/use cannot
        # recover a unique continuation reference.
        if all_insts is None:
            all_insts = module.all_instructions()
        if inst_pos_map is None:
            inst_pos_map = {other.index: i for i, other in enumerate(all_insts)}
        if inst_context_map is None:
            inst_context_map = self._build_inst_context_map(module)

        window_target = self._resolve_target_from_neighbor_window(
            inst=inst,
            module=module,
            all_insts=all_insts,
            inst_pos_map=inst_pos_map,
            inst_context_map=inst_context_map,
        )
        if window_target is not None:
            self._cont_ref_resolution_stats["window_hits"] += 1
            return window_target, []

        cfg_target = self._resolve_target_from_cfg_backward_search(inst, module)
        if cfg_target is not None:
            self._cont_ref_resolution_stats["cfg_backward_hits"] += 1
            return cfg_target, []

        if ambiguous_candidates:
            # Conservative over-approximation: ambiguous explicit refs are retained
            # as candidate targets instead of dropping SAVE effect entirely.
            self._cont_ref_resolution_stats["ambiguous_overapprox_hits"] += 1
            return None, ambiguous_candidates

        self._cont_ref_resolution_stats["unresolved"] += 1
        return None, []

    def _resolve_target_from_continuation_stack(
        self,
        inst: TVMInstruction,
        module: TVMModule,
    ) -> Optional[str]:
        """Resolve SAVE target from CFG reaching continuation-stack definitions."""
        stack_state = self._continuation_stack_in.get(inst.index, tuple())
        if not stack_state:
            return None

        # Primary source is top-of-stack. Some SAVE patterns involve one transient
        # helper value, so also inspect depth-1 as a conservative fallback.
        max_depth = min(2, len(stack_state))
        for depth in range(max_depth):
            slot = stack_state[depth]
            if slot is None or not slot:
                continue
            resolved_refs = sorted(
                ref for ref in slot if self._is_resolved_cont_ref(ref, module)
            )
            if len(resolved_refs) == 1:
                return resolved_refs[0]
            if len(resolved_refs) > 1:
                return None
        return None

    def _resolve_target_from_neighbor_window(
        self,
        *,
        inst: TVMInstruction,
        module: TVMModule,
        all_insts: List[TVMInstruction],
        inst_pos_map: Dict[int, int],
        inst_context_map: Dict[int, str],
    ) -> Optional[str]:
        """Resolve SAVE target using local window heuristic around SAVE."""
        inst_idx_in_list = inst_pos_map.get(inst.index)
        if inst_idx_in_list is None:
            return None

        current_context = inst_context_map.get(inst.index)
        for distance in range(1, self._cont_ref_search_window + 1):
            # Prefer previous instruction first: SAVE often follows PUSHCONT.
            for offset in (-distance, distance):
                neighbor_pos = inst_idx_in_list + offset
                if not (0 <= neighbor_pos < len(all_insts)):
                    continue
                neighbor = all_insts[neighbor_pos]
                if inst_context_map.get(neighbor.index) != current_context:
                    continue
                if neighbor.opcode.upper() not in self.CONT_CREATE_OPCODES:
                    continue
                neighbor_refs = [
                    ref
                    for ref in neighbor.continuation_refs
                    if ref and not ref.startswith("__")
                ]
                resolved_neighbor_refs = [
                    ref for ref in neighbor_refs if self._is_resolved_cont_ref(ref, module)
                ]
                if len(neighbor_refs) == 1:
                    return neighbor_refs[0]
                if len(resolved_neighbor_refs) == 1:
                    return resolved_neighbor_refs[0]
        return None

    def _resolve_target_from_cfg_backward_search(
        self,
        inst: TVMInstruction,
        module: TVMModule,
    ) -> Optional[str]:
        """Resolve SAVE target by bounded backward CFG search in same context."""
        pred_map = self._continuation_cfg_pred_map
        context_map = self._continuation_cfg_context_map
        if not pred_map or inst.index not in context_map:
            return None

        source_context = context_map.get(inst.index)
        queue = deque([(inst.index, 0)])
        visited: Set[int] = {inst.index}
        found: Set[str] = set()
        max_depth = 16
        max_nodes = 192
        nodes_seen = 0

        # Build a quick instruction index map on-demand from register-values map
        # coverage plus module instruction list.
        inst_by_index = {i.index: i for i in module.all_instructions()}

        while queue and nodes_seen < max_nodes:
            current_idx, depth = queue.popleft()
            nodes_seen += 1
            if depth >= max_depth:
                continue
            for pred_idx in pred_map.get(current_idx, []):
                if pred_idx in visited:
                    continue
                visited.add(pred_idx)
                if context_map.get(pred_idx) != source_context:
                    continue
                pred_inst = inst_by_index.get(pred_idx)
                if pred_inst is None:
                    continue
                if pred_inst.opcode.upper() in self.CONT_CREATE_OPCODES:
                    refs = [
                        ref
                        for ref in pred_inst.continuation_refs
                        if isinstance(ref, str)
                        and ref
                        and not ref.startswith("__")
                        and self._is_resolved_cont_ref(ref, module)
                    ]
                    found.update(refs)
                    if len(found) > 1:
                        return None
                    if len(found) == 1:
                        # Keep exploring within depth limit to detect ambiguity;
                        # this preserves safety over early return.
                        pass
                queue.append((pred_idx, depth + 1))

        return next(iter(found)) if len(found) == 1 else None

    def _build_register_values(self, module: TVMModule) -> None:
        """Build a map of register values at each instruction.

        Performs a CFG-aware forward fixpoint analysis to track register contents.
        """
        self._register_values = {}

        inst_map, pred_map, succ_map = self._build_instruction_cfg(module)
        inst_map, pred_map, succ_map = self._restrict_cfg_to_reachable(
            inst_map=inst_map,
            pred_map=pred_map,
            succ_map=succ_map,
        )
        if not inst_map:
            return

        state_in: Dict[int, Dict[int, TVMAbstractValue]] = {}
        state_out: Dict[int, Dict[int, TVMAbstractValue]] = {}

        worklist = deque(sorted(inst_map.keys()))
        queued: Set[int] = set(worklist)
        max_iterations = max(64, len(inst_map) * 16)
        iterations = 0

        while worklist and iterations < max_iterations:
            iterations += 1
            idx = worklist.popleft()
            queued.discard(idx)

            predecessors = pred_map.get(idx, [])
            pred_states = [state_out[p] for p in predecessors if p in state_out]
            merged_in = (
                self._merge_register_states(pred_states) if pred_states else {}
            )

            prev_in = state_in.get(idx)
            in_changed = prev_in != merged_in
            if in_changed:
                state_in[idx] = merged_in

            new_out = self._transfer_register_state(inst_map[idx], merged_in)
            prev_out = state_out.get(idx)
            out_changed = prev_out != new_out
            if out_changed:
                state_out[idx] = new_out
                for succ_idx in succ_map.get(idx, []):
                    if succ_idx not in queued:
                        worklist.append(succ_idx)
                        queued.add(succ_idx)

        if worklist:
            logger.warning(
                "Register-value fixpoint reached iteration cap "
                f"({max_iterations}); remaining instructions may be conservative."
            )

        for idx in sorted(inst_map.keys()):
            self._register_values[idx] = dict(state_in.get(idx, {}))

    def _compute_reachable_instruction_indices(self, module: TVMModule) -> Set[int]:
        """Compute instruction reachability from function/global entry blocks."""
        blocks = {block.id: block for block in module.all_blocks()}
        if not blocks:
            return set()

        succ_by_block: Dict[int, Set[int]] = {block_id: set() for block_id in blocks}
        pred_count_by_block: Dict[int, int] = {block_id: 0 for block_id in blocks}
        for block in blocks.values():
            for edge in block.successors:
                dst = edge.target_block
                if dst not in blocks:
                    continue
                if dst not in succ_by_block[block.id]:
                    succ_by_block[block.id].add(dst)
                    pred_count_by_block[dst] = pred_count_by_block.get(dst, 0) + 1

        entry_blocks: Set[int] = set()
        for func in module.functions.values():
            if func.entry_block_id in blocks:
                entry_blocks.add(func.entry_block_id)
        # Include root global blocks with no predecessors as independent entry roots.
        for block in module.global_blocks:
            if block.id in blocks and pred_count_by_block.get(block.id, 0) == 0:
                entry_blocks.add(block.id)
        if not entry_blocks:
            entry_blocks = {block_id for block_id, pred_count in pred_count_by_block.items() if pred_count == 0}
        if not entry_blocks:
            entry_blocks = {min(blocks)}

        reachable_blocks: Set[int] = set()
        queue = deque(sorted(entry_blocks))
        while queue:
            block_id = queue.popleft()
            if block_id in reachable_blocks:
                continue
            reachable_blocks.add(block_id)
            for succ in sorted(succ_by_block.get(block_id, set())):
                if succ not in reachable_blocks:
                    queue.append(succ)

        reachable_indices: Set[int] = set()
        for block_id in reachable_blocks:
            block = blocks[block_id]
            for inst in block.instructions:
                reachable_indices.add(inst.index)
        return reachable_indices

    def _restrict_cfg_to_reachable(
        self,
        *,
        inst_map: Dict[int, TVMInstruction],
        pred_map: Dict[int, List[int]],
        succ_map: Dict[int, List[int]],
    ) -> Tuple[Dict[int, TVMInstruction], Dict[int, List[int]], Dict[int, List[int]]]:
        """Restrict instruction CFG maps to reachable instructions only."""
        if not self._reachable_instruction_indices:
            return inst_map, pred_map, succ_map

        reachable = set(self._reachable_instruction_indices)
        filtered_inst_map = {idx: inst for idx, inst in inst_map.items() if idx in reachable}
        filtered_pred_map: Dict[int, List[int]] = {}
        filtered_succ_map: Dict[int, List[int]] = {}
        for idx in filtered_inst_map:
            filtered_pred_map[idx] = [
                pred for pred in pred_map.get(idx, [])
                if pred in filtered_inst_map
            ]
            filtered_succ_map[idx] = [
                succ for succ in succ_map.get(idx, [])
                if succ in filtered_inst_map
            ]
        return filtered_inst_map, filtered_pred_map, filtered_succ_map

    def _build_instruction_cfg(
        self, module: TVMModule
    ) -> Tuple[
        Dict[int, TVMInstruction],
        Dict[int, List[int]],
        Dict[int, List[int]],
    ]:
        """Build instruction-level predecessor/successor maps from block CFG."""
        inst_map: Dict[int, TVMInstruction] = {}
        pred_map: Dict[int, Set[int]] = {}
        succ_map: Dict[int, Set[int]] = {}
        first_inst_by_block: Dict[int, int] = {}
        last_inst_by_block: Dict[int, int] = {}
        block_map = {block.id: block for block in module.all_blocks()}

        def add_edge(src_idx: int, dst_idx: int) -> None:
            succ_map.setdefault(src_idx, set()).add(dst_idx)
            pred_map.setdefault(dst_idx, set()).add(src_idx)

        for block in block_map.values():
            ordered = sorted(block.instructions, key=lambda inst: inst.index)
            if not ordered:
                continue
            first_inst_by_block[block.id] = ordered[0].index
            last_inst_by_block[block.id] = ordered[-1].index
            for inst in ordered:
                inst_map[inst.index] = inst
                pred_map.setdefault(inst.index, set())
                succ_map.setdefault(inst.index, set())
            for i in range(len(ordered) - 1):
                add_edge(ordered[i].index, ordered[i + 1].index)

        for block in block_map.values():
            src_last = last_inst_by_block.get(block.id)
            if src_last is None:
                continue
            for edge in block.successors:
                dst_first = first_inst_by_block.get(edge.target_block)
                if dst_first is not None:
                    add_edge(src_last, dst_first)

        pred_list_map = {idx: sorted(preds) for idx, preds in pred_map.items()}
        succ_list_map = {idx: sorted(succs) for idx, succs in succ_map.items()}
        return inst_map, pred_list_map, succ_list_map

    def _build_continuation_stack_values(self, module: TVMModule) -> None:
        """Build CFG-aware continuation stack reaching definitions per instruction."""
        self._continuation_stack_in = {}
        self._continuation_cfg_pred_map = {}
        self._continuation_cfg_context_map = {}

        inst_map, pred_map, succ_map = self._build_instruction_cfg(module)
        inst_map, pred_map, succ_map = self._restrict_cfg_to_reachable(
            inst_map=inst_map,
            pred_map=pred_map,
            succ_map=succ_map,
        )
        if not inst_map:
            return
        self._continuation_cfg_pred_map = dict(pred_map)
        full_context_map = self._build_inst_context_map(module)
        self._continuation_cfg_context_map = {
            idx: full_context_map[idx]
            for idx in inst_map.keys()
            if idx in full_context_map
        }

        state_in: Dict[int, Tuple[Optional[FrozenSet[str]], ...]] = {}
        state_out: Dict[int, Tuple[Optional[FrozenSet[str]], ...]] = {}
        worklist = deque(sorted(inst_map.keys()))
        queued: Set[int] = set(worklist)
        max_iterations = max(256, len(inst_map) * 32)
        iterations = 0

        while worklist and iterations < max_iterations:
            iterations += 1
            idx = worklist.popleft()
            queued.discard(idx)

            predecessors = pred_map.get(idx, [])
            pred_states = [state_out[p] for p in predecessors if p in state_out]
            merged_in = (
                self._merge_continuation_stack_states(pred_states)
                if pred_states
                else tuple()
            )

            if state_in.get(idx) != merged_in:
                state_in[idx] = merged_in

            new_out = self._transfer_continuation_stack_state(inst_map[idx], merged_in)
            if state_out.get(idx) != new_out:
                state_out[idx] = new_out
                for succ_idx in succ_map.get(idx, []):
                    if succ_idx not in queued:
                        worklist.append(succ_idx)
                        queued.add(succ_idx)

        self._cont_stack_fixpoint_capped = bool(worklist)
        if worklist:
            logger.warning(
                "Continuation-stack fixpoint reached iteration cap "
                f"({max_iterations}); SAVE target resolution may be conservative."
            )

        for idx in sorted(inst_map.keys()):
            self._continuation_stack_in[idx] = state_in.get(idx, tuple())

    def _merge_continuation_stack_states(
        self,
        states: List[Tuple[Optional[FrozenSet[str]], ...]],
    ) -> Tuple[Optional[FrozenSet[str]], ...]:
        """Merge predecessor continuation-stack summaries at a CFG join."""
        if not states:
            return tuple()

        max_depth = min(
            self._cont_stack_tracking_depth,
            max(len(state) for state in states),
        )
        merged: List[Optional[FrozenSet[str]]] = []

        for depth in range(max_depth):
            refs: Set[str] = set()
            has_unknown = False
            for state in states:
                if depth >= len(state):
                    has_unknown = True
                    continue
                slot = state[depth]
                if slot is None:
                    has_unknown = True
                    continue
                refs.update(slot)
            if has_unknown:
                merged.append(None)
            else:
                merged.append(frozenset(sorted(refs)))

        return self._compact_continuation_stack(merged)

    def _transfer_continuation_stack_state(
        self,
        inst: TVMInstruction,
        in_state: Tuple[Optional[FrozenSet[str]], ...],
    ) -> Tuple[Optional[FrozenSet[str]], ...]:
        """Apply one-instruction transfer for continuation-stack references."""
        state = list(in_state)
        upper = inst.opcode.upper()

        def pop_n(count: int) -> None:
            nonlocal state
            if count <= 0:
                return
            state = state[count:] if len(state) > count else []

        def push_slot(slot: Optional[FrozenSet[str]]) -> None:
            nonlocal state
            state.insert(0, slot)

        def ensure_depth(depth: int) -> None:
            while len(state) <= depth:
                state.append(None)

        def copy_from(depth: int) -> None:
            slot = state[depth] if depth < len(state) else None
            push_slot(slot)

        def swap(i: int, j: int) -> None:
            if i < 0 or j < 0:
                return
            ensure_depth(max(i, j))
            state[i], state[j] = state[j], state[i]

        if upper in self.CONT_CREATE_OPCODES:
            refs = sorted(
                {
                    ref
                    for ref in inst.continuation_refs
                    if isinstance(ref, str) and ref and not ref.startswith("__")
                }
            )
            cont_slot = frozenset(refs) if refs else None
            effect = get_stack_effect(upper)
            inputs = effect.inputs if effect else 0
            outputs = effect.outputs if effect and effect.outputs > 0 else 1
            pop_n(inputs)
            push_slot(cont_slot)
            for _ in range(outputs - 1):
                push_slot(frozenset())
            return self._compact_continuation_stack(state)

        if upper == "PUSH":
            copy_from(self._extract_int_operand(inst, 0, default=0))
            return self._compact_continuation_stack(state)
        if upper == "DUP":
            copy_from(0)
            return self._compact_continuation_stack(state)
        if upper == "OVER":
            copy_from(1)
            return self._compact_continuation_stack(state)
        if upper in {"POP", "DROP"}:
            pop_n(1)
            return self._compact_continuation_stack(state)
        if upper == "NIP":
            ensure_depth(1)
            if len(state) > 1:
                state.pop(1)
            return self._compact_continuation_stack(state)
        if upper in {"SWAP", "XCHG"}:
            swap(0, 1)
            return self._compact_continuation_stack(state)
        if upper == "ROT":
            ensure_depth(2)
            if len(state) > 2:
                state[0], state[1], state[2] = state[1], state[2], state[0]
            return self._compact_continuation_stack(state)
        if upper in {"ROTREV", "-ROT"}:
            ensure_depth(2)
            if len(state) > 2:
                state[0], state[1], state[2] = state[2], state[0], state[1]
            return self._compact_continuation_stack(state)
        if upper in {"XCHG_0I", "XCHG_0I_LONG"}:
            swap(0, self._extract_int_operand(inst, 0, default=0))
            return self._compact_continuation_stack(state)
        if upper == "XCHG_1I":
            swap(1, self._extract_int_operand(inst, 0, default=1))
            return self._compact_continuation_stack(state)
        if upper == "XCHG_IJ":
            i = self._extract_int_operand(inst, 0)
            j = self._extract_int_operand(inst, 1)
            if i is not None and j is not None:
                swap(i, j)
            return self._compact_continuation_stack(state)
        if upper == "XCHG2":
            i = self._extract_int_operand(inst, 0)
            j = self._extract_int_operand(inst, 1)
            if i is not None and j is not None:
                swap(1, i)
                swap(0, j)
            return self._compact_continuation_stack(state)
        if upper in {"XCHG3", "XCHG3_ALT"}:
            i = self._extract_int_operand(inst, 0)
            j = self._extract_int_operand(inst, 1)
            k = self._extract_int_operand(inst, 2)
            if i is not None and j is not None and k is not None:
                swap(2, i)
                swap(1, j)
                swap(0, k)
            return self._compact_continuation_stack(state)
        if upper == "PICK":
            pop_n(1)
            push_slot(None)
            return self._compact_continuation_stack(state)

        effect = get_stack_effect(upper)
        if effect is None:
            # Unknown stack semantics: preserve depth, but drop continuation identity.
            return self._compact_continuation_stack([None for _ in state])

        pop_n(effect.inputs)
        for _ in range(effect.outputs):
            push_slot(frozenset())
        return self._compact_continuation_stack(state)

    def _compact_continuation_stack(
        self,
        state: List[Optional[FrozenSet[str]]],
    ) -> Tuple[Optional[FrozenSet[str]], ...]:
        """Cap tracked depth and trim trailing known-non-continuation slots."""
        trimmed = list(state[: self._cont_stack_tracking_depth])
        while trimmed and trimmed[-1] == frozenset():
            trimmed.pop()
        return tuple(trimmed)

    @staticmethod
    def _extract_int_operand(
        inst: TVMInstruction,
        arg_idx: int,
        default: Optional[int] = None,
    ) -> Optional[int]:
        """Extract integer operand from original args/immediates with fallback."""
        from_args = extract_int_arg(getattr(inst, "original_args", None), arg_idx)
        if from_args is not None:
            return from_args
        immediates = getattr(inst, "immediates", []) or []
        if arg_idx < len(immediates) and isinstance(immediates[arg_idx], int):
            return immediates[arg_idx]
        return default

    def _merge_register_value_candidates(
        self,
        values: List[TVMAbstractValue],
    ) -> Optional[TVMAbstractValue]:
        """Merge incoming register abstract values at CFG join points."""
        if not values:
            return TVMAbstractValue(source="merge")

        unique = {
            (
                value.definition_site,
                value.source,
            )
            for value in values
        }
        if len(unique) == 1:
            value = values[0]
            return TVMAbstractValue(
                definition_site=value.definition_site,
                source=value.source,
                metadata=dict(value.metadata),
            )

        self._register_merge_conflicts += 1
        if self._register_merge_strategy == "drop_conflict":
            return None

        definition_sites = sorted(
            {
                value.definition_site
                for value in values
                if value.definition_site is not None
            }
        )
        merged_def_site = definition_sites[0] if len(definition_sites) == 1 else None
        metadata: Dict[str, Any] = {}
        if definition_sites:
            metadata["candidate_definition_sites"] = definition_sites
        return TVMAbstractValue(
            definition_site=merged_def_site,
            source="merge",
            metadata=metadata,
        )

    def _merge_register_states(
        self,
        states: List[Dict[int, TVMAbstractValue]],
    ) -> Dict[int, TVMAbstractValue]:
        """Merge predecessor register maps into a single conservative map."""
        if not states:
            return {}

        merged: Dict[int, TVMAbstractValue] = {}
        all_regs: Set[int] = set()
        for state in states:
            all_regs.update(state.keys())

        for reg_idx in sorted(all_regs):
            candidates = [state[reg_idx] for state in states if reg_idx in state]
            if not candidates:
                continue
            merged_value = self._merge_register_value_candidates(candidates)
            if merged_value is None:
                continue
            merged[reg_idx] = merged_value
        return merged

    def _transfer_register_state(
        self,
        inst: TVMInstruction,
        in_state: Dict[int, TVMAbstractValue],
    ) -> Dict[int, TVMAbstractValue]:
        """Apply register transfer function for a single instruction."""
        current_regs = dict(in_state)
        upper = inst.opcode.upper()

        # POPCTR writes to register
        if upper == "POPCTR":
            reg_idx = self._get_register_index(inst)
            if reg_idx is not None:
                current_regs[reg_idx] = TVMAbstractValue(
                    definition_site=inst.index,
                    source="register_store",
                )

        # SETCONT/BLESS family creates continuations on stack; no direct register store.
        elif upper in {"SETCONT", "SETCONTARGS", "BLESS", "BLESSARGS", "BLESSVARARGS"}:
            pass

        # SETCONTCTR - store continuation to specific register
        elif upper == "SETCONTCTR":
            reg_idx = self._get_register_index(inst)
            if reg_idx is not None:
                current_regs[reg_idx] = TVMAbstractValue(
                    definition_site=inst.index,
                    source="continuation_store",
                )

        # POPSAVE - pop from stack and write to register while saving old value
        elif upper == "POPSAVE":
            reg_idx = self._get_register_index(inst)
            if reg_idx is not None:
                current_regs[reg_idx] = TVMAbstractValue(
                    definition_site=inst.index,
                    source="register_store",
                )

        # SAMEALT - copy c0 to c1
        elif upper == "SAMEALT":
            if 0 in current_regs:
                current_regs[1] = TVMAbstractValue(
                    definition_site=inst.index,
                    source="register_copy_c0_to_c1",
                )

        # SAMEALTSAVE - save c1 into c0's savelist, then set c1 := c0
        elif upper == "SAMEALTSAVE":
            if 0 in current_regs:
                current_regs[1] = TVMAbstractValue(
                    definition_site=inst.index,
                    source="register_copy_c0_to_c1",
                )

        # SETRETCTR / SETALTCTR - specialized register writes
        elif upper in {"SETRETCTR", "SETALTCTR"}:
            target_reg = 0 if upper == "SETRETCTR" else 1
            current_regs[target_reg] = TVMAbstractValue(
                definition_site=inst.index,
                source="register_store",
            )

        # PUSHCTR and POPCTRX currently do not mutate statically tracked register map.
        return current_regs

    @staticmethod
    def _build_inst_context_map(module: TVMModule) -> Dict[int, str]:
        """Build instruction index -> block context map."""
        context_map: Dict[int, str] = {}
        for block in module.all_blocks():
            for inst in block.instructions:
                context_map[inst.index] = block.context_id
        return context_map

    def _get_register_index(self, inst: TVMInstruction) -> Optional[int]:
        """Extract register index from instruction."""
        for loc in inst.inputs + inst.outputs:
            if isinstance(loc, RegisterLocation):
                return loc.index
        if inst.immediates and isinstance(inst.immediates[0], int):
            return inst.immediates[0]
        return None

    def _build_savelists(self, module: TVMModule) -> None:
        """Build SaveList for each ContinuationDescriptor."""
        # Process cross-function continuations
        for cont_id, descriptor in module.cross_function_continuations.items():
            savelist = self._build_savelist_for_continuation(cont_id, module)

            # Update the descriptor with the new savelist
            module.cross_function_continuations[cont_id] = descriptor.with_savelist(
                savelist
            )

            if savelist.has_saved_values():
                logger.debug(
                    f"Built SaveList for {cont_id}: {len(savelist.registers)} registers"
                )

        # Also update function-local continuations
        for func in module.functions.values():
            for cont_id, descriptor in func.continuations.items():
                savelist = self._build_savelist_for_continuation(cont_id, module)
                func.continuations[cont_id] = descriptor.with_savelist(savelist)

    def _build_savelist_for_continuation(
        self, cont_id: str, module: TVMModule
    ) -> SaveList:
        """Build SaveList for a specific continuation."""
        savelist = SaveList()
        widened_regs_seen: Set[int] = set()

        # Find all save points targeting this continuation
        for inst_idx, save_point in self._save_points.items():
            if (
                save_point.target_continuation == cont_id
                or cont_id in save_point.target_candidates
            ):
                # Get register values at save point
                reg_values = self._register_values.get(inst_idx, {})

                for reg_idx in save_point.saved_registers:
                    was_widened = self._is_savelist_reg_widened(savelist, reg_idx)
                    if reg_idx in reg_values:
                        savelist = savelist.save(
                            reg_idx,
                            reg_values[reg_idx],
                            max_candidates=self._max_savelist_candidates_per_register,
                        )
                    else:
                        if self._register_merge_strategy == "drop_conflict":
                            # Precision-oriented mode: if register value could not be
                            # established at save point, skip synthetic placeholder.
                            continue
                        # Create placeholder value
                        self._savelist_placeholder_count += 1
                        savelist = savelist.save(
                            reg_idx,
                            TVMAbstractValue(
                                definition_site=inst_idx,
                                source=f"saved_c{reg_idx}",
                            ),
                            max_candidates=self._max_savelist_candidates_per_register,
                        )
                    if (
                        not was_widened
                        and self._is_savelist_reg_widened(savelist, reg_idx)
                        and reg_idx not in widened_regs_seen
                    ):
                        widened_regs_seen.add(reg_idx)
                        self._savelist_widened_registers += 1

        return savelist

    @staticmethod
    def _is_savelist_reg_widened(savelist: SaveList, reg_idx: int) -> bool:
        """Check whether register's SaveList candidates are widened."""
        values = savelist.restore_all(reg_idx)
        return (
            len(values) == 1
            and bool(values[0].metadata.get("savelist_widened"))
        )

    def _resolve_dynamic_targets(self, module: TVMModule) -> Dict[str, int]:
        """Resolve dynamic continuation targets where possible."""
        # Lazy import to avoid solver<->ir package import cycle at module import time.
        from ..solver.dynamic_target_solver import DynamicTargetSolver

        all_insts = module.all_instructions()
        idx_to_inst = {inst.index: inst for inst in all_insts}
        inst_to_context, prev_inst_idx = self._build_instruction_context_and_prev_maps(module)
        uncertain_branches = module.analysis_metadata.get("uncertain_branches", {})
        solver = DynamicTargetSolver()

        for inst in all_insts:
            if self._reachable_instruction_indices and inst.index not in self._reachable_instruction_indices:
                continue
            upper = inst.opcode.upper()
            if upper not in self.CONT_INVOKE_OPCODES:
                continue

            has_resolved_ref = any(
                self._is_resolved_cont_ref(ref, module)
                for ref in inst.continuation_refs
            )
            if has_resolved_ref:
                continue

            context_id = inst_to_context.get(inst.index, "main")
            prev_inst = idx_to_inst.get(prev_inst_idx.get(inst.index))
            pre_prev_inst = idx_to_inst.get(prev_inst_idx.get(prev_inst.index)) if prev_inst else None
            reasons_for_inst = []
            if isinstance(uncertain_branches, dict):
                if inst.index in uncertain_branches:
                    raw_reasons = uncertain_branches.get(inst.index, [])
                else:
                    raw_reasons = uncertain_branches.get(str(inst.index), [])
                if isinstance(raw_reasons, list):
                    reasons_for_inst = list(raw_reasons)
            solve_result = solver.solve(
                inst=inst,
                module=module,
                context_id=context_id,
                prev_inst=prev_inst,
                pre_prev_inst=pre_prev_inst,
                uncertain_reasons=reasons_for_inst,
            )

            candidate_refs: List[str] = []
            for cont_ref in solve_result.candidate_continuations:
                if self._is_resolved_cont_ref(cont_ref, module):
                    if cont_ref not in inst.continuation_refs:
                        inst.continuation_refs.append(cont_ref)
                    candidate_refs.append(cont_ref)

            solved = any(self._is_resolved_cont_ref(ref, module) for ref in inst.continuation_refs)
            solver_reported_sat = solve_result.status == "sat"
            unsat_nonlocal_method = (
                solve_result.status == "unsat"
                and solve_result.reason == "method_id_not_in_module"
                and upper in {"CALL", "CALLDICT", "CALLDICT_LONG", "JMPDICT", "JMPDICT_LONG"}
            )
            # A method-id miss in local module map means there is no local
            # continuation target to add. Treat this as out-of-module dispatch
            # rather than unresolved local dynamic target.
            treat_as_resolved = (solver_reported_sat and solved) or unsat_nonlocal_method
            if not treat_as_resolved:
                marker = self._dynamic_marker_for_opcode(upper)
                if marker not in inst.continuation_refs:
                    inst.continuation_refs.append(marker)
                self._dynamic_targets.add(inst.index)
                self._solver_obligations.append(
                    {
                        "instruction_index": inst.index,
                        "opcode": inst.opcode,
                        "context_id": context_id,
                        "status": solve_result.status,
                        "strategy": solve_result.strategy,
                        "reason": solve_result.reason or "dynamic_target_not_solved",
                        "candidate_continuations": candidate_refs,
                    }
                )
                logger.debug(f"Dynamic target at {inst.index}: {upper}")

            result_status = "sat" if (solver_reported_sat and solved) else solve_result.status
            self._solver_results.append(
                {
                    "instruction_index": inst.index,
                    "opcode": inst.opcode,
                    "context_id": context_id,
                    "status": result_status,
                    "strategy": solve_result.strategy,
                    "reason": solve_result.reason,
                    "candidate_continuations": candidate_refs,
                    "from_cache": solve_result.from_cache,
                    "model": dict(solve_result.model),
                }
            )

        return solver.stats()

    def _build_instruction_context_and_prev_maps(
        self, module: TVMModule
    ) -> Tuple[Dict[int, str], Dict[int, int]]:
        """Build instruction->context and instruction->previous instruction maps."""
        inst_to_context: Dict[int, str] = {}
        by_context: Dict[str, List[int]] = {}
        for block in module.all_blocks():
            for inst in block.instructions:
                inst_to_context[inst.index] = block.context_id
                by_context.setdefault(block.context_id, []).append(inst.index)

        prev_inst_idx: Dict[int, int] = {}
        for context_id, indices in by_context.items():
            ordered = sorted(set(indices))
            for i in range(1, len(ordered)):
                prev_inst_idx[ordered[i]] = ordered[i - 1]

        return inst_to_context, prev_inst_idx

    def _is_resolved_cont_ref(self, cont_ref: str, module: TVMModule) -> bool:
        """Return True when a continuation reference resolves to a known descriptor."""
        if not cont_ref or cont_ref.startswith("__"):
            return False
        if cont_ref in module.cross_function_continuations:
            return True
        for func in module.functions.values():
            if cont_ref in func.continuations:
                return True
        return False

    def _dynamic_marker_for_opcode(self, opcode: str) -> str:
        """Return dynamic target marker by opcode family."""
        if opcode in self.REF_TARGET_OPCODES:
            return "__cellref__"
        if opcode in self.STACK_TARGET_OPCODES:
            return "__stack__"
        return "__dynamic__"

    def _propagate_savelists_to_calls(self, module: TVMModule) -> None:
        """Propagate SaveLists to continuation call sites."""
        for inst in module.all_instructions():
            if self._reachable_instruction_indices and inst.index not in self._reachable_instruction_indices:
                continue
            if inst.kind in {InstructionKind.CONT_CALL, InstructionKind.CONT_JUMP}:
                for cont_ref in inst.continuation_refs:
                    # are conservatively SKIPPED rather than over-approximated.
                    # Trade-off: No false positives from dynamic targets, but potential
                    # false negatives if taint flows through dynamically resolved
                    # continuations. This is documented as a known limitation.
                    # The paper's Theorem 1 explicitly excludes dynamic targets.
                    if cont_ref.startswith("__"):
                        continue

                    # Look up the continuation's SaveList
                    descriptor = module.cross_function_continuations.get(cont_ref)
                    if descriptor is None:
                        # Try function-local continuations
                        for func in module.functions.values():
                            descriptor = func.continuations.get(cont_ref)
                            if descriptor:
                                break

                    if descriptor and descriptor.savelist.has_saved_values():
                        # Store SaveList info in instruction metadata
                        # This enables the dataflow analyzer to use it
                        inst.original_args.append(
                            ("__savelist__", descriptor.savelist)
                        )
                        logger.debug(
                            f"Propagated SaveList to call at {inst.index}: "
                            f"{len(descriptor.savelist.registers)} registers"
                        )

    def _build_continuation_graph(self, module: TVMModule) -> None:
        """Build the continuation reference graph with cycle detection.

        Creates ContinuationReference entries for each call/jump between
        continuations, enabling inter-procedural analysis.

        Uses visited sets to detect and skip cycles in the continuation graph,
        preventing infinite loops when continuations reference each other.
        """
        visited_insts: Set[int] = set()
        visited_edges: Set[Tuple[str, str]] = set()  # (source, target) pairs

        inst_to_context: Dict[int, str] = {}
        for func in module.functions.values():
            for block in func.blocks.values():
                for inst in block.instructions:
                    inst_to_context[inst.index] = block.context_id
        for block in getattr(module, 'global_blocks', []):
            for inst in block.instructions:
                inst_to_context[inst.index] = block.context_id

        for inst in module.all_instructions():
            if self._reachable_instruction_indices and inst.index not in self._reachable_instruction_indices:
                continue
            # Cycle detection: skip if instruction already processed
            if inst.index in visited_insts:
                logger.warning(
                    f"Cycle detected in _build_continuation_graph: "
                    f"instruction {inst.index} already visited, skipping"
                )
                continue
            visited_insts.add(inst.index)

            upper = inst.opcode.upper()

            if upper in self.CONT_INVOKE_OPCODES:
                # Determine source continuation
                source_cont = self._find_containing_continuation(inst, module, inst_to_context)

                # Determine target continuation(s)
                for target in inst.continuation_refs:
                    # Detect edge cycles (same source->target pair)
                    edge = (source_cont or "main", target)
                    if edge in visited_edges:
                        logger.debug(
                            f"Cycle detected in continuation graph: "
                            f"{edge[0]} -> {edge[1]}, skipping duplicate edge"
                        )
                        continue
                    visited_edges.add(edge)

                    is_call = upper in self.CONT_CALL_OPCODES

                    # Extract nargs if available
                    nargs = -1
                    if inst.immediates and isinstance(inst.immediates[0], int):
                        nargs = inst.immediates[0]

                    ref = ContinuationReference(
                        source_cont=source_cont or "main",
                        target_cont=target,
                        call_site=inst.index,
                        is_call=is_call,
                        nargs=nargs,
                    )
                    self._continuation_refs.append(ref)

        # Store in module metadata for later analysis
        module.analysis_metadata["continuation_graph"] = [
            {
                "source": ref.source_cont,
                "target": ref.target_cont,
                "call_site": ref.call_site,
                "is_call": ref.is_call,
                "nargs": ref.nargs,
            }
            for ref in self._continuation_refs
        ]

    def _find_containing_continuation(
        self, inst: TVMInstruction, module: TVMModule, inst_to_context: Optional[Dict[int, str]] = None
    ) -> Optional[str]:
        """Find which continuation contains the given instruction."""
        if inst_to_context is not None:
            return inst_to_context.get(inst.index)

        # Check function blocks
        for func in module.functions.values():
            for block in func.blocks.values():
                for block_inst in block.instructions:
                    if block_inst.index == inst.index:
                        return block.context_id

        # Check global blocks
        for block in module.global_blocks:
            for block_inst in block.instructions:
                if block_inst.index == inst.index:
                    return block.context_id

        return None

    def get_savelist_at_call(
        self, inst_index: int, module: TVMModule
    ) -> Optional[SaveList]:
        """Get the SaveList that applies at a specific call site.

        Args:
            inst_index: Index of the call/jump instruction
            module: The TVMModule being analyzed

        Returns:
            SaveList if one applies, None otherwise
        """
        all_insts = module.all_instructions()
        for inst in all_insts:
            if inst.index == inst_index:
                for arg in inst.original_args:
                    if isinstance(arg, tuple) and arg[0] == "__savelist__":
                        return arg[1]
        return None

    def get_continuation_refs(self) -> List[ContinuationReference]:
        """Get all continuation references found during linking."""
        return list(self._continuation_refs)

    def get_dynamic_targets(self) -> Set[int]:
        """Get instruction indices with dynamic (unresolved) targets."""
        return set(self._dynamic_targets)

    def is_dynamic_target(self, inst_index: int) -> bool:
        """Check if an instruction has a dynamic target."""
        return inst_index in self._dynamic_targets


def link_continuations(module: TVMModule) -> TVMModule:
    """Convenience function to link continuations in a TVMModule.

    Args:
        module: Raw TVMModule from IRLifter

    Returns:
        Linked TVMModule with populated SaveLists and continuation graph
    """
    linker = ContinuationLinker()
    return linker.link(module)
