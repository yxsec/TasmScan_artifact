"""
Path-Sensitive Data Flow Analysis for TVM bytecode.

This module provides path-sensitive analysis that tracks data flow
along different execution paths, reducing false positives compared
to path-insensitive analysis.

Provides path-sensitive control-flow analysis.

Known limitations - cross-continuation taint propagation:
=================================================================
This path-sensitive analyzer uses a continuation-aware CFG and can traverse
resolved continuation calls/returns. It still has limitations:

1. Dynamic continuation targets (CALLREF/CALLDICT/JMPDICT, computed continuations)
   may be unresolved, so edges into the actual target body may be missing.

2. Variable-arity call/jump instructions (*VARARGS) have runtime-dependent
   argument counts. The analysis approximates argument passing and may over-taint.

3. Path State at Continuation Entry: The PathState reflects the caller's
   current stack/registers, not any earlier PUSHCONT state. This is correct
   for TVM call semantics but may lose historical taint context if continuations
   are reused across multiple call sites.

Continuation Taint Strategy:
- PUSHCONT: Continuation value pushed to stack is NOT tainted (it's code, not data)
- CALLX/EXECUTE: Uses caller state at call site; unresolved targets are conservative
- Control registers (c0-c3): Tracked for basic return address flow

Future improvements:
- More precise modeling of argument passing for *ARGS/*VARARGS calls
- Better resolution for dictionary-based continuation dispatch
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import hashlib
import heapq
import logging

logger = logging.getLogger("tasmscan.path_sensitive_dataflow")

from ..config import (
    MAX_PATHS as DEFAULT_MAX_PATHS,
    MAX_LOOP_UNROLL as DEFAULT_MAX_LOOP_UNROLL,
    MAX_ANALYSIS_DEPTH as DEFAULT_MAX_ANALYSIS_DEPTH,
    MAX_BLOCK_VISITS as DEFAULT_MAX_BLOCK_VISITS,
    MAX_WORKLIST_SIZE as DEFAULT_MAX_WORKLIST_SIZE,
    MIN_MATERIAL_TRUNCATION_EVENTS as DEFAULT_MIN_MATERIAL_TRUNCATION_EVENTS,
    validate_config_value,
    CONFIG_BOUNDS,
)
from .dataflow import (
    DataFlowValue,
    DataFlowState,
    DataFlowGraph,
    DataFlowAnalyzer,
)
from .dataflow.program_view import (
    ProgramBlock,
    ProgramInstruction,
    ProgramView,
    ensure_program_view,
)


@dataclass
class PathState:
    """Represents the data flow state along a specific execution path"""

    path_id: int  # Unique identifier for this path
    dataflow_state: DataFlowState  # Data flow state at this point
    path_condition: List[int] = field(default_factory=list)  # Branch instructions taken
    merged_from_paths: List[int] = field(default_factory=list)  # Path IDs merged into this state
    _fingerprint_cache: Optional[str] = field(default=None, init=False, repr=False)

    def copy(self) -> "PathState":
        """Create a deep copy of this path state"""
        return PathState(
            path_id=self.path_id,
            dataflow_state=self.dataflow_state.copy(),
            path_condition=self.path_condition.copy(),
            merged_from_paths=self.merged_from_paths.copy()
        )

    def fingerprint(self) -> str:
        """
        Generate a fingerprint for this path state for deduplication.

        The fingerprint captures the essential state information:
        - Stack size and taint/check status
        - Register state and taint/check status
        - Guarded values (full set, not just presence/absence)
        - Security context indicator (has_guards flag for explicit distinction)

        Design Decision: path_id is intentionally NOT included in the fingerprint.
        The fingerprint is used for state deduplication at merge points - two paths
        that arrive at the same program point with equivalent data flow state
        (same stack structure, taint patterns, and guarded values) can be merged
        regardless of which path_id they originated from. Including path_id would
        defeat this deduplication optimization.

        Security Context Consideration (context-sensitive handling):
        The fingerprint includes both:
        1. A "has_guards" flag to explicitly distinguish states where guarded_values
           is empty vs non-empty. This is a fast-path check.
        2. The full guarded_values set contents for precise differentiation.

        This prevents false path deduplication where two paths with identical
        stack/register state but different security contexts (one has guards,
        one doesn't) would incorrectly merge. Without this, a guarded path could
        be skipped because an unguarded path with the same data flow state was
        already visited, leading to missed vulnerability reports.

        Trade-off Analysis:
        - Including full guarded_values increases fingerprint precision
        - May reduce deduplication efficiency for paths that differ only in guards
        - This is acceptable because:
          a) guarded_values sets are typically small (< 10 elements)
          b) Security correctness outweighs performance optimization
          c) Missing a vulnerability is worse than redundant analysis

        Uses SHA-256 hashing over the coarse signature to keep deterministic
        deduplication keys while limiting per-state comparison overhead.

        Returns:
            String fingerprint (hex digest) suitable for hashing
        """
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache

        hasher = hashlib.sha256()

        # This ensures states with guards are never deduplicated with unguarded states
        has_guards = bool(self.dataflow_state.guarded_values)
        hasher.update(f"has_guards:{has_guards}|".encode())

        # Encode stack size
        hasher.update(f"stack_len:{len(self.dataflow_state.stack)}|".encode())

        # Capture stack structure: size + coarse taint/check pattern.
        # We intentionally avoid origin-level detail here to reduce state churn
        # in large cyclic CFGs; taint origin sets are still preserved in metadata.
        for i, val in enumerate(self.dataflow_state.stack):
            if val:
                if val.tainted:
                    hasher.update(f"{i}:T:{int(bool(val.checked))}|".encode())
                else:
                    hasher.update(f"{i}:C|".encode())
            else:
                hasher.update(f"{i}:_|".encode())

        # Capture registers state
        hasher.update(b"registers:")
        for reg_name in sorted(self.dataflow_state.registers.keys()):
            val = self.dataflow_state.registers[reg_name]
            if val:
                if val.tainted:
                    hasher.update(f"{reg_name}:T:{int(bool(val.checked))}|".encode())
                else:
                    hasher.update(f"{reg_name}:C|".encode())
            else:
                hasher.update(f"{reg_name}:_|".encode())

        # Capture guarded values with separator to prevent boundary collisions
        hasher.update(b"guarded:")
        for v in sorted(self.dataflow_state.guarded_values.keys()):
            hasher.update(f"{v},".encode())

        self._fingerprint_cache = hasher.hexdigest()
        return self._fingerprint_cache


class PathSensitiveDataFlowAnalyzer:
    """
    Path-sensitive data flow analyzer.

    Tracks data flow along different execution paths to reduce false positives.
    Uses CFG information to enumerate paths and merge states at join points.
    """

    # Class-level defaults (can be overridden via constructor)
    MAX_PATHS = DEFAULT_MAX_PATHS
    MAX_LOOP_UNROLL = DEFAULT_MAX_LOOP_UNROLL
    MAX_ANALYSIS_DEPTH = DEFAULT_MAX_ANALYSIS_DEPTH

    def __init__(
        self,
        max_paths: Optional[int] = None,
        max_loop_unroll: Optional[int] = None,
        max_analysis_depth: Optional[int] = None,
        max_block_visits: Optional[int] = None,
        max_worklist_size: Optional[int] = None,
    ):
        self.base_analyzer = DataFlowAnalyzer()
        self.path_states: Dict[int, List[PathState]] = defaultdict(list)  # inst_idx -> states
        self.merged_states: Dict[int, DataFlowState] = {}  # inst_idx -> merged state
        self.graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[], analysis_type="path_sensitive")
        # State deduplication: (block_id, state_fingerprint) -> result
        # Addresses Low #24: Path state deduplication for performance
        self.visited_states: Dict[Tuple[int, str], bool] = {}
        # Global block visit counter to prevent path explosion in complex CFGs
        self.block_visit_counts: Dict[int, int] = {}  # block_id -> visit count

        # Instance-level configurable parameters with validation
        self.max_paths = validate_config_value(
            "MAX_PATHS",
            max_paths if max_paths is not None else DEFAULT_MAX_PATHS,
            *CONFIG_BOUNDS["MAX_PATHS"]
        )
        self.max_loop_unroll = validate_config_value(
            "MAX_LOOP_UNROLL",
            max_loop_unroll if max_loop_unroll is not None else DEFAULT_MAX_LOOP_UNROLL,
            *CONFIG_BOUNDS["MAX_LOOP_UNROLL"]
        )
        self.max_analysis_depth = validate_config_value(
            "MAX_ANALYSIS_DEPTH",
            max_analysis_depth if max_analysis_depth is not None else DEFAULT_MAX_ANALYSIS_DEPTH,
            *CONFIG_BOUNDS["MAX_ANALYSIS_DEPTH"]
        )
        self.max_block_visits = validate_config_value(
            "MAX_BLOCK_VISITS",
            max_block_visits if max_block_visits is not None else DEFAULT_MAX_BLOCK_VISITS,
            *CONFIG_BOUNDS["MAX_BLOCK_VISITS"]
        )
        self.max_worklist_size = validate_config_value(
            "MAX_WORKLIST_SIZE",
            max_worklist_size if max_worklist_size is not None else DEFAULT_MAX_WORKLIST_SIZE,
            *CONFIG_BOUNDS["MAX_WORKLIST_SIZE"]
        )
        # Truncation tracking for completeness reporting
        self.truncated_paths: List[Tuple[int, int]] = []  # (block_id, depth)
        # Track loop paths truncated due to unroll limit: (block_id, succ_id, depth)
        # Preserves info about different taint states that couldn't be explored
        self.truncated_loop_paths: List[Tuple[int, int, int]] = []
        # Format: List[Dict] with keys: block_id, inst_idx, opcode, loop_back_edge
        self.truncated_sensitive_ops: List[Dict[str, Any]] = []
        # Resource-limit truncations that are not loop/depth specific.
        self.resource_truncation_count = 0
        self._truncated_path_seen: Set[Tuple[int, int]] = set()
        self.analysis_incomplete = False
        self._scc_by_block: Dict[int, int] = {}
        self._scc_members: Dict[int, Set[int]] = {}
        self._loop_blocks: Set[int] = set()
        self._loop_edge_summary: Dict[Tuple[int, int], PathState] = {}
        self._loop_edge_update_counts: Dict[Tuple[int, int], int] = {}
        self._loop_fixpoint_converged_edges: Set[Tuple[int, int]] = set()
        self._loop_widened_states = 0
        self._block_input_summary: Dict[int, PathState] = {}
        self._block_fixpoint_updates = 0
        # Resource guard for fixpoint updates on a single loop edge.
        self.max_scc_iterations = max(8, self.max_loop_unroll * 8)
        # Loop-summary widening cap: keeps fixpoint finite for stack-growing loops.
        self.loop_summary_stack_cap = max(4, self.max_loop_unroll * 4)
        # Global abstract-state caps to force convergence outside loops as well.
        self.global_summary_stack_cap = max(16, self.max_loop_unroll * 8)
        # Keep origin metadata bounded aggressively to stabilize fixpoint updates.
        self.max_taint_origins = 8
        self._effective_depth_limit = self.max_analysis_depth
        self.min_material_truncation_events = DEFAULT_MIN_MATERIAL_TRUNCATION_EVENTS
        self.instructions_by_idx: Dict[int, ProgramInstruction] = {}
        # Block map reference for _scan_sensitive_ops_in_block
        self._block_map: Dict[int, ProgramBlock] = {}
        # Static per-block priority component: count sensitive instructions once.
        self._block_base_priority: Dict[int, int] = {}
        # Reuse one analyzer instance per run to avoid per-instruction allocation churn.
        self._instruction_analyzer = DataFlowAnalyzer()
        # Cache loop-sensitive-op scans and avoid repeatedly handling saturated loop edges.
        self._loop_sensitive_scan_cache: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self._loop_over_budget_edges: Set[Tuple[int, int]] = set()
        self._truncated_loop_path_seen: Set[Tuple[int, int, int]] = set()
        self._truncated_sensitive_op_seen: Set[Tuple[int, int, str, int, int]] = set()
        # Frontier antichain per block for enqueue-time subsumption pruning.
        self._pending_block_frontier: Dict[int, List[PathState]] = defaultdict(list)
        # Subsumption counters for observability.
        self._subsumed_enqueue_skips = 0
        self._subsumed_block_skips = 0
        self._subsumed_loop_skips = 0
        self._semantic_branch_split_count = 0
        self._loop_edges_filtered_by_kind = 0
        self._edge_kind_histogram: Dict[str, int] = {}

    def _merge_comparison_operands(
        self, values: List[DataFlowValue]
    ) -> Optional[List[Any]]:
        """Merge comparison_operands from multiple DataFlowValue objects.

        Collects all unique comparison operands from all values and returns
        them as a merged list. This preserves guard checking information
        across path merges.

        Args:
            values: List of DataFlowValue objects to merge

        Returns:
            Merged list of comparison_operands, or None if no operands found
        """
        merged: List[Any] = []
        for v in values:
            comp_ops = v.metadata.get("comparison_operands")
            if comp_ops:
                for op in comp_ops:
                    if op not in merged:
                        merged.append(op)
        return merged if merged else None

    def analyze(self, program: Any) -> DataFlowGraph:
        """
        Perform path-sensitive data flow analysis.

        Args:
            program: AnalysisFacts, TASIR TVMModule, or ProgramView

        Returns:
            DataFlowGraph with path-sensitive results
        """
        # Check if we have CFG information for path-sensitive analysis
        view = ensure_program_view(program)

        if not view.basic_blocks or len(view.basic_blocks) == 0:
            # Fall back to path-insensitive analysis
            return self.base_analyzer.analyze(view)

        self.path_states.clear()
        self.merged_states.clear()
        self.visited_states.clear()
        self.block_visit_counts.clear()
        self.graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[], analysis_type="path_sensitive")
        # Reset truncation tracking
        self.truncated_paths.clear()
        self.truncated_loop_paths.clear()
        self.truncated_sensitive_ops.clear()
        self.resource_truncation_count = 0
        self._truncated_path_seen.clear()
        self.analysis_incomplete = False
        self._scc_by_block.clear()
        self._scc_members.clear()
        self._loop_blocks.clear()
        self._loop_edge_summary.clear()
        self._loop_edge_update_counts.clear()
        self._loop_fixpoint_converged_edges.clear()
        self._loop_widened_states = 0
        self._block_input_summary.clear()
        self._block_fixpoint_updates = 0
        self._effective_depth_limit = self.max_analysis_depth
        self._loop_sensitive_scan_cache.clear()
        self._loop_over_budget_edges.clear()
        self._truncated_loop_path_seen.clear()
        self._truncated_sensitive_op_seen.clear()
        self._pending_block_frontier.clear()
        self._subsumed_enqueue_skips = 0
        self._subsumed_block_skips = 0
        self._subsumed_loop_skips = 0
        self._semantic_branch_split_count = 0
        self._loop_edges_filtered_by_kind = 0
        self._edge_kind_histogram.clear()
        self._block_base_priority.clear()
        self.instructions_by_idx = {inst.index: inst for inst in view.instructions}
        self._block_base_priority = {
            block.id: sum(
                10
                for idx in block.instruction_indices
                if (inst := self.instructions_by_idx.get(idx)) and inst.opcode in {"SENDRAWMSG", "RAWRESERVE", "ACCEPT"}
            )
            for block in view.basic_blocks
        }

        # Perform path-sensitive analysis
        return self._analyze_with_paths(view)

    def _analyze_with_paths(self, view: ProgramView) -> DataFlowGraph:
        """Perform path-sensitive analysis using CFG"""

        # Build basic block map
        block_map = {block.id: block for block in view.basic_blocks}
        # Store reference for _scan_sensitive_ops_in_block
        self._block_map = block_map
        self._compute_sccs(block_map)
        # Adaptive depth budget: tiny fixed depth caps cause false truncation on
        # large linear/branching CFGs. We keep the configured minimum and scale by
        # CFG size to preserve coverage without disabling safeguards.
        self._effective_depth_limit = max(self.max_analysis_depth, len(block_map) * 2)

        entry_block_ids = view.entry_block_ids()
        if not entry_block_ids:
            entry_block_ids = [view.basic_blocks[0].id] if view.basic_blocks else []
        if not entry_block_ids:
            return self.graph

        path_id = 0
        for entry_id in entry_block_ids:
            entry_block = block_map.get(entry_id)
            if not entry_block:
                continue

            initial_state = PathState(
                path_id=path_id,
                dataflow_state=DataFlowState(stack=[], registers={}, guarded_values={}),
                path_condition=[]
            )
            path_id += 1

            self._analyze_block_paths(
                entry_block, [initial_state], view, block_map,
                visited_loops={}, path_blocks=set()
            )

        # Build final graph from path states
        self._build_graph_from_paths(view)

        unknown_successor_blocks = sorted(
            block.id for block in view.basic_blocks if getattr(block, "has_unknown_successor", False)
        )
        total_truncation_events = (
            len(self._truncated_path_seen)
            + len(self.truncated_loop_paths)
            + self.resource_truncation_count
        )
        material_truncation = (
            len(self.truncated_loop_paths) > 0
            or
            total_truncation_events > self.min_material_truncation_events
            or len(self.truncated_sensitive_ops) > 0
        )
        self.analysis_incomplete = bool(unknown_successor_blocks) or material_truncation

        analysis_reasons: List[str] = []
        if material_truncation:
            analysis_reasons.append("path_truncation")
        if unknown_successor_blocks:
            analysis_reasons.append("cfg_unknown_successor")

        # Expose analysis completeness information.
        # `truncated` is specific to path truncation (depth/loop/worklist limits),
        # while `analysis_incomplete` also includes unknown CFG successors.
        truncated = bool(
            self.truncated_paths
            or self.truncated_loop_paths
            or self.resource_truncation_count > 0
        )
        self.graph.analysis_metadata = {
            "analysis_incomplete": self.analysis_incomplete,
            "truncated": truncated,  # Explicit flag for easy access
            "truncated_paths": self.truncated_paths.copy(),
            "truncated_count": len(self.truncated_paths),
            "truncated_unique_count": len(self._truncated_path_seen),
            # Include loop truncation info for downstream analysis
            "truncated_loop_paths": self.truncated_loop_paths.copy(),
            "truncated_loop_count": len(self.truncated_loop_paths),
            "resource_truncation_count": self.resource_truncation_count,
            "total_truncation_events": total_truncation_events,
            "material_truncation": material_truncation,
            "truncation_materiality_threshold": self.min_material_truncation_events,
            # These require manual review as they may involve unchecked tainted data
            "truncated_sensitive_ops": self.truncated_sensitive_ops.copy(),
            "truncated_sensitive_ops_count": len(self.truncated_sensitive_ops),
            "unknown_successor_blocks": unknown_successor_blocks,
            "unknown_successor_count": len(unknown_successor_blocks),
            "loop_fixpoint_edge_count": len(self._loop_edge_update_counts),
            "loop_fixpoint_update_count": sum(self._loop_edge_update_counts.values()),
            "loop_fixpoint_converged_edges": sorted(self._loop_fixpoint_converged_edges),
            "loop_fixpoint_converged_count": len(self._loop_fixpoint_converged_edges),
            "loop_scc_count": len(self._scc_members),
            "loop_block_count": len(self._loop_blocks),
            "loop_widened_states": self._loop_widened_states,
            "block_fixpoint_updates": self._block_fixpoint_updates,
            "effective_depth_limit": self._effective_depth_limit,
            "subsumed_enqueue_skips": self._subsumed_enqueue_skips,
            "subsumed_block_skips": self._subsumed_block_skips,
            "subsumed_loop_skips": self._subsumed_loop_skips,
            "semantic_branch_split_count": self._semantic_branch_split_count,
            "loop_edges_filtered_by_kind": self._loop_edges_filtered_by_kind,
            "successor_edge_kind_histogram": dict(sorted(self._edge_kind_histogram.items())),
            "exception_edge_count": int(self._edge_kind_histogram.get("guard_throw", 0)),
            "return_edge_count": int(
                self._edge_kind_histogram.get("return_cont", 0)
                + self._edge_kind_histogram.get("call_return", 0)
                + self._edge_kind_histogram.get("guard_return", 0)
            ),
        }
        if analysis_reasons:
            self.graph.analysis_metadata["analysis_incomplete_reasons"] = analysis_reasons

        # Log analysis truncation for debugging
        if self.analysis_incomplete:
            total_truncated = total_truncation_events
            sensitive_count = len(self.truncated_sensitive_ops)
            if material_truncation and total_truncated > 0:
                logger.warning(
                    f"Path analysis truncated: {total_truncated} paths truncated "
                    f"(depth: {len(self.truncated_paths)}, loops: {len(self.truncated_loop_paths)})"
                )
            elif unknown_successor_blocks:
                logger.warning(
                    "Path analysis incomplete: %d CFG blocks have unknown successors",
                    len(unknown_successor_blocks),
                )
            else:
                logger.warning("Path analysis incomplete")
            if sensitive_count > 0:
                logger.warning(
                    f"Found {sensitive_count} sensitive operations in truncated loop paths "
                    f"that require manual review"
                )

        return self.graph

    # These operations can have security implications if executed with unchecked data
    LOOP_SENSITIVE_OPCODES = frozenset({
        # Message sending - may transfer funds with unchecked parameters
        "SENDRAWMSG", "SENDMSG",
        # Gas operations - may accept external messages without proper validation
        "ACCEPT", "SETGASLIMIT",
        # Fund operations - may reserve funds with unchecked amounts
        "RAWRESERVE", "RAWRESERVEX",
    })

    def _scan_sensitive_ops_in_block(
        self,
        block_id: int,
        loop_back_edge: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Scan a block and its reachable successors for sensitive operations.

        Loop truncation handling: When a loop is truncated due to max_loop_unroll limit, we need to
        identify any sensitive operations within the loop body that may be executed
        with unchecked tainted data in iterations beyond our analysis limit.

        Args:
            block_id: Starting block ID to scan
            loop_back_edge: The back-edge (source_block, target_block) that was truncated

        Returns:
            List of dicts with keys: block_id, inst_idx, opcode, loop_back_edge
        """
        cache_key = (loop_back_edge[0], loop_back_edge[1])
        cached = self._loop_sensitive_scan_cache.get(cache_key)
        if cached is not None:
            return cached

        sensitive_ops: List[Dict[str, Any]] = []
        visited: Set[int] = set()
        to_visit = deque([block_id])

        # Limit scan depth to avoid excessive work for large loops
        max_scan_blocks = 50
        blocks_scanned = 0

        while to_visit and blocks_scanned < max_scan_blocks:
            current_id = to_visit.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            blocks_scanned += 1

            block = self._block_map.get(current_id)
            if block is None:
                continue

            # Scan instructions in this block for sensitive operations
            for inst_idx in block.instruction_indices:
                inst = self.instructions_by_idx.get(inst_idx)
                if inst and inst.opcode in self.LOOP_SENSITIVE_OPCODES:
                    sensitive_ops.append({
                        "block_id": current_id,
                        "inst_idx": inst_idx,
                        "opcode": inst.opcode,
                        "loop_back_edge": loop_back_edge,
                        "needs_manual_review": True,
                    })

            # Continue scanning successors within the loop
            # Stop at the back-edge target to avoid infinite expansion
            if hasattr(block, 'successors') and block.successors:
                for succ_id in block.successors:
                    # Don't cross the back-edge again
                    if succ_id != loop_back_edge[1] and succ_id not in visited:
                        to_visit.append(succ_id)

        self._loop_sensitive_scan_cache[cache_key] = sensitive_ops
        return sensitive_ops

    def _compute_sccs(self, block_map: Dict[int, ProgramBlock]) -> None:
        """Compute SCC index for CFG blocks (iterative Kosaraju)."""
        self._scc_by_block.clear()
        self._scc_members.clear()
        self._loop_blocks.clear()

        if not block_map:
            return

        succ_map: Dict[int, List[int]] = {
            bid: [sid for sid in getattr(block, "successors", []) if sid in block_map]
            for bid, block in block_map.items()
        }
        rev_map: Dict[int, List[int]] = {bid: [] for bid in block_map}
        for src, succs in succ_map.items():
            for dst in succs:
                rev_map[dst].append(src)

        order: List[int] = []
        visited: Set[int] = set()

        for start in block_map:
            if start in visited:
                continue
            stack: List[Tuple[int, int]] = [(start, 0)]
            visited.add(start)
            while stack:
                node, nxt_idx = stack[-1]
                neighbors = succ_map.get(node, [])
                if nxt_idx < len(neighbors):
                    nxt = neighbors[nxt_idx]
                    stack[-1] = (node, nxt_idx + 1)
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append((nxt, 0))
                else:
                    order.append(node)
                    stack.pop()

        assigned: Set[int] = set()
        scc_id = 0
        for start in reversed(order):
            if start in assigned:
                continue
            members: Set[int] = set()
            stack = [start]
            assigned.add(start)
            while stack:
                node = stack.pop()
                members.add(node)
                for prev in rev_map.get(node, []):
                    if prev not in assigned:
                        assigned.add(prev)
                        stack.append(prev)

            self._scc_members[scc_id] = members
            for bid in members:
                self._scc_by_block[bid] = scc_id
            scc_id += 1

        for sid, members in self._scc_members.items():
            if len(members) > 1:
                self._loop_blocks.update(members)
                continue
            only = next(iter(members))
            if only in succ_map and only in succ_map[only]:
                self._loop_blocks.add(only)

    def _is_loop_edge(self, src_id: int, dst_id: int) -> bool:
        """Return True when an edge is inside a loop SCC."""
        src_scc = self._scc_by_block.get(src_id)
        dst_scc = self._scc_by_block.get(dst_id)
        if src_scc is None or dst_scc is None or src_scc != dst_scc:
            return False
        if src_id == dst_id:
            return src_id in self._loop_blocks
        return src_id in self._loop_blocks or dst_id in self._loop_blocks

    def _effective_block_visit_limit(self, block_id: int) -> int:
        """Return per-block visit limit with adaptive relaxation."""
        if block_id not in self._loop_blocks:
            # Non-loop revisits are mostly due divergent incoming summaries.
            # Allow additional iterations for block-level fixpoint convergence.
            return max(self.max_block_visits, 64)
        # Keep loop head/body visit budget tight to avoid SCC-wide blowups.
        # We only need a few extra visits for summary convergence.
        extra_visits = max(2, self.max_loop_unroll * 2)
        return self.max_block_visits + extra_visits

    def _merge_values_conservative(
        self,
        values: List[DataFlowValue],
    ) -> DataFlowValue:
        """Merge values with conservative taint/check semantics."""
        template = values[0]
        is_tainted = any(v.tainted for v in values)
        is_checked = False
        metadata: Dict[str, Any] = {}

        if is_tainted:
            tainted_values = [v for v in values if v.tainted]
            if tainted_values:
                is_checked = all(v.checked for v in tainted_values)
            sources, origins = self.base_analyzer._merge_taint_metadata(values)
            if sources:
                metadata["taint_sources"] = list(sorted(sources))
            if origins:
                metadata["taint_origins"] = self._cap_taint_origins(list(origins))

        merged_ops = self._merge_comparison_operands(values)
        if merged_ops:
            metadata["comparison_operands"] = merged_ops

        return DataFlowValue(
            source=template.source,
            definition_site=template.definition_site,
            tainted=is_tainted,
            checked=is_checked,
            metadata=metadata,
        )

    def _cap_taint_origins(self, origins: List[Any]) -> List[Any]:
        """Cap taint-origin cardinality to stabilize fixpoint convergence."""
        if not origins:
            return []
        # Fast path: list/tuple inputs are usually already deterministic from
        # previous widening; avoid repeated sort work on hot paths.
        iterable = origins
        if isinstance(origins, (set, frozenset)):
            iterable = sorted(origins)
        unique: List[Any] = []
        seen = set()
        for item in iterable:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
            if len(unique) >= self.max_taint_origins:
                break
        return unique

    def _widen_value_metadata(self, value: Optional[DataFlowValue]) -> None:
        """Apply lightweight metadata widening in-place."""
        if value is None:
            return
        metadata = value.metadata if isinstance(value.metadata, dict) else {}
        origins = metadata.get("taint_origins")
        if isinstance(origins, (list, tuple, set, frozenset)):
            capped = self._cap_taint_origins(list(origins))
            if len(capped) < len(list(origins)):
                metadata["taint_origins_widened"] = True
            metadata["taint_origins"] = capped
        sources = metadata.get("taint_sources")
        if isinstance(sources, (list, tuple, set, frozenset)):
            # Source domain is small; keep deterministic ordering.
            metadata["taint_sources"] = list(sorted(set(sources)))
        value.metadata = metadata

    def _widen_path_state(
        self,
        state: PathState,
        *,
        stack_cap: Optional[int] = None,
    ) -> PathState:
        """Widen state for block-level fixpoint convergence."""
        cap = stack_cap if stack_cap is not None else self.global_summary_stack_cap
        cap = max(1, int(cap))
        widened = state.copy()

        stack = widened.dataflow_state.stack
        for value in stack:
            self._widen_value_metadata(value)
        for value in widened.dataflow_state.registers.values():
            self._widen_value_metadata(value)

        if len(stack) > cap:
            tail_values = [v for v in stack[cap - 1:] if v is not None]
            merged_tail = self._merge_values_conservative(tail_values) if tail_values else None
            if merged_tail is not None:
                merged_tail.metadata = copy.deepcopy(merged_tail.metadata)
                merged_tail.metadata["global_widened_tail"] = True
                merged_tail.metadata["global_widened_depth"] = len(stack) - (cap - 1)
            widened.dataflow_state.stack = stack[: cap - 1] + [merged_tail]

        return widened

    def _widen_loop_summary(self, state: PathState) -> PathState:
        """Apply loop-summary widening to force convergence on stack-growing loops."""
        cap = max(1, self.loop_summary_stack_cap)
        stack = state.dataflow_state.stack
        if len(stack) <= cap:
            return state

        widened = state.copy()
        widened_stack = widened.dataflow_state.stack
        tail_start = cap - 1
        tail_values = [v for v in widened_stack[tail_start:] if v is not None]

        merged_tail: Optional[DataFlowValue] = None
        if tail_values:
            merged_tail = self._merge_values_conservative(tail_values)
            merged_tail.metadata = copy.deepcopy(merged_tail.metadata)
            merged_tail.metadata["loop_widened_tail"] = True
            merged_tail.metadata["loop_widened_depth"] = len(widened_stack) - tail_start

        widened.dataflow_state.stack = widened_stack[:tail_start] + [merged_tail]
        self._loop_widened_states += 1
        return widened

    def _calculate_path_priority(self, block: ProgramBlock, state: PathState) -> int:
        """
        Calculate priority score for worklist items (worklist prioritization).

        Higher priority (lower value for min-heap) for paths with:
        - Sensitive operations (SENDRAWMSG, RAWRESERVE, ACCEPT)
        - Unguarded taint in top stack positions

        Returns negative value for use with min-heap (higher priority = lower number).
        """
        priority = self._block_base_priority.get(block.id, 0)
        # Increase priority for paths with unguarded taint in top 3 stack positions
        stack = state.dataflow_state.stack
        for v in stack[:3]:
            if v and v.tainted and not v.checked:
                priority += 5
        # Return negative for min-heap (higher priority = explored first)
        return -priority

    def _value_subsumes(
        self,
        lhs: Optional[DataFlowValue],
        rhs: Optional[DataFlowValue],
    ) -> bool:
        """
        Return True when lhs is at least as conservative as rhs.

        Conservative lattice ordering:
        - tainted=True subsumes tainted=False
        - checked=False subsumes checked=True for tainted values
        """
        if rhs is None:
            return True
        if lhs is None:
            return False

        if rhs.tainted and not lhs.tainted:
            return False
        if lhs.tainted and rhs.tainted and lhs.checked and not rhs.checked:
            return False
        return True

    def _state_subsumes(self, lhs: PathState, rhs: PathState) -> bool:
        """
        Return True when lhs abstractly covers rhs at the same program point.
        """
        lhs_state = lhs.dataflow_state
        rhs_state = rhs.dataflow_state

        if len(lhs_state.stack) != len(rhs_state.stack):
            return False
        for lval, rval in zip(lhs_state.stack, rhs_state.stack):
            if not self._value_subsumes(lval, rval):
                return False

        for reg_key, rval in rhs_state.registers.items():
            if not self._value_subsumes(lhs_state.registers.get(reg_key), rval):
                return False

        lhs_guarded = lhs_state.guarded_values
        rhs_guarded = rhs_state.guarded_values
        for def_site, lhs_count in lhs_guarded.items():
            rhs_count = rhs_guarded.get(def_site)
            if rhs_count is None or lhs_count > rhs_count:
                return False

        return True

    def _enqueue_if_not_subsumed(
        self,
        worklist: List[Tuple[int, int, ProgramBlock, List[PathState], Dict[Tuple[int, int], int], int, Set[int]]],
        *,
        priority: int,
        counter: int,
        block: ProgramBlock,
        state: PathState,
        loops: Dict[Tuple[int, int], int],
        depth: int,
        path_blocks: Set[int],
    ) -> bool:
        """
        Enqueue successor state if it is not subsumed by pending frontier states.
        """
        frontier = self._pending_block_frontier[block.id]
        for pending in frontier:
            if self._state_subsumes(pending, state):
                self._subsumed_enqueue_skips += 1
                return False

        new_frontier: List[PathState] = []
        for pending in frontier:
            if self._state_subsumes(state, pending):
                continue
            new_frontier.append(pending)
        new_frontier.append(state)
        self._pending_block_frontier[block.id] = new_frontier

        heapq.heappush(
            worklist,
            (
                priority,
                counter,
                block,
                [state],
                loops,
                depth,
                path_blocks,
            ),
        )
        return True

    def _analyze_block_paths(
        self,
        block: ProgramBlock,
        incoming_states: List[PathState],
        view: ProgramView,
        block_map: Dict[int, ProgramBlock],
        visited_loops: Dict[Tuple[int, int], int],
        path_blocks: Set[int],
        depth: int = 0
    ):
        """
        Analyze basic blocks using iterative worklist algorithm with priority queue.

        This replaces the recursive approach to prevent path explosion.
        Uses a global visited_blocks set for cycle detection instead of
        per-path path_blocks tracking, which caused exponential blowup
        when the same block was visited via different paths.

        worklist prioritization: Uses priority queue to prioritize vulnerability-relevant paths
        (paths with sensitive operations or unguarded taint).

        Args:
            block: Starting basic block to analyze
            incoming_states: Path states entering this block
            facts: Analysis facts
            block_map: Map from block ID to block
            visited_loops: Track loop iterations (passed for API compat, managed internally)
            path_blocks: Initial path blocks (passed for API compat, used for initial visited set)
            depth: Initial depth (passed for API compat)
        """
        # WorkItem: (priority, counter, block, incoming_states, visited_loops, depth, path_blocks)
        # visited_loops tracks per-edge loop iterations for controlled unrolling
        # counter is used to break ties and ensure stable ordering
        counter = 0
        initial_priority = self._calculate_path_priority(block, incoming_states[0]) if incoming_states else 0
        initial_path_blocks = set(path_blocks)
        initial_path_blocks.add(block.id)
        worklist: List[Tuple[int, int, ProgramBlock, List[PathState], Dict[Tuple[int, int], int], int, Set[int]]] = [
            (initial_priority, counter, block, incoming_states, visited_loops.copy(), depth, initial_path_blocks)
        ]
        self._pending_block_frontier[block.id] = [self._widen_path_state(self._merge_path_states(incoming_states))]
        heapq.heapify(worklist)

        while worklist:
            if len(worklist) > self.max_worklist_size:
                logger.warning(
                    f"Worklist size {len(worklist)} exceeds limit {self.max_worklist_size}, "
                    f"truncating analysis"
                )
                self.resource_truncation_count += 1
                self.analysis_incomplete = True
                break

            _, _, current_block, current_incoming, current_loops, current_depth, current_path_blocks = heapq.heappop(worklist)
            current_path_blocks = set(current_path_blocks)
            current_path_blocks.add(current_block.id)
            current_summary = self._widen_path_state(self._merge_path_states(current_incoming))
            frontier = self._pending_block_frontier.get(current_block.id)
            if frontier:
                self._pending_block_frontier[current_block.id] = [
                    state for state in frontier if state.fingerprint() != current_summary.fingerprint()
                ]

            # Prevent excessive depth - record truncation
            if current_depth > self._effective_depth_limit:
                hit = (current_block.id, current_depth)
                if hit not in self._truncated_path_seen:
                    self._truncated_path_seen.add(hit)
                    self.truncated_paths.append(hit)
                self.analysis_incomplete = True
                continue

            # Block-level fixpoint summary: only reprocess a block when incoming
            # abstract state grows. This prevents path-enumeration churn where
            # many distinct path IDs carry equivalent dataflow information.
            incoming_summary = current_summary
            previous_summary = self._block_input_summary.get(current_block.id)
            if previous_summary is None:
                self._block_input_summary[current_block.id] = incoming_summary
                self._block_fixpoint_updates += 1
                current_incoming = [incoming_summary]
            else:
                if self._state_subsumes(previous_summary, incoming_summary):
                    self._subsumed_block_skips += 1
                    continue
                if self._state_subsumes(incoming_summary, previous_summary):
                    self._block_input_summary[current_block.id] = incoming_summary
                    self._block_fixpoint_updates += 1
                    current_incoming = [incoming_summary]
                    previous_summary = None
                if previous_summary is None:
                    pass
                else:
                    merged_summary = self._widen_path_state(
                        self._merge_path_states([previous_summary, incoming_summary])
                    )
                    if merged_summary.fingerprint() == previous_summary.fingerprint():
                        self._subsumed_block_skips += 1
                        continue
                    self._block_input_summary[current_block.id] = merged_summary
                    self._block_fixpoint_updates += 1
                    current_incoming = [merged_summary]

            # Limit total visits per block across all paths (prevent path explosion)
            block_visits = self.block_visit_counts.get(current_block.id, 0)
            block_visit_limit = self._effective_block_visit_limit(current_block.id)
            if block_visits >= block_visit_limit:
                hit = (current_block.id, current_depth)
                if hit not in self._truncated_path_seen:
                    self._truncated_path_seen.add(hit)
                    self.truncated_paths.append(hit)
                self.analysis_incomplete = True
                continue
            self.block_visit_counts[current_block.id] = block_visits + 1

            # Limit number of paths to prevent explosion
            if len(current_incoming) > self.max_paths:
                # Merge paths when too many - mark as incomplete
                current_incoming = [self._merge_path_states(current_incoming)]
                self.resource_truncation_count += 1
                self.analysis_incomplete = True

            # Low #24: State deduplication to avoid redundant analysis
            # Filter out states that have already been analyzed for this block
            deduplicated_states = []
            for state in current_incoming:
                state_fp = state.fingerprint()
                key = (current_block.id, state_fp)
                if key not in self.visited_states:
                    self.visited_states[key] = True
                    deduplicated_states.append(state)

            # If all states were duplicates, skip this block
            if not deduplicated_states:
                continue

            current_incoming = deduplicated_states

            # Process each instruction in the block with each incoming state
            current_states = current_incoming

            for inst_idx in current_block.instruction_indices:
                # Use index lookup instead of list position (more robust)
                inst = self.instructions_by_idx.get(inst_idx)
                if inst is None:
                    continue

                # Analyze instruction for each path state
                new_states = []
                for state in current_states:
                    new_state = self._analyze_instruction_with_state(inst, state)
                    new_states.append(new_state)

                # Store path states for this instruction
                self.path_states[inst_idx].extend(new_states)

                current_states = new_states

            # Handle successors (branches)
            if hasattr(current_block, 'successors') and current_block.successors:
                semantic_successor_count = self._semantic_successor_count(current_block)
                for succ_id in current_block.successors:
                    if succ_id not in block_map:
                        continue
                    edge_kinds = self._edge_kinds_for_successor(current_block, succ_id)
                    for edge_kind in edge_kinds:
                        self._edge_kind_histogram[edge_kind] = (
                            self._edge_kind_histogram.get(edge_kind, 0) + 1
                        )

                    # Copy states for branching (only if multiple successors)
                    if semantic_successor_count > 1:
                        branch_states = [state.copy() for state in current_states]
                        self._semantic_branch_split_count += 1
                    else:
                        branch_states = current_states

                    # Treat only path-observed back-edges as loop edges.
                    # SCC membership alone marks cyclic regions, but forward
                    # edges inside SCCs should still advance depth normally.
                    raw_loop_edge = self._is_loop_edge(current_block.id, succ_id)
                    is_candidate_loop_edge = (
                        raw_loop_edge
                        and self._edge_kind_allows_loop_fixpoint(edge_kinds)
                    )
                    if raw_loop_edge and not is_candidate_loop_edge:
                        self._loop_edges_filtered_by_kind += 1
                    is_loop_edge = is_candidate_loop_edge and succ_id in current_path_blocks
                    if is_loop_edge:
                        loop_key = (current_block.id, succ_id)
                        if loop_key in self._loop_over_budget_edges:
                            self.analysis_incomplete = True
                            continue
                        incoming_summary = self._widen_loop_summary(
                            self._merge_path_states(branch_states)
                        )
                        previous_summary = self._loop_edge_summary.get(loop_key)

                        if previous_summary is None:
                            self._loop_edge_summary[loop_key] = incoming_summary
                            self._loop_edge_update_counts[loop_key] = 1
                            branch_states = [incoming_summary]
                        else:
                            if self._state_subsumes(previous_summary, incoming_summary):
                                self._subsumed_loop_skips += 1
                                self._loop_fixpoint_converged_edges.add(loop_key)
                                continue
                            if self._state_subsumes(incoming_summary, previous_summary):
                                widened = incoming_summary
                            else:
                                widened = self._widen_loop_summary(
                                    self._merge_path_states([previous_summary, incoming_summary])
                                )
                            if widened.fingerprint() == previous_summary.fingerprint():
                                self._subsumed_loop_skips += 1
                                self._loop_fixpoint_converged_edges.add(loop_key)
                                continue

                            updates = self._loop_edge_update_counts.get(loop_key, 1) + 1
                            if updates > self.max_scc_iterations:
                                self._loop_over_budget_edges.add(loop_key)
                                hit = (current_block.id, succ_id, current_depth)
                                if hit not in self._truncated_loop_path_seen:
                                    self._truncated_loop_path_seen.add(hit)
                                    self.truncated_loop_paths.append(hit)
                                self.analysis_incomplete = True
                                sensitive_ops = self._scan_sensitive_ops_in_block(
                                    succ_id, loop_key
                                )
                                if sensitive_ops:
                                    for op in sensitive_ops:
                                        dedup_key = (
                                            int(op.get("block_id", -1)),
                                            int(op.get("inst_idx", -1)),
                                            str(op.get("opcode", "")),
                                            loop_key[0],
                                            loop_key[1],
                                        )
                                        if dedup_key in self._truncated_sensitive_op_seen:
                                            continue
                                        self._truncated_sensitive_op_seen.add(dedup_key)
                                        self.truncated_sensitive_ops.append(op)
                                continue

                            self._loop_edge_update_counts[loop_key] = updates
                            self._loop_edge_summary[loop_key] = widened
                            self._loop_fixpoint_converged_edges.discard(loop_key)
                            branch_states = [widened]

                    new_loops = current_loops

                    # Track per-path visited blocks to avoid false cycle detection at merges
                    succ_path_blocks = set(current_path_blocks)
                    if not is_loop_edge:
                        succ_path_blocks.add(succ_id)

                    succ_block = block_map[succ_id]
                    succ_priority = self._calculate_path_priority(
                        succ_block, branch_states[0]
                    ) if branch_states else 0

                    # rapid growth when processing blocks with many successors
                    if len(worklist) >= self.max_worklist_size:
                        self.resource_truncation_count += 1
                        self.analysis_incomplete = True
                        continue  # Skip adding this successor

                    counter += 1
                    succ_state = self._widen_path_state(
                        self._merge_path_states(branch_states)
                    )
                    self._enqueue_if_not_subsumed(
                        worklist,
                        priority=succ_priority,
                        counter=counter,
                        block=succ_block,
                        state=succ_state,
                        loops=new_loops,
                        depth=current_depth if is_loop_edge else current_depth + 1,
                        path_blocks=succ_path_blocks,
                    )

    @staticmethod
    def _edge_kinds_for_successor(
        block: ProgramBlock,
        succ_id: int,
    ) -> List[str]:
        """Return normalized edge kinds for a successor block."""
        raw = getattr(block, "successor_edge_kinds", None)
        if isinstance(raw, dict):
            kinds = raw.get(succ_id, [])
            if isinstance(kinds, list):
                normalized = [str(kind) for kind in kinds if kind]
                if normalized:
                    return normalized
        return ["fallthrough"]

    def _semantic_successor_count(self, block: ProgramBlock) -> int:
        """Count successor alternatives including multi-kind transitions."""
        if not getattr(block, "successors", None):
            return 0
        total = 0
        for succ_id in block.successors:
            total += max(1, len(self._edge_kinds_for_successor(block, succ_id)))
        return total

    @staticmethod
    def _edge_kind_allows_loop_fixpoint(edge_kinds: List[str]) -> bool:
        """Decide whether edge kind should contribute to loop back-edge fixpoint."""
        procedural_only = {"call_cont", "return_cont"}
        return any(kind not in procedural_only for kind in edge_kinds)

    def _analyze_instruction_with_state(
        self,
        inst: ProgramInstruction,
        path_state: PathState,
    ) -> PathState:
        """
        Analyze a single instruction within a specific path state.

        Returns updated path state.

        Note: Reuses a dedicated per-analysis DataFlowAnalyzer instance while
        resetting mutable fields before each instruction to avoid state leakage.
        """
        opcode = inst.opcode
        idx = inst.index

        # Reuse a per-analysis instance while resetting mutable state each step.
        temp_analyzer = self._instruction_analyzer
        temp_analyzer.current_state = path_state.dataflow_state.copy()
        # previous use or default initialization that could leak across analyses
        temp_analyzer.graph = DataFlowGraph(values={}, edges=[], tainted_propagation=[])
        temp_analyzer._guard_access_counter = 0

        # Analyze instruction using the isolated analyzer
        temp_analyzer._analyze_instruction(inst)

        # Extract the updated state from the isolated analyzer
        updated_dataflow_state = temp_analyzer.current_state

        # Merge tainted propagation discovered during instruction analysis
        # (e.g., sensitive op consumes tainted input)
        for pair in temp_analyzer.graph.tainted_propagation:
            if pair not in self.graph._taint_dedup:
                self.graph._taint_dedup.add(pair)
                self.graph.tainted_propagation.append(pair)

        # Create new path state with updated dataflow state (avoid double copy)
        new_path_state = PathState(
            path_id=path_state.path_id,
            dataflow_state=updated_dataflow_state,  # Take ownership, no copy needed
            path_condition=path_state.path_condition.copy(),
        )

        # Track branch decisions
        if opcode in {"IF", "IFNOT", "IFJMP", "IFNOTJMP"}:
            new_path_state.path_condition.append(idx)

        # Merge any analysis metadata from temp analyzer
        self._merge_analysis_metadata(temp_analyzer.graph.analysis_metadata)

        return new_path_state

    def _merge_analysis_metadata(self, source_metadata: Dict) -> None:
        """Merge analysis metadata from source into self.graph.analysis_metadata."""
        if not source_metadata:
            return

        for key, value in source_metadata.items():
            if key not in self.graph.analysis_metadata:
                # New key - just set it (with copy for mutable types)
                if isinstance(value, list):
                    self.graph.analysis_metadata[key] = list(value)
                elif isinstance(value, set):
                    self.graph.analysis_metadata[key] = set(value)
                elif isinstance(value, dict):
                    self.graph.analysis_metadata[key] = dict(value)
                else:
                    self.graph.analysis_metadata[key] = value
            else:
                # Existing key - merge based on type
                existing = self.graph.analysis_metadata[key]
                if isinstance(existing, list) and isinstance(value, list):
                    existing.extend(value)
                elif isinstance(existing, set) and isinstance(value, set):
                    existing.update(value)
                elif isinstance(existing, dict) and isinstance(value, dict):
                    existing.update(value)
                elif isinstance(existing, bool) and isinstance(value, bool):
                    # For booleans like 'analysis_incomplete', use OR
                    self.graph.analysis_metadata[key] = existing or value
                elif isinstance(existing, (int, float)) and isinstance(value, (int, float)):
                    # For numeric counts, sum them
                    self.graph.analysis_metadata[key] = existing + value
                # else: keep existing value (first wins for non-mergeable types)

    def _merge_path_states(self, states: List[PathState]) -> PathState:
        """
        Merge multiple path states at a join point.

        Uses conservative merging:
        - Value is tainted if tainted on ANY path
        - Value is checked if checked on ALL paths where it's tainted

        Empty states handling:
            When states list is empty (no incoming paths), returns a fresh
            PathState with empty stack/registers. This represents unreachable
            code or analysis entry points where no prior state exists.
        """
        # Handle empty states: return fresh state for unreachable/entry points
        if not states:
            return PathState(
                path_id=0,
                dataflow_state=DataFlowState(stack=[], registers={}, guarded_values={}),
                merged_from_paths=[]
            )

        if len(states) == 1:
            return states[0]

        # Collect all path IDs being merged (including transitively merged paths)
        all_path_ids: List[int] = []
        for state in states:
            all_path_ids.append(state.path_id)
            all_path_ids.extend(state.merged_from_paths)
        # Deduplicate while preserving that we have this info
        merged_from_paths = list(set(all_path_ids))

        # Merge stack states conservatively
        # Use maximum stack size to preserve all taint information from all paths.
        #
        # When paths have different stack depths, positions beyond a shorter path's depth
        # contribute None to the merge. This represents "position never existed on this path"
        # rather than "position had unknown value". The distinction matters for taint analysis:
        # - None contributions don't affect taint status (only non-None values do)
        # - A position tainted on path A but non-existent on path B remains tainted
        # - This is conservative: we assume the longer-stack path's values could be reached
        #
        # Example: Path A has stack [a, b, c], Path B has stack [x, y]
        # Merged stack positions: [merge(a,x), merge(b,y), merge(c, None)]
        # Position 2 (c) is preserved because path A reached it, even though path B didn't.
        max_stack_size = max(len(s.dataflow_state.stack) for s in states)
        merged_stack = []

        for i in range(max_stack_size):
            # Collect values at position i, using None for paths with shorter stacks
            # None means "this position didn't exist on this path" (not "unknown value")
            values = [
                s.dataflow_state.stack[i] if i < len(s.dataflow_state.stack) else None
                for s in states
            ]

            if all(v is None for v in values):
                merged_stack.append(None)
                continue

            # Merge: tainted if ANY is tainted, checked if ALL tainted values are checked
            # Note: None values (missing positions) don't contribute to taint status
            non_none_values = [v for v in values if v is not None]
            is_tainted = any(v.tainted for v in non_none_values)
            is_checked = False

            if is_tainted:
                tainted_values = [v for v in non_none_values if v.tainted]
                if tainted_values:
                    is_checked = all(v.checked for v in tainted_values)

            # Use first non-None value as template
            template = next((v for v in values if v is not None), None)
            if template:
                metadata = {}
                if is_tainted:
                    # Filter out None values before merging taint metadata
                    sources, origins = self.base_analyzer._merge_taint_metadata(non_none_values)
                    if sources:
                        metadata["taint_sources"] = list(sources)
                    if origins:
                        metadata["taint_origins"] = list(origins)
                # This is critical for proper guard checking of comparison results
                merged_ops = self._merge_comparison_operands(non_none_values)
                if merged_ops:
                    metadata["comparison_operands"] = merged_ops
                merged_value = DataFlowValue(
                    source=template.source,
                    definition_site=template.definition_site,
                    tainted=is_tainted,
                    checked=is_checked,
                    metadata=copy.deepcopy(metadata),
                )
                merged_stack.append(merged_value)
            else:
                merged_stack.append(None)

        # Merge guarded values using UNION - a value is considered guarded if it was
        # checked on ANY path. This reduces false positives because:
        # 1. If path A checks value X and path B doesn't, then path A's usage of X is safe
        # 2. The taint analysis already tracks per-value checked status conservatively
        # 3. guarded_values serves as an additional safety net, not the primary check
        # Using intersection was too conservative and caused false positives when a value
        # was checked on some paths but not others.
        # Merge by keeping maximum access_count for each key
        merged_guarded: Dict[int, int] = {}
        for s in states:
            for def_site, access_count in s.dataflow_state.guarded_values.items():
                if def_site not in merged_guarded or access_count > merged_guarded[def_site]:
                    merged_guarded[def_site] = access_count

        # Merge registers conservatively (similar to stack merge)
        # Value is tainted if tainted on ANY path
        # Value is checked if checked on ALL paths where it's tainted
        merged_registers: Dict[int, Optional[DataFlowValue]] = {}

        # Collect all register keys from all states
        all_reg_keys = set()
        for s in states:
            all_reg_keys.update(s.dataflow_state.registers.keys())

        for reg_key in all_reg_keys:
            values = [
                s.dataflow_state.registers.get(reg_key)
                for s in states
            ]

            # Skip if all None
            if all(v is None for v in values):
                continue

            # Merge: tainted if ANY is tainted
            is_tainted = any(v and v.tainted for v in values if v)
            is_checked = False

            if is_tainted:
                tainted_values = [v for v in values if v and v.tainted]
                if tainted_values:
                    is_checked = all(v.checked for v in tainted_values)

            # Use first non-None value as template
            template = next((v for v in values if v), None)
            if template:
                metadata = {}
                if is_tainted:
                    # Filter out None values before merging taint metadata
                    non_none_reg_values = [v for v in values if v is not None]
                    sources, origins = self.base_analyzer._merge_taint_metadata(non_none_reg_values)
                    if sources:
                        metadata["taint_sources"] = list(sources)
                    if origins:
                        metadata["taint_origins"] = list(origins)
                non_none_reg_values = [v for v in values if v is not None]
                merged_ops = self._merge_comparison_operands(non_none_reg_values)
                if merged_ops:
                    metadata["comparison_operands"] = merged_ops

                merged_registers[reg_key] = DataFlowValue(
                    source=template.source,
                    definition_site=template.definition_site,
                    tainted=is_tainted,
                    checked=is_checked,
                    metadata=copy.deepcopy(metadata),
                )

        # Create merged state
        merged_dataflow_state = DataFlowState(
            stack=merged_stack,
            registers=merged_registers,
            guarded_values=merged_guarded
        )

        return PathState(
            path_id=states[0].path_id,  # Use first path's ID
            dataflow_state=merged_dataflow_state,
            path_condition=[],  # No specific path condition after merge
            merged_from_paths=merged_from_paths  # Track all merged path IDs
        )

    def _build_graph_from_paths(self, _view: ProgramView):
        """Build final data flow graph from path states"""

        for inst_idx, path_states_list in self.path_states.items():
            if not path_states_list:
                continue

            # Merge all path states for this instruction
            merged_state = self._merge_path_states(path_states_list)
            self.merged_states[inst_idx] = merged_state.dataflow_state

            # Record values in graph
            for stack_val in merged_state.dataflow_state.stack:
                if stack_val:
                    self.graph.values.setdefault(inst_idx, []).append(stack_val)

                    # Check for tainted unchecked values
                    if stack_val.tainted and not stack_val.checked:
                        # This is a potential vulnerability
                        inst = self.instructions_by_idx.get(inst_idx)
                        if inst is not None:
                            if inst.opcode in self.base_analyzer.SENSITIVE_OPCODES:
                                origins = self.base_analyzer._taint_origins_from_value(stack_val)
                                if not origins:
                                    origins = {stack_val.definition_site}
                                for origin in origins:
                                    pair = (origin, inst_idx)
                                    if pair not in self.graph._taint_dedup:
                                        self.graph._taint_dedup.add(pair)
                                        self.graph.tainted_propagation.append(pair)

    def get_tainted_paths(self) -> List[Tuple[int, int]]:
        """Get all paths where tainted data reaches sensitive operations"""
        return self.graph.tainted_propagation

    def get_path_states_at(self, inst_idx: int) -> List[PathState]:
        """Get all path states at a given instruction index"""
        return self.path_states.get(inst_idx, [])
