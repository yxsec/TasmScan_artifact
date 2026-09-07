"""Common CFG traversal utilities for detectors.

This module provides reusable BFS-based CFG traversal functionality
to reduce code duplication across detectors.

CFG handling Unknown Successor Handling Strategy
==========================================

When traversing the CFG, blocks may have `has_unknown_successor=True`,
indicating unresolved jump targets (computed continuations, indirect calls).
This module implements a conservative strategy for handling these cases:

1. `traverse_cfg_for_unguarded_sinks`:
   - Tracks `encountered_unknown_unguarded` flag in result
   - Optionally flags a specified index when unknown branches encountered unguarded
   - Used by: unchecked_sender detector

2. `traverse_cfg_for_specific_guard`:
   - Does NOT assume unknown paths contain the guard (conservative)
   - Callers must separately check `has_unknown_successor` and include in metadata
   - Used by: bounced_handler_pattern detector

Rationale:
- Unknown paths represent unanalyzable control flow
- For security analysis, we must NOT assume unknown paths are "safe" (have guards)
- Callers should report unknown successor presence to users for manual review
"""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from ..analyzer.facts import BasicBlock
from ..config import MAX_SEARCH_ITERATIONS, GUARD_OPCODES as CONFIG_GUARD_OPCODES


# Re-export for backward compatibility with existing imports
GUARD_OPCODES: FrozenSet[str] = CONFIG_GUARD_OPCODES

# Bytecode patterns commonly used for sender authentication that may not emit
# explicit guard events in the dataflow analyzer.
AUTH_OPCODES: FrozenSet[str] = frozenset({
    "SDEQ",
    "SDLEXCMP",
    "CHKSIGNU",
    "CHKSIGNS",
    "EQUAL",
})

# Only conditional guard opcodes participate in auth gating. Unconditional
# throws such as THROW/THROWARG do not consume a preceding auth predicate.
AUTH_CONTROL_OPCODES: FrozenSet[str] = frozenset({
    opcode for opcode in GUARD_OPCODES
    if opcode.startswith("IF") or "IF" in opcode
})

AUTH_GUARD_WINDOW = 4


@dataclass
class CFGTraversalResult:
    """Result of a CFG traversal operation."""

    flagged_indices: Set[int]
    """Instruction indices that matched the sink condition without guard."""

    truncated: bool
    """Whether the analysis was stopped due to iteration limit."""

    visited_states: int
    """Number of unique (block_id, guard_state, start_from) state tuples visited.
    Note: This counts state tuples, not unique blocks. A single block may be
    counted multiple times if visited with different guard states."""

    encountered_unknown: bool
    """Whether blocks with has_unknown_successor flag were encountered unguarded."""


# Sentinel for indicating full block traversal (no start_from offset)
_FULL_BLOCK_MARKER = object()


def traverse_cfg_for_unguarded_sinks(
    block_map: Dict[int, BasicBlock],
    start_block_id: int,
    start_from_idx: Optional[int],
    guard_indices: Set[int],
    sink_indices: Set[int],
    max_iterations: Optional[int] = None,
    flag_unknown_unguarded: bool = False,
    unknown_flag_idx: Optional[int] = None,
) -> CFGTraversalResult:
    """
    BFS traversal of CFG to find sink operations reachable without passing through guards.

    This is a generalized version of the pattern used by unchecked_sender and bounced_message
    detectors. It tracks whether a guard has been seen on each path, and flags sinks that
    are reached without having seen a guard.

    Args:
        block_map: Mapping from block ID to BasicBlock
        start_block_id: ID of the block to start traversal from
        start_from_idx: Optional instruction index to start from within the first block
                       (instructions before this index in the first block are skipped)
        guard_indices: Set of instruction indices that act as guards (once seen, path is safe)
        sink_indices: Set of instruction indices that are sinks (flagged if reached unguarded)
        max_iterations: Maximum BFS iterations (default: MAX_SEARCH_ITERATIONS from config)
        flag_unknown_unguarded: If True, flag unknown_flag_idx when unknown branches encountered unguarded
        unknown_flag_idx: Instruction index to flag when unknown branches are encountered unguarded

    Returns:
        CFGTraversalResult with flagged indices, truncation status, and statistics

    Example:
        >>> result = traverse_cfg_for_unguarded_sinks(
        ...     block_map=block_map,
        ...     start_block_id=entry_block_id,
        ...     start_from_idx=sender_read_idx,
        ...     guard_indices={guard_idx},
        ...     sink_indices={send_idx1, send_idx2},
        ... )
        >>> if result.flagged_indices:
        ...     print(f"Found {len(result.flagged_indices)} unguarded sinks")
    """
    if max_iterations is None:
        max_iterations = MAX_SEARCH_ITERATIONS

    flagged: Set[int] = set()
    queue: Deque[Tuple[int, bool, Optional[int]]] = deque()
    # State key: (block_id, guard_seen, start_from or sentinel)
    visited: Set[Tuple[int, bool, object]] = set()

    queue.append((start_block_id, False, start_from_idx))
    encountered_unknown_unguarded = False
    truncated = False
    iteration_count = 0

    while queue:
        iteration_count += 1
        if iteration_count > max_iterations:
            truncated = True
            break

        block_id, guard_seen, start_from = queue.popleft()

        # Build state key for deduplication
        if start_from is not None:
            state_key = (block_id, guard_seen, start_from)
        else:
            state_key = (block_id, guard_seen, _FULL_BLOCK_MARKER)

        if state_key in visited:
            continue
        visited.add(state_key)

        block = block_map.get(block_id)
        if not block:
            continue

        current_guard_seen = guard_seen

        # Process instructions in this block
        for idx in block.instruction_indices:
            # Skip instructions before start_from in the first block
            if start_from is not None and idx < start_from:
                continue

            # Check if this is a guard
            if idx in guard_indices:
                current_guard_seen = True

            # Check if this is a sink reached without guard
            if idx in sink_indices and not current_guard_seen:
                flagged.add(idx)

        # Process successors
        if block.has_unknown_successor and not current_guard_seen:
            encountered_unknown_unguarded = True
        for succ_id in block.successors:
            # Skip self-loops when guard state unchanged (optimization)
            if succ_id == block.id and current_guard_seen == guard_seen:
                continue
            queue.append((succ_id, current_guard_seen, None))

    # Handle unknown branches: flag the specified index if encountered unguarded
    if flag_unknown_unguarded and encountered_unknown_unguarded and unknown_flag_idx is not None:
        flagged.add(unknown_flag_idx)

    return CFGTraversalResult(
        flagged_indices=flagged,
        truncated=truncated,
        visited_states=len(visited),
        encountered_unknown=encountered_unknown_unguarded,
    )


def _iter_block_indices(
    block: BasicBlock,
    start_from_idx: Optional[int] = None,
    stop_before_idx: Optional[int] = None,
) -> Iterable[int]:
    """Yield instruction indices inside a block with optional slicing."""
    for idx in block.instruction_indices:
        if start_from_idx is not None and idx < start_from_idx:
            continue
        if stop_before_idx is not None and idx >= stop_before_idx:
            continue
        yield idx


def find_auth_guard_indices_in_block(
    block: BasicBlock,
    instructions_by_index: Dict[int, object],
    start_from_idx: Optional[int] = None,
) -> Set[int]:
    """
    Identify conditional guard instructions in a block that are fed by sender
    auth opcodes such as SDEQ/CHKSIGNU/EQUAL.
    """
    auth_positions: Deque[int] = deque()
    matched: Set[int] = set()

    for position, idx in enumerate(_iter_block_indices(block, start_from_idx=start_from_idx)):
        instruction = instructions_by_index.get(idx)
        if instruction is None:
            continue
        opcode = instruction.opcode

        while auth_positions and position - auth_positions[0] > AUTH_GUARD_WINDOW:
            auth_positions.popleft()

        if opcode in AUTH_OPCODES:
            auth_positions.append(position)
            continue

        if opcode in AUTH_CONTROL_OPCODES and auth_positions:
            matched.add(idx)

    return matched


def collect_auth_guard_indices(
    basic_blocks: List[BasicBlock],
    instructions_by_index: Dict[int, object],
) -> Set[int]:
    """Collect all synthetic auth-guard instruction indices across blocks."""
    matched: Set[int] = set()
    for block in basic_blocks:
        matched.update(find_auth_guard_indices_in_block(block, instructions_by_index))
    return matched


def build_predecessor_map(basic_blocks: List[BasicBlock]) -> Dict[int, Set[int]]:
    """Build a mapping from block ID to predecessor block IDs."""
    predecessors: Dict[int, Set[int]] = {block.id: set() for block in basic_blocks}
    for block in basic_blocks:
        for succ_id in block.successors:
            predecessors.setdefault(succ_id, set()).add(block.id)
    return predecessors


def is_block_guarded_on_all_predecessor_paths(
    block_map: Dict[int, BasicBlock],
    predecessor_map: Dict[int, Set[int]],
    start_block_id: int,
    stop_before_idx: int,
    guard_indices: Set[int],
    max_iterations: Optional[int] = None,
) -> Tuple[bool, bool]:
    """
    Check whether every reachable predecessor path to `stop_before_idx` crosses a
    guard/auth instruction before reaching the current block.

    Returns:
        Tuple of (all_paths_guarded, truncated)
    """
    if max_iterations is None:
        max_iterations = MAX_SEARCH_ITERATIONS

    queue: Deque[Tuple[int, bool, Optional[int]]] = deque()
    visited: Set[Tuple[int, bool, object]] = set()
    queue.append((start_block_id, False, stop_before_idx))
    iterations = 0

    while queue:
        iterations += 1
        if iterations > max_iterations:
            return False, True

        block_id, guard_seen, stop_before = queue.popleft()
        state_key = (
            block_id,
            guard_seen,
            stop_before if stop_before is not None else _FULL_BLOCK_MARKER,
        )
        if state_key in visited:
            continue
        visited.add(state_key)

        block = block_map.get(block_id)
        if block is None:
            continue

        current_guard_seen = guard_seen
        indices = list(_iter_block_indices(block, stop_before_idx=stop_before))
        for idx in reversed(indices):
            if idx in guard_indices:
                current_guard_seen = True
                break

        # Unknown successors imply incomplete predecessor coverage. If we still
        # have not seen a guard, conservatively treat the sender as unguarded.
        if block.has_unknown_successor and not current_guard_seen:
            return False, False

        predecessors = predecessor_map.get(block_id, set())
        if not predecessors:
            if not current_guard_seen:
                return False, False
            continue

        for pred_id in predecessors:
            queue.append((pred_id, current_guard_seen, None))

    return True, False


def traverse_cfg_for_specific_guard(
    block_map: Dict[int, BasicBlock],
    start_block_id: int,
    start_from_idx: int,
    specific_guard_idx: int,
    sink_indices: Set[int],
    max_iterations: int = MAX_SEARCH_ITERATIONS,
) -> CFGTraversalResult:
    """
    BFS traversal to find sinks reachable without passing through a specific guard instruction.

    This is a specialized version where we track passing through ONE specific guard instruction
    (e.g., a bounced flag check) rather than any guard event.

    CFG handling Conservative Handling:
    When blocks have has_unknown_successor=True and the guard has not been seen,
    we treat this conservatively by NOT assuming the unknown path has the guard.
    The encountered_unknown flag in the result indicates if such paths were found.

    Args:
        block_map: Mapping from block ID to BasicBlock
        start_block_id: ID of the block to start traversal from
        start_from_idx: Instruction index to start from
        specific_guard_idx: The specific guard instruction index to track
        sink_indices: Set of instruction indices that are sinks
        max_iterations: Maximum BFS iterations

    Returns:
        CFGTraversalResult with flagged indices, truncation status, visited blocks count,
        and encountered_unknown flag indicating if unknown successors were encountered unguarded.
    """
    flagged: Set[int] = set()
    queue: Deque[Tuple[int, bool, Optional[int]]] = deque()
    # Include start_from in state key for consistency with traverse_cfg_for_unguarded_sinks
    visited: Set[Tuple[int, bool, object]] = set()

    queue.append((start_block_id, False, start_from_idx))
    truncated = False
    encountered_unknown_unguarded = False
    iterations = 0

    while queue and iterations < max_iterations:
        iterations += 1
        block_id, passed_guard, start_from = queue.popleft()

        # Build state key including start_from for proper deduplication
        if start_from is not None:
            state_key = (block_id, passed_guard, start_from)
        else:
            state_key = (block_id, passed_guard, _FULL_BLOCK_MARKER)
        if state_key in visited:
            continue
        visited.add(state_key)

        block = block_map.get(block_id)
        if not block:
            continue

        current_passed_guard = passed_guard

        # Check instructions in this block
        for idx in block.instruction_indices:
            if start_from is not None and idx < start_from:
                continue

            # Check if this is THE specific guard
            if idx == specific_guard_idx:
                current_passed_guard = True

            # Check if this is a sink without having passed the guard
            if idx in sink_indices and not current_passed_guard:
                flagged.add(idx)

        # Note: BasicBlock always has has_unknown_successor attr (dataclass field)
        if block.has_unknown_successor and not current_passed_guard:
            encountered_unknown_unguarded = True

        # Continue to successor blocks
        # Note: BasicBlock always has successors attr (dataclass field with default_factory)
        for succ_id in block.successors:
            # Skip self-loops when guard state unchanged (optimization)
            if succ_id == block.id and current_passed_guard == passed_guard:
                continue
            if succ_id in block_map:
                queue.append((succ_id, current_passed_guard, None))

    if iterations >= max_iterations:
        truncated = True

    return CFGTraversalResult(
        flagged_indices=flagged,
        truncated=truncated,
        visited_states=len(visited),
        encountered_unknown=encountered_unknown_unguarded,
    )


def build_instruction_to_block_map(basic_blocks: List[BasicBlock]) -> Dict[int, int]:
    """
    Build a mapping from instruction index to containing block ID.

    Args:
        basic_blocks: List of basic blocks

    Returns:
        Dict mapping instruction index to block ID
    """
    instr_to_block: Dict[int, int] = {}
    for block in basic_blocks:
        for idx in block.instruction_indices:
            instr_to_block[idx] = block.id
    return instr_to_block
