"""
Stack analysis for TVM program analysis.

This module handles:
- Computing stack height at each instruction
- Handling join points with CFG awareness
- Stack delta calculation for opcodes
"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import CONTRACT_ENTRY_STACK_HEIGHT, RUN_TICKTOCK_STACK_HEIGHT
from .constants import CALL_OPCODES, MAIN_CONTEXT, UNCONDITIONAL_BRANCHES
from .facts import InstructionFact, StackState

logger = logging.getLogger(__name__)


class StackAnalyzer:
    """Analyzes stack heights for TVM instruction sequences."""

    def __init__(self) -> None:
        self.last_analysis_metadata: Dict[str, Any] = {}

    def analyze_stack(
        self,
        instructions: Sequence[InstructionFact],
        continuations: Optional[Dict[str, Any]] = None,
        cfg_edges: Optional[Sequence] = None,
        initial_height: Optional[int] = None,
    ) -> List[StackState]:
        """
        Analyze stack heights per context (main + continuations).

        Uses CFG edges to properly handle join points where multiple predecessors
        may have different stack heights.

        Args:
            instructions: List of instructions to analyze
            continuations: Optional continuation definitions
            cfg_edges: Optional CFG edges for join point handling
            initial_height: Initial stack height for analysis. If None, uses
                TON contract entry height (5) as default. Pass 0 for tests
                simulating empty stack scenarios.
        """
        if continuations is None:
            continuations = {}
        if cfg_edges is None:
            cfg_edges = []
        if not instructions:
            self.last_analysis_metadata = {
                "global_converged": True,
                "global_iterations": 0,
                "global_iteration_limit": 0,
            }
            return []
        self.last_analysis_metadata = {}

        # Build predecessor map from CFG edges
        # predecessors[target] = list of source instruction indices
        predecessors: Dict[int, List[int]] = {}
        for edge in cfg_edges:
            if edge.target is not None:
                predecessors.setdefault(edge.target, []).append(edge.source)

        states_by_index: Dict[int, StackState] = {}

        # Use CONTRACT_ENTRY_STACK_HEIGHT from config (default TON entry point stack)
        entry_height = initial_height if initial_height is not None else CONTRACT_ENTRY_STACK_HEIGHT

        main_facts = [f for f in instructions if f.continuation_id is None]
        cont_facts_by_id: Dict[str, List[InstructionFact]] = {
            cont_id: [f for f in instructions if f.continuation_id == cont_id]
            for cont_id in sorted(continuations.keys())
        }

        context_order: List[str] = []
        if main_facts:
            context_order.append(MAIN_CONTEXT)
        context_order.extend(
            cont_id for cont_id in sorted(cont_facts_by_id.keys())
            if cont_facts_by_id.get(cont_id)
        )

        def derive_entry_from_state(state: StackState) -> Tuple[int, int, Optional[int], bool]:
            start_height_local = state.height_after
            if start_height_local is None:
                start_height_local = state.height_before or 0
            start_min_local = (
                state.height_min
                if state.height_min is not None
                else start_height_local
            )
            if state.height_max is not None:
                start_max_local: Optional[int] = state.height_max
            else:
                start_max_local = None if state.unknown else start_height_local
            start_unknown_local = state.unknown or state.height_after is None
            return (
                start_height_local,
                start_min_local,
                start_max_local,
                start_unknown_local,
            )

        def initial_entry_for_context(context_id: str) -> Tuple[int, int, Optional[int], bool]:
            if context_id == MAIN_CONTEXT:
                return (entry_height, entry_height, entry_height, False)

            continuation = continuations.get(context_id)
            if continuation is None:
                return (0, 0, None, True)

            parent_idx = continuation.parent_instruction_index
            parent_state = states_by_index.get(parent_idx) if parent_idx is not None else None
            if parent_state is not None:
                return derive_entry_from_state(parent_state)

            # Method entry points: stack height depends on calling convention.
            # See: https://docs.ton.org/v3/documentation/tvm/tvm-initialization
            method_id = continuation.method_id
            if method_id == 0:
                # recv_internal: 5 args (selector=0, body, msg_cell, msg_value, balance)
                h = CONTRACT_ENTRY_STACK_HEIGHT
                return (h, h, h, False)
            if method_id == -1:
                # recv_external: 5 args (selector=-1, body, msg_cell, msg_value, balance)
                h = CONTRACT_ENTRY_STACK_HEIGHT
                return (h, h, h, False)
            if method_id == -2:
                # run_ticktock: 4 args (selector=-2, is_tock, account_addr, balance)
                h = RUN_TICKTOCK_STACK_HEIGHT
                return (h, h, h, False)

            # Get methods (method_id > 0) or unknown: conservative handling.
            # Argument count varies, use unbounded max.
            return (0, 0, None, True)

        max_global_iterations = max(4, min(128, max(1, len(context_order)) * 4))
        global_iteration = 0
        changed = True
        while changed and global_iteration < max_global_iterations:
            changed = False
            for context_id in context_order:
                if context_id == MAIN_CONTEXT:
                    facts_in_context = main_facts
                else:
                    facts_in_context = cont_facts_by_id.get(context_id, [])
                if not facts_in_context:
                    continue

                start_height, start_min, start_max, start_unknown = initial_entry_for_context(context_id)
                prev_states = {
                    fact.index: states_by_index.get(fact.index)
                    for fact in facts_in_context
                }
                new_states = self._analyze_stack_for_context_cfg_aware(
                    facts_in_context,
                    initial_height=start_height,
                    initial_min=start_min,
                    initial_max=start_max,
                    initial_unknown=start_unknown,
                    predecessors=predecessors,
                    states_by_index=states_by_index,
                )
                for state in new_states:
                    if prev_states.get(state.instruction_index) != state:
                        changed = True
                    states_by_index[state.instruction_index] = state
            global_iteration += 1

        if changed:
            logger.warning(
                "Global stack analysis did not converge after %d iterations, results may be incomplete",
                max_global_iterations,
            )
        self.last_analysis_metadata = {
            "global_converged": not changed,
            "global_iterations": global_iteration,
            "global_iteration_limit": max_global_iterations,
        }

        ordered_states: List[StackState] = []
        for fact in instructions:
            state = states_by_index.get(fact.index)
            if state:
                ordered_states.append(state)
        return ordered_states

    def _merge_predecessor_states(
        self,
        predecessor_states: List[StackState],
    ) -> Tuple[int, int, Optional[int], bool]:
        """Merge multiple predecessor states at a join point using conservative bounds.

        Returns: (height, height_min, height_max, unknown)
        """
        if not predecessor_states:
            return (0, 0, 0, True)
        if len(predecessor_states) == 1:
            s = predecessor_states[0]
            height = s.height_after if s.height_after is not None else (s.height_before or 0)
            h_min = s.height_min if s.height_min is not None else height
            h_max = s.height_max
            return (height, h_min, h_max, s.unknown)

        # Collect heights from all predecessors
        heights = []
        mins = []
        maxs: List[int] = []
        any_unbounded = False
        any_unknown = False

        for s in predecessor_states:
            h = s.height_after if s.height_after is not None else (s.height_before or 0)
            heights.append(h)
            mins.append(s.height_min if s.height_min is not None else h)
            if s.height_max is None:
                any_unbounded = True
            else:
                maxs.append(s.height_max)
            if s.unknown or s.height_after is None:
                any_unknown = True

        # Conservative merge: use min for height_min, max for height_max
        merged_height_min = min(mins)
        merged_height_max = None if any_unbounded else max(maxs)

        # Use min height as the concrete estimate to avoid "false precision"
        # when predecessors disagree.
        merged_height = merged_height_min

        # Mark as uncertain if any predecessor was uncertain or heights differ
        if merged_height_max is None:
            merged_unknown = True
        else:
            height_spread = merged_height_max - merged_height_min
            merged_unknown = any_unknown or height_spread > 0

        return (merged_height, merged_height_min, merged_height_max, merged_unknown)

    def _analyze_stack_for_context_cfg_aware(
        self,
        facts: Sequence[InstructionFact],
        initial_height: int,
        initial_min: int,
        initial_max: Optional[int],
        initial_unknown: bool,
        predecessors: Dict[int, List[int]],
        states_by_index: Dict[int, StackState],
    ) -> List[StackState]:
        """Analyze stack heights for a context with CFG awareness.

        At join points (instructions with multiple predecessors), merges predecessor
        states using conservative bounds to handle different stack heights on
        different paths.
        """
        states: List[StackState] = []
        if not facts:
            return states

        # Build set of instruction indices in this context for quick lookup
        context_indices = {f.index for f in facts}
        first_idx = facts[0].index if facts else None

        # Seed with any previously computed states for this context
        current_states: Dict[int, StackState] = {
            idx: states_by_index[idx]
            for idx in context_indices
            if idx in states_by_index
        }

        entry_state = StackState(
            instruction_index=-1,
            height_before=initial_height,
            height_after=initial_height,
            delta=0,
            unknown=initial_unknown,
            height_min=initial_min,
            height_max=initial_max,
        )

        # Iterate to a fixpoint with a bounded number of passes to handle loops.
        max_iterations = max(4, min(128, len(facts) * 2))
        iteration = 0
        changed = True

        while changed and iteration < max_iterations:
            logger.debug("Fixpoint iteration %d", iteration)
            changed = False
            height = initial_height
            height_min = initial_min
            height_max = initial_max
            certainty = 0.5 if initial_unknown else 1.0

            for fact in facts:
                # Check if this instruction has predecessors (join point)
                pred_indices = predecessors.get(fact.index, [])

                # Collect predecessor states from current pass or prior contexts
                analyzed_preds = []
                for pred_idx in pred_indices:
                    pred_state = current_states.get(pred_idx) or states_by_index.get(pred_idx)
                    if pred_state:
                        analyzed_preds.append(pred_state)

                # Include implicit entry state for the first instruction
                if fact.index == first_idx and analyzed_preds:
                    analyzed_preds = analyzed_preds + [entry_state]

                # If this is a join point with multiple analyzed predecessors, merge them
                if len(analyzed_preds) > 1:
                    merged_h, merged_min, merged_max, merged_unknown = \
                        self._merge_predecessor_states(analyzed_preds)
                    height = merged_h
                    height_min = merged_min
                    height_max = merged_max
                    if merged_unknown:
                        certainty = min(certainty, 0.5)
                elif len(analyzed_preds) == 1 and fact.index != first_idx:
                    # Single predecessor from CFG - use its state if different from linear
                    pred_state = analyzed_preds[0]
                    pred_h = pred_state.height_after if pred_state.height_after is not None \
                        else (pred_state.height_before or 0)
                    pred_min = pred_state.height_min if pred_state.height_min is not None else pred_h
                    if pred_state.height_max is not None:
                        pred_max = pred_state.height_max
                    else:
                        pred_max = None if pred_state.unknown else pred_h
                    # Update from predecessor state
                    height = pred_h
                    height_min = pred_min
                    height_max = pred_max
                    if pred_state.unknown:
                        certainty = min(certainty, 0.5)

                min_delta, max_delta, known = self._stack_delta_bounds(fact.opcode)

                if known:
                    height_after = height + min_delta
                    height_min_after = height_min + min_delta
                    height_max_after = height_max + min_delta if height_max is not None else None

                    state = StackState(
                        instruction_index=fact.index,
                        height_before=height,
                        height_after=height_after,
                        delta=min_delta,
                        unknown=(certainty < 0.8),
                        height_min=height_min_after,
                        height_max=height_max_after,
                    )
                    height = height_after
                    height_min = height_min_after
                    height_max = height_max_after
                else:
                    height_min_after = height_min + min_delta
                    if height_max is None or max_delta is None:
                        height_max_after = None
                    else:
                        height_max_after = height_max + max_delta
                    height_after = height + min_delta if max_delta is not None and max_delta == min_delta else None

                    state = StackState(
                        instruction_index=fact.index,
                        height_before=height,
                        height_after=height_after,
                        delta=min_delta,
                        unknown=True,
                        height_min=height_min_after,
                        height_max=height_max_after,
                    )

                    height = height + min_delta
                    height_min = height_min_after
                    height_max = height_max_after
                    certainty *= 0.9

                if current_states.get(fact.index) != state:
                    changed = True
                current_states[fact.index] = state

            iteration += 1

        if changed:
            logger.warning(
                "Stack analysis did not converge after %d iterations, results may be incomplete",
                max_iterations,
            )
        elif iteration > 1:
            logger.debug(f"Stack analysis fixpoint reached after {iteration} iterations")

        for fact in facts:
            state = current_states.get(fact.index)
            if state:
                states.append(state)

        return states

    def _stack_delta_bounds(
        self, opcode: str, *, allow_branch_effects: bool = True
    ) -> Tuple[int, Optional[int], bool]:
        """Get stack delta bounds for opcode.

        Uses StackEffectDatabase as the single source of truth for opcode
        stack effects, with prefix rules as fallback for truly unknown opcodes.

        Returns: (min_delta, max_delta, known_exact)
            min_delta: minimum possible net stack effect
            max_delta: maximum possible net stack effect (None if unknown/unbounded)
            known_exact: True if delta is exact (min == max and non-dynamic)
        """
        upper = opcode.upper()

        # For calls/branches, stack effect is complex (continuation-dependent)
        # Allow callers to suppress branch effects (e.g., continuation mapping)
        if not allow_branch_effects and (upper in CALL_OPCODES or upper in UNCONDITIONAL_BRANCHES):
            return 0, 0, False

        # Primary: StackEffectDatabase (single source of truth)
        from ..ir.stack_effects import get_stack_effect
        effect = get_stack_effect(upper)
        if effect:
            min_effect, max_effect = effect.net_effect_range()
            if min_effect is None:
                min_effect = effect.net_effect
            if (
                max_effect is not None
                and min_effect == max_effect
                and not effect.is_dynamic
            ):
                return min_effect, max_effect, True
            return min_effect, max_effect, False

        # Fallback: Prefix rules for truly unknown opcodes
        if upper.startswith("PUSH"):
            return 1, 1, True
        if upper.startswith("POP") or upper.startswith("DROP"):
            return -1, -1, True
        if upper.startswith("XCHG") or upper.startswith("SWAP"):
            return 0, 0, True

        # Conservative fallback: assume no minimum stack effect
        return 0, None, False

    def stack_delta(self, opcode: str, *, allow_branch_effects: bool = True) -> Tuple[int, bool]:
        """Get stack delta for opcode.

        Returns: (delta, known) where:
            delta: estimated stack effect (negative = pop, positive = push)
            known: True if delta is accurate, False if it's a conservative estimate
        """
        min_delta, _max_delta, known = self._stack_delta_bounds(
            opcode, allow_branch_effects=allow_branch_effects
        )
        return min_delta, known
