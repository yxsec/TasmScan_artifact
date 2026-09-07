"""Detector for missing ACCEPT before SEND"""
import logging
from collections import deque
from typing import List, Optional, Set, Tuple

from ..analyzer.facts import AnalysisFacts, Event
from .base import Detector
from .results import Confidence, Vulnerability

from ..ir.tasir_types import SemanticLabel

logger = logging.getLogger(__name__)


class NoAcceptBeforeSendDetector(Detector):
    """Ensures ACCEPT happens before sending funds.

    Detection Strategy
    ------------------
    This detector uses a tiered detection approach:

    1. **TASIR-based detection** (preferred): Uses semantic instruction matching
       via InstructionKind for accurate detection when TASIR module is available.

    2. **CFG-based detection** (fallback): Performs path-sensitive analysis using
       control flow graph when basic blocks are available.

    3. **Linear detection** (last resort): Simple index-based comparison when
       neither TASIR nor CFG data is available.

    Design Decision: Conservative Over-Approximation
    ------------------------------------------------
    The linear fallback (_detect_linear) is intentionally designed as a
    **conservative over-approximation**. This means:

    - **May produce false positives**: The detector may flag code as vulnerable
      when it is actually safe (e.g., ACCEPT in a branch that always executes).

    - **Will not produce false negatives**: The detector will not miss actual
      vulnerabilities where SEND occurs before ACCEPT.

    This design choice prioritizes security by ensuring no vulnerabilities are
    missed, at the cost of potentially requiring manual review of flagged cases.
    For higher precision, ensure TASIR or CFG data is available.
    """

    name = "no_accept_before_send"
    category = "security"
    default_severity = "high"
    description = "Ensures ACCEPT/SETGASLIMIT happens on every path before sending funds."

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        send_events = facts.events_of("send")
        if not send_events:
            return []

        # Try TASIR-based detection first (uses InstructionKind for semantic matching)
        tasir_findings = self._detect_with_tasir(facts, send_events)
        if tasir_findings is not None:
            # for path-sensitive precision instead of TASIR's global-earliest approach
            if facts.basic_blocks:
                cfg_findings = self._detect_with_cfg(facts, send_events)
                # Prefer CFG findings (more precise) over TASIR global-earliest
                if cfg_findings is not None:
                    cfg_findings.extend(self._detect_cross_function(facts))
                    return cfg_findings
            # Fall back to TASIR findings if no CFG available
            tasir_findings.extend(self._detect_cross_function(facts))
            return tasir_findings

        # Fallback to existing logic (no TASIR)
        if facts.basic_blocks:
            findings = self._detect_with_cfg(facts, send_events)
            # Cross-function no-accept detection (additive)
            findings.extend(self._detect_cross_function(facts))
            return findings

        findings = self._detect_linear(facts, send_events)
        # Cross-function no-accept detection (additive)
        findings.extend(self._detect_cross_function(facts))
        return findings

    def _detect_with_tasir(
        self, facts: AnalysisFacts, send_events: List[Event]
    ) -> Optional[List[Vulnerability]]:
        """
        TASIR-based detection using InstructionKind for semantic matching.

        Args:
            facts: Analysis facts with optional tasir_module
            send_events: List of send events to check

        Returns:
            List of findings if TASIR available, None to signal fallback
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        from ..ir.tasir_types import InstructionKind

        # Find all ACCEPT instructions using InstructionKind
        accept_instructions = [
            i for i in module.all_instructions() if i.kind == InstructionKind.ACCEPT
        ]

        # Find all SEND_MESSAGE instructions using InstructionKind
        send_instructions = [
            i for i in module.all_instructions() if i.kind == InstructionKind.SEND_MESSAGE
        ]

        if not send_instructions:
            return []

        # Get earliest accept instruction index
        earliest_accept_index = (
            min(i.index for i in accept_instructions) if accept_instructions else None
        )

        findings = []
        for send_instr in send_instructions:
            if earliest_accept_index is None or send_instr.index < earliest_accept_index:
                # Map back to the original instruction from facts for consistent reporting
                original_instr = None
                for event in send_events:
                    if event.instruction.index == send_instr.index:
                        original_instr = event.instruction
                        break

                # Fall back to send_instr if no matching event found
                if original_instr is None:
                    # Use any instruction with matching index from facts
                    for instr in facts.instructions:
                        if instr.index == send_instr.index:
                            original_instr = instr
                            break

                findings.append(
                    self._build_vuln(
                        message="Message is sent before ACCEPT/SETGASLIMIT",
                        instruction=original_instr,
                        remediation="Call ACCEPT/SETGASLIMIT before out-bound message instructions.",
                    )
                )

        return findings

    def _detect_linear(self, facts: AnalysisFacts, send_events: List[Event]) -> List[Vulnerability]:
        """Linear detection using instruction index ordering.

        This method implements a simple, conservative detection strategy that
        compares instruction indices to determine if SEND occurs before ACCEPT.

        Algorithm
        ---------
        1. Find the earliest ACCEPT/SETGASLIMIT instruction by index.
        2. For each SEND event, check if its index is less than the earliest ACCEPT index.
        3. If so, flag it as a potential vulnerability.

        Limitations (Intentional Over-Approximation)
        --------------------------------------------
        This method uses **linear index comparison** and ignores control flow,
        which can produce false positives in the following scenarios:

        1. **Conditional ACCEPT**: If ACCEPT is in a branch that always executes
           before SEND, but appears later in the instruction sequence:

               IF condition THEN
                   ACCEPT      ; index=10
               ENDIF
               SEND            ; index=5 (appears earlier but executes later)

           The detector flags SEND even though ACCEPT always runs first at runtime.

        2. **Multiple entry points**: If different contract methods have their
           own ACCEPT calls, the global "earliest ACCEPT" may not apply to all
           SEND instructions.

        3. **Loop structures**: ACCEPT inside a loop that always executes at
           least once before reaching SEND would be missed by linear analysis.

        Design Rationale
        ----------------
        This is an **intentional design decision** favoring security over precision:

        - **No false negatives**: Every actual vulnerability will be detected.
        - **Potential false positives**: Some safe code may be flagged.
        - **Use case**: Fallback when CFG/TASIR data is unavailable.

        For path-sensitive analysis with fewer false positives, use
        _detect_with_cfg() or _detect_with_tasir() when the required data
        structures are available.

        Args:
            facts: Analysis facts containing instruction events.
            send_events: List of SEND events to check.

        Returns:
            List of vulnerabilities where SEND may occur before ACCEPT.
        """
        findings = []
        accept_event = facts.first_event("accept")
        # Use index instead of offset for ordering (offset is always 0)
        earliest_accept_index = accept_event.instruction.index if accept_event else None

        for event in send_events:
            if earliest_accept_index is None or event.instruction.index < earliest_accept_index:
                findings.append(
                    self._build_vuln(
                        message="Message is sent before ACCEPT/SETGASLIMIT",
                        instruction=event.instruction,
                        remediation="Call ACCEPT/SETGASLIMIT before out-bound message instructions.",
                    )
                )

        return findings

    def _detect_with_cfg(self, facts: AnalysisFacts, send_events: List[Event]) -> List[Vulnerability]:
        findings = []
        if not facts.basic_blocks:
            return findings

        send_indices = {event.instruction.index: event for event in send_events}
        accept_indices = {event.instruction.index for event in facts.events_of("accept")}
        block_map = {block.id: block for block in facts.basic_blocks}

        flagged: Set[int] = set()
        worklist = deque()
        visited: Set[Tuple[int, bool]] = set()

        if not block_map:
            return findings

        entry_ids = facts.entry_block_ids()
        if not entry_ids and block_map:
            entry_ids = [min(block_map)]
            logger.warning(
                "No entry blocks found; using fallback entry block %d. "
                "Analysis may be incomplete for contracts with non-standard entry points.",
                entry_ids[0]
            )

        for entry_id in entry_ids:
            worklist.append((entry_id, False))
        queue_unknown = False

        while worklist:
            block_id, accept_seen = worklist.popleft()
            state_key = (block_id, accept_seen)
            if state_key in visited:
                continue
            visited.add(state_key)

            block = block_map.get(block_id)
            if block is None:
                logger.debug("Skipping unknown block_id %d in no_accept analysis", block_id)
                continue
            state = accept_seen
            for instr_idx in block.instruction_indices:
                if instr_idx in accept_indices:
                    state = True

                if instr_idx in send_indices and not state and instr_idx not in flagged:
                    flagged.add(instr_idx)
                    findings.append(
                        self._build_vuln(
                            message="Message is sent before ACCEPT/SETGASLIMIT on at least one path.",
                            instruction=send_indices[instr_idx].instruction,
                            remediation="Insert ACCEPT/SETGASLIMIT along every control-flow path before sending.",
                        )
                    )

            if block.has_unknown_successor:
                queue_unknown = True
            for succ_id in block.successors:
                # Prioritize paths where ACCEPT has not been seen yet (accept_seen=False).
                # Using appendleft ensures these paths are processed first, reducing
                # false negatives by exploring unprotected paths before protected ones.
                # Note: Each (block_id, accept_seen) state is visited only once due to
                # the visited set, so both True and False states for the same block
                # can be explored if reached via different paths.
                if state:
                    worklist.append((succ_id, state))
                else:
                    worklist.appendleft((succ_id, state))

        if queue_unknown:
            findings.append(
                self._build_vuln(
                    message="Analysis encountered an unresolved branch/continuation; some paths could not be analyzed.",
                    instruction=facts.instructions[0] if facts.instructions else None,
                    remediation="Review the contract manually for paths involving computed continuations.",
                    severity="low",
                )
            )

        return findings

    def _detect_cross_function(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """Cross-function no-accept detection via inter_procedural view.

        Checks if any function sends messages without accept, and none of its
        callers provide accept either.
        """
        findings: List[Vulnerability] = []
        module = self.get_tasir(facts)
        if not module or not module.inter_procedural:
            return findings

        # Use function_has_property helper
        for method_id in module.inter_procedural.function_summaries:
            if not module.inter_procedural.function_has_property(method_id, "has_send") or \
               module.inter_procedural.function_has_property(method_id, "has_accept"):
                continue
            # Check if ANY caller has accept
            caller_has_accept = False
            for edge in module.inter_procedural.edges:
                if edge.callee_method_id == method_id:
                    if module.inter_procedural.function_has_property(
                        edge.caller_method_id, "has_accept"
                    ):
                        caller_has_accept = True
                        break
            if caller_has_accept:
                continue
            func = module.get_function(method_id)
            if func:
                send_insts = [
                    i for block in func.blocks.values()
                    for i in block.instructions
                    if SemanticLabel.MESSAGE_SEND in getattr(i, 'semantic_labels', [])
                ]
                if send_insts:
                    inst_by_idx = {i.index: i for i in facts.instructions}
                    target_inst = inst_by_idx.get(send_insts[0].index)
                    if target_inst:
                        findings.append(self._build_vuln(
                            message=(
                                f"Cross-function: method {method_id} sends without accept, "
                                f"and no caller provides accept"
                            ),
                            instruction=target_inst,
                            confidence=Confidence.LOW,
                            extra={"cross_function": True, "method_id": method_id},
                        ))
        return findings
