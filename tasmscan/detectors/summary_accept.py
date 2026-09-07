"""Summary-based accept detector.

This detector performs a lightweight check using block summaries rather than
full CFG analysis. It is designed to be fast but may produce false positives
in certain cases.

Limitations and False Positive Scenarios:
-----------------------------------------
1. Inter-procedural ACCEPT: If a caller block executes ACCEPT before calling
   a continuation that sends messages, this detector will flag the callee
   even though the execution path is safe. The call graph aware check below
   mitigates this partially but not completely.

2. Intra-block control flow: Within a single block, if ACCEPT is in a branch
   that always executes before SENDRAWMSG, but they appear in the same block
   summary, this detector won't flag it (which is correct). However, if they
   are in different basic blocks within the same continuation context, this
   detector may miss complex control flow relationships.

3. Computed continuations: When continuations are computed at runtime (e.g.,
   from a dictionary lookup), the call graph may be incomplete, leading to
   either false positives or false negatives.

For more precise analysis, use the NoAcceptBeforeSendDetector which performs
full CFG path traversal and dominance-aware checking.
"""
from typing import Dict, List, Optional, Set

from ..analyzer.facts import AnalysisFacts, BlockSummary
from .base import Detector
from .results import Vulnerability


class SummarySendRequiresAcceptDetector(Detector):
    """Summary-level: checks blocks that send without accept.

    This is a fast, summary-based check that may produce false positives when:
    - ACCEPT is executed in a calling context before the send
    - ACCEPT dominates SENDRAWMSG through complex control flow not captured in summaries

    For precise CFG-aware analysis, use NoAcceptBeforeSendDetector instead.
    """

    name = "summary_send_requires_accept"
    category = "security"
    default_severity = "low"  # Lowered from medium due to false positive potential
    description = (
        "Checks block summaries for send without accept. "
        "May produce false positives; see NoAcceptBeforeSendDetector for precise analysis."
    )

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        # Try TASIR-based detection first for improved semantic analysis
        tasir_findings = self._detect_with_tasir(facts)
        if tasir_findings is not None:
            return tasir_findings

        # Fallback to summary-based detection when TASIR is not available
        return self._detect_with_summaries(facts)

    def _detect_with_tasir(self, facts: AnalysisFacts) -> Optional[List[Vulnerability]]:
        """
        Detect send-without-accept using TASIR semantic matching.

        Uses InstructionKind.ACCEPT and InstructionKind.SEND_MESSAGE for
        more precise semantic matching than opcode string comparison.

        Returns:
            List of findings if TASIR is available, None to signal fallback
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        # Import InstructionKind at runtime to avoid circular imports
        from ..ir.tasir_types import InstructionKind

        findings: List[Vulnerability] = []

        # Group TASIR instructions by context
        accept_contexts: Set[str] = set()
        send_by_context: Dict[str, List] = {}

        for func in module.functions.values():
            for block in func.blocks.values():
                context_id = block.context_id
                for instr in block.instructions:
                    if instr.kind == InstructionKind.ACCEPT:
                        accept_contexts.add(context_id)
                    elif instr.kind == InstructionKind.SEND_MESSAGE:
                        if context_id not in send_by_context:
                            send_by_context[context_id] = []
                        send_by_context[context_id].append(instr)

        # Also check global blocks
        for block in module.global_blocks:
            context_id = block.context_id
            for instr in block.instructions:
                if instr.kind == InstructionKind.ACCEPT:
                    accept_contexts.add(context_id)
                elif instr.kind == InstructionKind.SEND_MESSAGE:
                    if context_id not in send_by_context:
                        send_by_context[context_id] = []
                    send_by_context[context_id].append(instr)

        # Build call-aware context suppression for inter-procedural ACCEPT.
        # Use both summary call graph edges (legacy) and concrete CFG call_cont
        # edges (context-precise when available).
        summary_by_context: Dict[str, BlockSummary] = {
            s.context_id: s for s in facts.summaries
        } if facts.summaries else {}
        callers_with_accept = self._find_callers_with_accept(facts, summary_by_context)
        callers_with_accept.update(
            self._find_cfg_callers_with_accept(
                facts=facts,
                accept_contexts=accept_contexts,
            )
        )

        # Find contexts that send without accept
        for context_id, send_instrs in send_by_context.items():
            if context_id in accept_contexts:
                continue

            # Check if any caller has already executed ACCEPT
            if context_id in callers_with_accept:
                continue

            # Find instruction for reporting (use first send instruction)
            context_instruction = self._find_context_instruction(facts, context_id)
            if context_instruction is not None:
                findings.append(
                    self._build_vuln(
                        message=(
                            f"Block {context_id} sends messages without ACCEPT in local scope. "
                            "Note: This may be a false positive if ACCEPT is executed in a calling context."
                        ),
                        instruction=context_instruction,
                        remediation=(
                            "Add ACCEPT before sending messages, or verify that all callers "
                            "execute ACCEPT before invoking this block."
                        ),
                        extra={"detection_method": "tasir", "send_count": len(send_instrs)},
                    )
                )

        return findings

    def _find_cfg_callers_with_accept(
        self,
        facts: AnalysisFacts,
        accept_contexts: Set[str],
    ) -> Set[str]:
        """Find callee contexts whose concrete CFG callers already ACCEPTed."""
        if not facts.cfg_edges:
            return set()

        idx_to_context: Dict[int, str] = {}
        for instr in facts.instructions:
            idx_to_context[instr.index] = instr.continuation_id or "main"

        contexts_with_accept_caller: Set[str] = set()
        for edge in facts.cfg_edges:
            if edge.kind != "call_cont" or edge.target is None:
                continue
            caller_context = idx_to_context.get(edge.source)
            callee_context = idx_to_context.get(edge.target)
            if caller_context is None or callee_context is None:
                continue
            if caller_context in accept_contexts:
                contexts_with_accept_caller.add(callee_context)

        return contexts_with_accept_caller

    def _detect_with_summaries(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """
        Fallback detection using block summaries when TASIR is not available.

        This is the original detection logic that operates on BlockSummary
        event counts rather than semantic instruction matching.
        """
        findings = []

        if not facts.summaries:
            return findings

        # Use unified query interface when module is available
        module = self.get_tasir(facts)
        send_events = module.events_of("send") if module else facts.events_of("send")
        # These event lists are used for quick existence checks below
        has_any_send = len(send_events) > 0
        if not has_any_send:
            return findings

        # Build summary lookup and call graph for inter-procedural analysis
        summary_by_context: Dict[str, BlockSummary] = {
            s.context_id: s for s in facts.summaries
        }
        callers_with_accept = self._find_callers_with_accept(facts, summary_by_context)

        for summary in facts.summaries:
            has_send = summary.event_counts.get("send", 0) > 0
            has_accept = summary.event_counts.get("accept", 0) > 0

            if has_send and not has_accept:
                # Check if any caller has already executed ACCEPT
                # This reduces false positives from inter-procedural calls
                if summary.context_id in callers_with_accept:
                    # Caller already has ACCEPT; skip to avoid false positive
                    # Note: This is still imprecise - caller's ACCEPT may not
                    # dominate the call site. Full CFG analysis is needed for precision.
                    continue

                # Find first instruction in this context for reporting
                context_instruction = self._find_context_instruction(facts, summary.context_id)
                if context_instruction is not None:
                    findings.append(
                        self._build_vuln(
                            message=(
                                f"Block {summary.context_id} sends messages without ACCEPT in local scope. "
                                "Note: This may be a false positive if ACCEPT is executed in a calling context."
                            ),
                            instruction=context_instruction,
                            remediation=(
                                "Add ACCEPT before sending messages, or verify that all callers "
                                "execute ACCEPT before invoking this block."
                            ),
                            extra={"detection_method": "summary"},
                        )
                    )

        return findings

    def _find_callers_with_accept(
        self, facts: AnalysisFacts, summary_by_context: Dict[str, BlockSummary]
    ) -> Set[str]:
        """Find contexts whose callers have ACCEPT.

        Returns a set of context IDs where at least one caller has ACCEPT.
        This is a conservative approximation - the caller's ACCEPT may not
        actually dominate the call site.
        """
        contexts_with_accept_caller: Set[str] = set()

        for edge in facts.call_graph_edges:
            caller_summary = summary_by_context.get(edge.caller)
            if caller_summary and caller_summary.event_counts.get("accept", 0) > 0:
                contexts_with_accept_caller.add(edge.callee)

        return contexts_with_accept_caller

    def _find_context_instruction(self, facts: AnalysisFacts, context_id: str):
        """Find the first instruction belonging to a context for error reporting."""
        for instr in facts.instructions:
            if context_id == "main" and instr.continuation_id is None:
                return instr
            if instr.continuation_id == context_id:
                return instr
        # Fallback to first instruction if context not found
        return facts.instructions[0] if facts.instructions else None
