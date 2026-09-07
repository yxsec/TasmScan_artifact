"""Dynamic call detector"""
from collections import deque
from typing import Any, List, Optional

from ..analyzer.facts import AnalysisFacts
from .base import Detector
from .results import Vulnerability


class DynamicCallDetector(Detector):
    """Detects dynamic calls with unknown targets."""

    name = "dynamic_call"
    category = "security"
    default_severity = "medium"
    description = "Flags dynamic calls that may be risky"

    def _sender_flows_to_call(self, facts: AnalysisFacts, sender_event: Any, call_site: Any) -> Optional[bool]:
        """
        Check if sender value flows to the call target via dataflow (dataflow enhancement).

        Uses tainted_propagation from dataflow graph to verify actual data flow
        instead of just checking temporal ordering. Also checks graph.values for
        tainted values when the call instruction is not in SENSITIVE_OPCODES.

        Returns:
            True if sender data flows to call site
            False if no flow detected
            None if dataflow graph unavailable (analysis incomplete)
        """
        graph = self.get_dataflow(facts)
        if graph is None:
            return None  # Cannot determine - analysis incomplete
        sender_def = sender_event.instruction.index
        call_idx = call_site.instruction.index

        # Method 1: Check tainted_propagation for flow from sender to call
        # Build a graph and do BFS to avoid relying on propagation ordering.
        adjacency = {}
        for source, sink in graph.tainted_propagation:
            adjacency.setdefault(source, set()).add(sink)

        queue = deque([sender_def])
        visited = {sender_def}
        while queue:
            current = queue.popleft()
            for nxt in adjacency.get(current, ()):
                if nxt == call_idx:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        # Method 2: Conservative over-approximation check (fallback)
        # Check if ANY tainted value originating from sender exists before the call site.
        #
        # Note: With Approach A implementation, Method 1 now covers DYNAMIC_CALL_OPCODES
        # (CALLX, EXECUTE, JMPX, etc.) in addition to SENSITIVE_OPCODES. Method 2 serves
        # as a fallback for edge cases where tainted_propagation might miss the flow
        # (e.g., complex stack manipulations not fully tracked).
        #
        # Trade-off: May produce false positives, but ensures we don't miss real
        # vulnerabilities.
        for inst_idx, values in graph.values.items():
            if inst_idx == call_idx:
                continue
            for val in values:
                if not val.tainted:
                    continue
                # Check if this value originates from sender
                origins_raw = val.metadata.get('taint_origins', [])
                origins = set(origins_raw) if origins_raw else {val.definition_site}
                if sender_def in origins:
                    # Sender-derived tainted value exists before the call site.
                    # Conservatively assume it could influence the dynamic call target.
                    if val.definition_site <= call_idx:
                        return True

        return False

    def _detect_with_tasir(self, facts: AnalysisFacts) -> Optional[List[Vulnerability]]:
        """
        TASIR-based detection using InstructionKind for semantic matching.

        Uses CONT_CALL, CONT_JUMP, and TRY_CATCH instruction kinds to identify
        dynamic continuation invocations with more precision than opcode-based
        matching.

        Returns:
            List of vulnerabilities if TASIR analysis succeeds, None to signal fallback needed
        """
        module = self.get_tasir(facts)
        if module is None:
            return None  # Signal fallback needed

        from ..ir.tasir_types import InstructionKind

        findings = []

        # Build set of call site instruction indices for cross-reference
        call_site_indices = {cs.instruction.index for cs in facts.call_sites}

        # Find all continuation-invocation instructions.
        # TRY/TRYARGS are modeled as TRY_CATCH in TASIR but are still dynamic
        # continuation call sites in AnalysisFacts.call_sites.
        for inst in module.all_instructions():
            if inst.kind not in (
                InstructionKind.CONT_CALL,
                InstructionKind.CONT_JUMP,
                InstructionKind.TRY_CATCH,
            ):
                continue

            # Cross-reference with facts.call_sites to find unknown targets
            if inst.index not in call_site_indices:
                continue

            # Find matching call_site
            matching_call_site = None
            for cs in facts.call_sites:
                if cs.instruction.index == inst.index:
                    matching_call_site = cs
                    break

            if matching_call_site is None or matching_call_site.target_type != "unknown":
                continue

            # Check if call target may come from user input (sender_read before call)
            sender_reads = [
                e
                for e in facts.events_of("sender_read")
                if e.instruction.index < inst.index
            ]

            # Only elevate severity if sender data actually flows to call
            severity = "medium"
            analysis_incomplete = False
            if sender_reads:
                for sender_event in sender_reads:
                    flow_result = self._sender_flows_to_call(facts, sender_event, matching_call_site)
                    if flow_result is None:
                        # Dataflow unavailable - keep medium severity but flag incomplete
                        analysis_incomplete = True
                        continue
                    elif flow_result is True:
                        severity = "high"
                        break

            extra = {"tasir_kind": inst.kind.name}
            if analysis_incomplete:
                extra["analysis_incomplete"] = True
                extra["reason"] = "dataflow_graph not available, flow analysis skipped"

            findings.append(
                self._build_vuln(
                    message=f"Dynamic call to unknown target at instruction {inst.index}",
                    instruction=matching_call_site.instruction,
                    severity=severity,
                    remediation="Verify the call target is validated before use.",
                    extra=extra,
                )
            )

        return findings

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        # Try TASIR-based detection first
        tasir_findings = self._detect_with_tasir(facts)
        if tasir_findings is not None:
            return tasir_findings

        # Fallback to existing logic
        findings = []

        for call_site in facts.call_sites:
            if call_site.target_type == "unknown":
                # Check if call target may come from user input (sender_read before call)
                sender_reads = [
                    e
                    for e in facts.events_of("sender_read")
                    if e.instruction.index < call_site.instruction.index
                ]

                severity = "medium"
                analysis_incomplete = False
                if sender_reads:
                    for sender_event in sender_reads:
                        flow_result = self._sender_flows_to_call(facts, sender_event, call_site)
                        if flow_result is None:
                            # Dataflow unavailable - keep medium severity but flag incomplete
                            # Continue checking other sender_reads in case one has confirmed flow
                            analysis_incomplete = True
                            continue
                        elif flow_result is True:
                            severity = "high"
                            break

                extra = {}
                if analysis_incomplete:
                    extra["analysis_incomplete"] = True
                    extra["reason"] = "dataflow_graph not available, flow analysis skipped"

                findings.append(
                    self._build_vuln(
                        message=f"Dynamic call to unknown target at instruction {call_site.instruction.index}",
                        instruction=call_site.instruction,
                        severity=severity,
                        remediation="Verify the call target is validated before use.",
                        extra=extra if extra else None,
                    )
                )

        return findings
