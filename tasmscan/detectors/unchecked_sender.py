"""Detector for unchecked msg_sender"""
from typing import Any, Dict, List, Optional, Set, Tuple

from ..analyzer.facts import AnalysisFacts, BasicBlock, Event
from ..config import MAX_SEARCH_ITERATIONS
from ..ir.dataflow import DataFlowAnalyzer
from .base import Detector
from .cfg_utils import (
    build_instruction_to_block_map,
    build_predecessor_map,
    collect_auth_guard_indices,
    is_block_guarded_on_all_predecessor_paths,
    traverse_cfg_for_unguarded_sinks,
)
from .results import Confidence, Vulnerability

from ..ir.tasir_types import SemanticLabel


class UncheckedSenderDetector(Detector):
    """Ensures msg_sender is validated through a guard."""

    name = "unchecked_sender"
    category = "security"
    default_severity = "medium"
    description = "Requires msg_sender reads to be guarded before sensitive instructions."

    # Opcodes that indicate sender authentication at bytecode level
    SENDER_AUTH_OPCODES: Set[str] = {
        "SDEQ",       # slice deep-equal (equal_slices / equal_slice_bits)
        "SDLEXCMP",   # slice lexicographic compare
        "CHKSIGNU",   # check Ed25519 signature (unsigned hash)
        "CHKSIGNS",   # check Ed25519 signature (slice data)
    }
    BRANCH_ENTRY_AUTH_WINDOWS: Dict[str, int] = {
        # IFREFELSE auth setup is usually compact; longer windows grew TP risk.
        "IFREFELSE": 10,
        # IFELSE helper entries sometimes carry a longer equal_slices setup.
        "IFELSE": 30,
    }

    SENSITIVE_OPCODES: Set[str] = {
        "SENDRAWMSG", "SENDMSG", "RAWRESERVE", "RAWRESERVEX",
        "ACCEPT", "SETGASLIMIT", "SETCODE", "COMMIT",
    }

    def _has_bytecode_sender_guard(self, facts: AnalysisFacts) -> bool:
        """Heuristic: check if bytecode contains sender-authentication patterns.

        Returns True if any of these patterns are found:
        1. Slice comparison (SDEQ/SDLEXCMP) + conditional throw/branch
        2. Signature verification (CHKSIGNU/CHKSIGNS) anywhere
        3. EQUAL opcode near THROWIFNOT (hash-based address comparison)
        4. SDEQ near IFJMP/IFNOTJMP (branch-based sender check)
        """
        has_slice_cmp = False
        has_sig_check = False
        has_throwif = False
        has_equal_throw = False
        has_branch_guard = False

        for i, inst in enumerate(facts.instructions):
            if inst.opcode in ("SDEQ", "SDLEXCMP"):
                has_slice_cmp = True
                # Check if followed by branch within 3 instructions
                for j in range(i + 1, min(i + 4, len(facts.instructions))):
                    if facts.instructions[j].opcode in ("IFJMP", "IFNOTJMP",
                                                         "IF", "IFNOT", "IFELSE"):
                        has_branch_guard = True
                        break
            if inst.opcode in ("CHKSIGNU", "CHKSIGNS"):
                has_sig_check = True
            if inst.opcode in ("THROWIF", "THROWIFNOT", "THROWARGIF",
                               "THROWARGIFNOT"):
                has_throwif = True
            if inst.opcode == "EQUAL":
                # Check if followed by THROWIFNOT within 2 instructions
                for j in range(i + 1, min(i + 3, len(facts.instructions))):
                    if facts.instructions[j].opcode in ("THROWIFNOT", "THROWIF"):
                        has_equal_throw = True
                        break

        return (
            (has_slice_cmp and (has_throwif or has_branch_guard))
            or has_sig_check  # Signature check alone is sufficient auth
            or has_equal_throw  # EQUAL + THROW pattern (address hash comparison)
        )

    def _has_sensitive_ops(self, facts: AnalysisFacts) -> bool:
        """Check if contract has any security-sensitive operations."""
        return any(inst.opcode in self.SENSITIVE_OPCODES
                   for inst in facts.instructions)

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        findings = []

        if self._is_trivial_contract(facts):
            return findings

        # Use index instead of offset for ordering (offset is always 0)
        sender_reads = sorted(facts.events_of("sender_read"), key=lambda e: e.instruction.index)
        if not sender_reads:
            return findings

        # Try TASIR-based sink identification first for semantic classification
        # This uses InstructionKind.SEND_MESSAGE instead of opcode string matching
        tasir_send_indices = self._detect_with_tasir(facts, sender_reads)

        dataflow_findings, dataflow_sender_indices, dataflow_sink_indices = self._detect_with_dataflow(facts, sender_reads)
        findings.extend(dataflow_findings)

        if facts.basic_blocks:
            remaining_reads = [
                event for event in sender_reads
                if event.instruction.index not in dataflow_sender_indices
            ]
            if not remaining_reads:
                return self._filter_with_heuristic_guard(findings, facts)
            # Pass TASIR-identified sinks if available, otherwise use event-based detection
            findings.extend(
                self._detect_with_cfg(
                    facts, remaining_reads, dataflow_sink_indices, tasir_send_indices
                )
            )
            # Cross-function unchecked sender detection
            findings.extend(self._detect_cross_function(facts))
            return self._filter_with_heuristic_guard(findings, facts)

        # Cross-function unchecked sender detection
        findings.extend(self._detect_cross_function(facts))

        guard_events = sorted(facts.events_of("guard"), key=lambda e: e.instruction.index)

        # Use TASIR-identified send indices if available, fallback to events
        if tasir_send_indices is not None:
            # TASIR available: filter instructions by TASIR-identified send indices
            send_events = [
                Event(type="send", instruction=facts.instructions[idx])
                for idx in sorted(tasir_send_indices)
                if 0 <= idx < len(facts.instructions)
            ]
        else:
            # Fallback: use event-based send detection
            send_events = sorted(facts.events_of("send"), key=lambda e: e.instruction.index)

        for sender_event in sender_reads:
            if sender_event.instruction.index in dataflow_sender_indices:
                continue
            guard = self._find_guard_after(sender_event, guard_events)
            next_send = self._find_next_event(sender_event, send_events)

            guard_is_before_send = (
                guard is not None
                and (next_send is None or guard.instruction.index < next_send.instruction.index)
            )

            if guard_is_before_send:
                continue

            findings.append(
                self._build_vuln(
                    message="Sender is read without a guard/condition before sensitive actions.",
                    instruction=sender_event.instruction,
                    remediation="Validate msg_sender with IF*/THROW* before using it.",
                )
            )

        return self._filter_with_heuristic_guard(findings, facts)

    def _filter_with_heuristic_guard(self, findings, facts):
        """Remove findings if contract has no sensitive operations.

        Contract-global auth suppression was tested but removed because
        many contracts have auth on SOME paths but not all — suppressing
        all findings loses more TP than FP.
        """
        if not findings:
            return findings
        # No sensitive ops = nothing dangerous even if sender unchecked
        if not self._has_sensitive_ops(facts):
            return []
        return findings

    def _detect_with_dataflow(
        self, facts: AnalysisFacts, sender_reads: List[Event]
    ) -> Tuple[List[Vulnerability], Set[int], Set[int]]:
        """Use taint analysis to find sender-derived unchecked flows.

        Returns:
            Tuple of (findings, covered_origin_indices, covered_sink_indices)
        """
        if not sender_reads:
            return [], set(), set()

        sender_indices = {event.instruction.index for event in sender_reads}
        # Prefer get_dataflow (uses TVMModule's integrated copy if available)
        graph = self.get_dataflow(facts)
        if graph is None:
            graph = getattr(facts, "dataflow_graph", None)
        if graph is None:
            analyzer = DataFlowAnalyzer()
            graph = analyzer.analyze(facts, path_sensitive=bool(facts.basic_blocks))

        if not graph.tainted_propagation:
            return [], set(), set()

        block_map: Dict[int, BasicBlock] = {}
        instr_to_block: Dict[int, int] = {}
        predecessor_map: Dict[int, Set[int]] = {}
        instructions_by_index = {inst.index: inst for inst in facts.instructions}
        context_instruction_indices: Dict[str, List[int]] = {}
        for inst in sorted(facts.instructions, key=lambda item: item.index):
            context_instruction_indices.setdefault(
                inst.continuation_id or "main", []
            ).append(inst.index)
        guard_indices: Set[int] = set()
        if facts.basic_blocks:
            block_map = {block.id: block for block in facts.basic_blocks}
            instr_to_block = build_instruction_to_block_map(facts.basic_blocks)
            predecessor_map = build_predecessor_map(facts.basic_blocks)
            guard_indices = {event.instruction.index for event in facts.events_of("guard")}
            guard_indices.update(
                collect_auth_guard_indices(facts.basic_blocks, instructions_by_index)
            )
        continuations = facts.metadata.get("continuations", {}) if facts.metadata else {}

        sinks: Dict[int, Set[int]] = {}
        for origin_idx, sink_idx in graph.tainted_propagation:
            if origin_idx not in sender_indices:
                continue
            if facts.basic_blocks and not self._keep_dataflow_pair_after_cfg_replay(
                origin_idx=origin_idx,
                sink_idx=sink_idx,
                block_map=block_map,
                instr_to_block=instr_to_block,
                predecessor_map=predecessor_map,
                guard_indices=guard_indices,
                instructions_by_index=instructions_by_index,
                context_instruction_indices=context_instruction_indices,
                continuations=continuations,
            ):
                continue
            sinks.setdefault(sink_idx, set()).add(origin_idx)

        findings = []
        covered_origins: Set[int] = set()
        covered_sinks: Set[int] = set()
        for sink_idx, origins in sinks.items():
            if sink_idx < 0 or sink_idx >= len(facts.instructions):
                continue
            covered_origins.update(origins)
            covered_sinks.add(sink_idx)
            findings.append(
                self._build_vuln(
                    message="Sender-derived value reaches sensitive operation without guard (taint analysis).",
                    instruction=facts.instructions[sink_idx],
                    remediation="Validate msg_sender before using it in sensitive operations.",
                    extra={"sender_origins": sorted(origins), "signal": "dataflow"},
                )
            )

        return findings, covered_origins, covered_sinks

    @staticmethod
    def _context_key_for_instruction(
        instruction,
        block: Optional[BasicBlock],
    ) -> str:
        """Normalize instruction/block context for same-path guard replay."""
        if instruction is not None and instruction.continuation_id is not None:
            return instruction.continuation_id
        if block is not None and getattr(block, "context", None):
            return block.context
        return "main"

    def _keep_dataflow_pair_after_cfg_replay(
        self,
        origin_idx: int,
        sink_idx: int,
        block_map: Dict[int, BasicBlock],
        instr_to_block: Dict[int, int],
        predecessor_map: Dict[int, Set[int]],
        guard_indices: Set[int],
        instructions_by_index: Dict[int, object],
        context_instruction_indices: Dict[str, List[int]],
        continuations: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Re-check dataflow pairs against local CFG guards.

        Dataflow taint propagation is more global than the CFG guard model and
        can over-report when auth checks in the same continuation were missed by
        the taint analyzer. To avoid suppressing true cross-context flows, only
        replay CFG guards when the sender origin and sink stay in the same CFG
        context.
        """
        origin_block_id = instr_to_block.get(origin_idx)
        sink_block_id = instr_to_block.get(sink_idx)
        if origin_block_id is None or sink_block_id is None:
            return True

        origin_block = block_map.get(origin_block_id)
        sink_block = block_map.get(sink_block_id)
        if origin_block is None or sink_block is None:
            return True

        origin_inst = instructions_by_index.get(origin_idx)
        sink_inst = instructions_by_index.get(sink_idx)
        origin_context = self._context_key_for_instruction(origin_inst, origin_block)
        sink_context = self._context_key_for_instruction(sink_inst, sink_block)
        if origin_context != sink_context:
            return self._keep_cross_context_dataflow_pair_after_guarded_callref(
                origin_idx=origin_idx,
                sink_idx=sink_idx,
                origin_context=origin_context,
                sink_context=sink_context,
                block_map=block_map,
                instr_to_block=instr_to_block,
                predecessor_map=predecessor_map,
                guard_indices=guard_indices,
                instructions_by_index=instructions_by_index,
                context_instruction_indices=context_instruction_indices,
                continuations=continuations or {},
            )

        # Traversal starts at origin_idx, so a same-block sink appearing earlier
        # cannot be disproven by forward replay.
        if origin_block_id == sink_block_id and sink_idx <= origin_idx:
            return True

        sender_guarded, truncated = is_block_guarded_on_all_predecessor_paths(
            block_map=block_map,
            predecessor_map=predecessor_map,
            start_block_id=origin_block_id,
            stop_before_idx=origin_idx,
            guard_indices=guard_indices,
            max_iterations=MAX_SEARCH_ITERATIONS,
        )
        if sender_guarded and not truncated:
            return False

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=origin_block_id,
            start_from_idx=origin_idx,
            guard_indices=guard_indices,
            sink_indices={sink_idx},
            max_iterations=MAX_SEARCH_ITERATIONS,
            flag_unknown_unguarded=False,
        )
        if result.truncated:
            return True

        return sink_idx in result.flagged_indices

    @staticmethod
    def _get_continuation_parent_site(
        continuations: Dict[str, Any],
        continuation_id: str,
    ) -> Tuple[Optional[str], Optional[int]]:
        """Return the parent context and global parent instruction index."""
        continuation = continuations.get(continuation_id)
        if continuation is None:
            return None, None

        if hasattr(continuation, "parent_context"):
            parent_context = continuation.parent_context
            parent_index = continuation.parent_instruction_index
        elif isinstance(continuation, dict):
            parent_context = continuation.get("parent_context")
            parent_index = continuation.get("parent_instruction_index")
        else:
            return None, None

        return parent_context or "main", parent_index

    def _keep_cross_context_dataflow_pair_after_guarded_callref(
        self,
        origin_idx: int,
        sink_idx: int,
        origin_context: str,
        sink_context: str,
        block_map: Dict[int, BasicBlock],
        instr_to_block: Dict[int, int],
        predecessor_map: Dict[int, Set[int]],
        guard_indices: Set[int],
        instructions_by_index: Dict[int, object],
        context_instruction_indices: Dict[str, List[int]],
        continuations: Dict[str, Any],
    ) -> bool:
        """
        Suppress a narrow class of helper-call false positives.

        Some contracts authenticate the sender in a parent continuation and then
        enter a helper continuation via CALLREF. Dataflow still links the sender
        read in the parent to sends in the direct child even though the helper is
        only reachable after the guarded callsite. Limit this suppression to the
        direct-parent CALLREF case to avoid re-introducing the broad descendant
        replay that lost true positives.
        """
        if sink_context == "main":
            return True

        parent_context, parent_idx = self._get_continuation_parent_site(
            continuations, sink_context
        )
        if parent_context != origin_context or parent_idx is None:
            return True

        parent_inst = instructions_by_index.get(parent_idx)
        if parent_inst is None:
            return True

        origin_block_id = instr_to_block.get(origin_idx)
        parent_block_id = instr_to_block.get(parent_idx)
        if origin_block_id is None or parent_block_id is None:
            return True

        if parent_inst.opcode == "CALLREF":
            parent_guarded, truncated = is_block_guarded_on_all_predecessor_paths(
                block_map=block_map,
                predecessor_map=predecessor_map,
                start_block_id=parent_block_id,
                stop_before_idx=parent_idx,
                guard_indices=guard_indices,
                max_iterations=MAX_SEARCH_ITERATIONS,
            )
            if truncated or not parent_guarded:
                return True
        elif parent_inst.opcode in self.BRANCH_ENTRY_AUTH_WINDOWS:
            if not self._has_nearby_auth_opcode(
                context_instruction_indices=context_instruction_indices,
                instructions_by_index=instructions_by_index,
                context_id=origin_context,
                instruction_idx=parent_idx,
                window=self.BRANCH_ENTRY_AUTH_WINDOWS[parent_inst.opcode],
            ):
                return True
        else:
            return True

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=origin_block_id,
            start_from_idx=origin_idx,
            guard_indices=guard_indices,
            sink_indices={parent_idx},
            max_iterations=MAX_SEARCH_ITERATIONS,
            flag_unknown_unguarded=False,
        )
        if result.truncated:
            return True

        return parent_idx in result.flagged_indices

    def _has_nearby_auth_opcode(
        self,
        context_instruction_indices: Dict[str, List[int]],
        instructions_by_index: Dict[int, object],
        context_id: str,
        instruction_idx: int,
        window: int = 10,
    ) -> bool:
        """
        Check for a nearby auth opcode before a continuation-entry control site.

        Some branch-entered helper continuations are reached via IFREFELSE after
        a compact sender-auth sequence such as SDEQ/CHKSIGNU/EQUAL. These
        patterns may not synthesize a standalone guard event at the entry site,
        so replay a short lexical look-back within the same continuation.
        """
        indices = context_instruction_indices.get(context_id, [])
        if not indices:
            return False

        try:
            position = indices.index(instruction_idx)
        except ValueError:
            return False

        start = max(0, position - window)
        for idx in reversed(indices[start:position]):
            instruction = instructions_by_index.get(idx)
            if instruction is None:
                continue
            if instruction.opcode in self.SENDER_AUTH_OPCODES or instruction.opcode == "EQUAL":
                return True

        return False

    def _detect_with_cfg(
        self,
        facts: AnalysisFacts,
        sender_reads: List[Event],
        dataflow_sink_indices: Optional[Set[int]] = None,
        tasir_send_indices: Optional[Set[int]] = None,
    ) -> List[Vulnerability]:
        findings = []
        block_map: Dict[int, BasicBlock] = {block.id: block for block in facts.basic_blocks}
        instr_to_block = build_instruction_to_block_map(facts.basic_blocks)
        predecessor_map = build_predecessor_map(facts.basic_blocks)
        instructions_by_index = {inst.index: inst for inst in facts.instructions}

        guard_indices = {event.instruction.index for event in facts.events_of("guard")}
        guard_indices.update(
            collect_auth_guard_indices(facts.basic_blocks, instructions_by_index)
        )
        # Use TASIR-identified send indices if available, fallback to event-based detection
        if tasir_send_indices is not None:
            send_indices = tasir_send_indices
        else:
            send_indices = {event.instruction.index for event in facts.events_of("send")}
        exclude_indices = dataflow_sink_indices or set()
        any_truncated = False
        # Track reported (sender_idx, sink_idx) pairs to avoid duplicates
        reported_pairs: Set[Tuple[int, int]] = set()

        for sender_event in sender_reads:
            sender_idx = sender_event.instruction.index
            start_block_id = instr_to_block.get(sender_idx)

            if start_block_id is None:
                findings.append(
                    self._build_vuln(
                        message="Sender is read without a guard/condition before sensitive actions.",
                        instruction=sender_event.instruction,
                        remediation="Validate msg_sender with IF*/THROW* before using it.",
                    )
                )
                continue

            sender_pre_guarded, truncated = is_block_guarded_on_all_predecessor_paths(
                block_map=block_map,
                predecessor_map=predecessor_map,
                start_block_id=start_block_id,
                stop_before_idx=sender_idx,
                guard_indices=guard_indices,
                max_iterations=MAX_SEARCH_ITERATIONS,
            )
            if truncated:
                any_truncated = True
            if sender_pre_guarded:
                continue

            flagged, truncated = self._search_paths(
                block_map,
                start_block_id,
                sender_idx,
                guard_indices,
                send_indices,
            )
            if truncated:
                any_truncated = True
            # Exclude indices already reported by dataflow analysis to avoid duplicates
            flagged = flagged - exclude_indices
            for idx in flagged:
                pair = (sender_idx, idx)
                if pair not in reported_pairs:
                    reported_pairs.add(pair)
                    findings.append(
                        self._build_vuln(
                            message="Sender is read without guard on some control-flow path before sensitive actions.",
                            instruction=facts.instructions[idx],
                            remediation="Ensure msg_sender is validated along every reachable path.",
                        )
                    )

        if any_truncated:
            findings.append(
                self._build_vuln(
                    message=f"Analysis truncated after {MAX_SEARCH_ITERATIONS} iterations; some paths may not be fully analyzed.",
                    instruction=facts.instructions[0] if facts.instructions else None,
                    remediation="Manual verification recommended for complex control flow.",
                    severity="low",
                )
            )

        return findings

    def _search_paths(
        self,
        block_map: Dict[int, BasicBlock],
        start_block_id: int,
        sender_idx: int,
        guard_indices: Set[int],
        send_indices: Set[int],
    ) -> Tuple[Set[int], bool]:
        """Search paths for unguarded sends using common CFG traversal utility.

        Returns:
            Tuple of (flagged_indices, truncated) where truncated indicates
            if the analysis was stopped due to iteration limit.
        """
        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=start_block_id,
            start_from_idx=sender_idx,
            guard_indices=guard_indices,
            sink_indices=send_indices,
            max_iterations=MAX_SEARCH_ITERATIONS,
            flag_unknown_unguarded=True,
            unknown_flag_idx=sender_idx,
        )

        return result.flagged_indices, result.truncated

    @staticmethod
    def _find_guard_after(sender_event: Event, guard_events: List[Event]) -> Optional[Event]:
        for guard in guard_events:
            if guard.instruction.index > sender_event.instruction.index:
                return guard
        return None

    @staticmethod
    def _find_next_event(sender_event: Event, events: List[Event]) -> Optional[Event]:
        for event in events:
            if event.instruction.index > sender_event.instruction.index:
                return event
        return None


    def _detect_with_tasir(
        self, facts: AnalysisFacts, sender_reads: List[Event]
    ) -> Optional[Set[int]]:
        """Use TASIR for semantic sink identification via InstructionKind.

        This method uses TASIR's InstructionKind.SEND_MESSAGE for more robust
        sink identification compared to opcode string matching.

        Args:
            facts: Analysis facts containing TASIR module if available
            sender_reads: List of sender_read events to cross-reference

        Returns:
            Set of send instruction indices identified via TASIR,
            or None to signal fallback to traditional detection.
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        # Import at runtime to avoid circular imports
        from ..ir.tasir_types import InstructionKind

        # Collect SEND_MESSAGE instruction indices using semantic classification
        send_indices: Set[int] = set()
        for instr in module.all_instructions():
            if instr.kind == InstructionKind.SEND_MESSAGE:
                send_indices.add(instr.index)

        # If no sends found via TASIR, return empty set (not None)
        # This indicates TASIR was available but found no sinks
        if not send_indices:
            return set()

        # Use taint flow queries for precision
        confirmed_unguarded_sends: Set[int] = set()
        sender_events = module.events_of("sender_read")
        send_events = module.events_of("send")
        for sender_evt in sender_events:
            for send_evt in send_events:
                if module.get_taint_flow(sender_evt.instruction.index, send_evt.instruction.index):
                    if not module.get_guarded_taint_at(send_evt.instruction.index):
                        # Confirmed: taint flows from sender to send without guard
                        confirmed_unguarded_sends.add(send_evt.instruction.index)

        # Store confirmed sends in module metadata for downstream use
        if confirmed_unguarded_sends:
            if not hasattr(module, '_analysis_metadata'):
                module._analysis_metadata = {}
            module._analysis_metadata['confirmed_unguarded_sends'] = confirmed_unguarded_sends

        return send_indices

    def _detect_cross_function(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """Cross-function unchecked sender detection via inter_procedural view.

        Uses inter_procedural taint_paths to find cases where sender is read
        in one function and an unguarded send occurs in another.
        """
        findings: List[Vulnerability] = []
        module = self.get_tasir(facts)
        if not module or not module.inter_procedural:
            return findings

        # Use helper methods instead of direct access
        paths = module.inter_procedural.get_cross_function_taint_paths()
        for source_method, _, sink_method, _ in paths:
            if (module.inter_procedural.function_has_property(source_method, "has_sender_read")
                    and not module.inter_procedural.function_has_property(source_method, "has_guard")
                    and module.inter_procedural.function_has_property(sink_method, "has_send")
                    and not module.inter_procedural.function_has_property(sink_method, "has_guard")):
                # Find a representative instruction for the finding
                func = module.get_function(sink_method)
                if func:
                    send_insts = [
                        i for block in func.blocks.values()
                        for i in block.instructions
                        if SemanticLabel.MESSAGE_SEND in getattr(i, 'semantic_labels', [])
                    ]
                    if send_insts:
                        si = send_insts[0]
                        inst_by_idx = {i.index: i for i in facts.instructions}
                        target_inst = inst_by_idx.get(si.index)
                        if target_inst:
                            findings.append(self._build_vuln(
                                message=(
                                    f"Cross-function: sender read in method {source_method}, "
                                    f"unguarded send in method {sink_method} at index {si.index}"
                                ),
                                instruction=target_inst,
                                confidence=Confidence.MEDIUM,
                                extra={
                                    "cross_function": True,
                                    "source_method": source_method,
                                    "sink_method": sink_method,
                                },
                            ))
        return findings
