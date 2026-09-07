"""Summary-based sender guard detector.

Heuristic note: This detector provides a coarse-grained heuristic that checks if
a block reads the sender address but has no guard operations. It does NOT
verify whether the guard actually checks the sender value. For more precise
sender validation detection, use the `unchecked_sender` detector which
performs data flow analysis.

Use cases for this detector:
- Quick initial scan for potential issues
- Identifying blocks that may need further review
- Catching cases where sender is read but completely unguarded

Limitations:
- May produce false negatives when guard exists but doesn't check sender
- Does not track data flow between sender read and guard
- Works at block summary level, not instruction level
"""
from typing import List, Optional

from ..analyzer.facts import AnalysisFacts
from ..config import GUARD_OPCODES
from ..ir.tasir_types import RegisterLocation, TVMModule
from .base import Detector
from .cfg_utils import (
    build_instruction_to_block_map,
    build_predecessor_map,
    collect_auth_guard_indices,
    is_block_guarded_on_all_predecessor_paths,
)
from .results import Vulnerability


class SummarySenderGuardDetector(Detector):
    """
    Summary-level detector: checks sender reads without guards.

    Heuristic limitation: This detector only verifies that a guard EXISTS in the
    same block as a sender read. It does NOT verify that the guard actually
    checks the sender value. For precise taint tracking, use the
    `unchecked_sender` detector with data flow analysis enabled.

    This is intentionally a low-severity, high-recall detector designed to
    catch obvious cases quickly. False positives are expected and should be
    filtered by downstream analysis.
    """

    name = "summary_sender_guard"
    category = "security"
    default_severity = "low"
    description = "Checks block summaries for sender reads without guards (coarse heuristic)"

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        # Try TASIR-based detection first
        tasir_result = self._detect_with_tasir(facts)
        if tasir_result is not None:
            return tasir_result

        # Fallback to summary-based detection
        findings = []

        for summary in facts.summaries:
            has_sender_read = summary.event_counts.get("sender_read", 0) > 0
            has_guard = summary.event_counts.get("guard", 0) > 0

            if has_sender_read and not has_guard:
                # Find the actual sender_read instruction in this context
                sender_instruction = self._find_sender_read_instruction(
                    facts, summary.context_id
                )
                if sender_instruction is not None:
                    findings.append(
                        self._build_vuln(
                            message=f"Block {summary.context_id} reads sender without guards",
                            instruction=sender_instruction,
                            severity="low",
                            remediation="Add sender validation with IF/THROW instructions.",
                        )
                    )

        return findings

    def _find_sender_read_instruction(self, facts: AnalysisFacts, context_id: str):
        """Find the first sender_read instruction in the specified context.

        Improved handling: Returns None if no sender_read found, instead of returning
        an unrelated instruction. The caller should handle None appropriately.
        """
        # ProgramAnalyzer summary context uses block hash (e.g., "root") for main
        # context summaries. Treat those aliases as main-context lookup to avoid
        # missing sender_read instructions when continuation_id is None.
        summary_main_aliases = {"main", "root"}
        block_hash = facts.metadata.get("block_hash")
        if isinstance(block_hash, str) and block_hash:
            summary_main_aliases.add(block_hash)
        is_main_summary = context_id in summary_main_aliases

        # Find a sender_read event in this context
        for event in facts.events_of("sender_read"):
            instr = event.instruction
            # Check if instruction belongs to this context
            if is_main_summary and instr.continuation_id is None:
                return instr
            if instr.continuation_id == context_id:
                return instr

        # This prevents vulnerability reports pointing to wrong locations
        return None

    def _detect_with_tasir(self, facts: AnalysisFacts) -> Optional[List[Vulnerability]]:
        """
        Attempt TASIR-based detection of sender reads without guards.

        Uses TASIR instruction opcodes aligned with centralized guard policy:
        - Identifies sender_read events from facts
        - Uses config.GUARD_OPCODES for guard detection
        - Provides more precise guard identification than event counting

        Returns:
            List of findings if TASIR detection is conclusive, None to signal fallback.
            Empty list means no issues found via TASIR.
        """
        module = self.get_tasir(facts)
        if module is None:
            return None

        findings = []
        instr_to_block = build_instruction_to_block_map(facts.basic_blocks)
        predecessor_map = build_predecessor_map(facts.basic_blocks)
        block_map = {block.id: block for block in facts.basic_blocks}
        instructions_by_index = {inst.index: inst for inst in facts.instructions}
        guard_indices = {event.instruction.index for event in facts.events_of("guard")}
        guard_indices.update(
            collect_auth_guard_indices(facts.basic_blocks, instructions_by_index)
        )

        # Use taint query methods to enhance guard analysis
        tainted_indices = module.get_tainted_instructions() if module else []
        tainted_set = set(tainted_indices)

        # In TASIR detection, check if writes to security-critical registers are guarded
        if module:
            for inst in module.all_instructions():
                for output in getattr(inst, 'outputs', []):
                    if isinstance(output, RegisterLocation) and hasattr(output, 'security_level'):
                        level = output.security_level
                        if level == "write_critical":  # c5 = output_actions
                            # Check: this security-critical register write on a tainted path?
                            pass

        # Group sender_read events by context
        sender_reads_by_context: dict = {}
        for event in facts.events_of("sender_read"):
            instr = event.instruction
            ctx_id = instr.continuation_id if instr.continuation_id else "main"
            if ctx_id not in sender_reads_by_context:
                sender_reads_by_context[ctx_id] = []
            sender_reads_by_context[ctx_id].append(instr)

        # For each context with sender reads, check for guards using TASIR
        for ctx_id, sender_instrs in sender_reads_by_context.items():
            # Find guard instructions in this context using InstructionKind
            has_guard = self._context_has_guard_tasir(module, ctx_id)
            if not has_guard:
                all_sender_paths_guarded = True
                for sender_instr in sender_instrs:
                    start_block_id = instr_to_block.get(sender_instr.index)
                    if start_block_id is None:
                        all_sender_paths_guarded = False
                        break
                    guarded, _ = is_block_guarded_on_all_predecessor_paths(
                        block_map=block_map,
                        predecessor_map=predecessor_map,
                        start_block_id=start_block_id,
                        stop_before_idx=sender_instr.index,
                        guard_indices=guard_indices,
                    )
                    if not guarded:
                        all_sender_paths_guarded = False
                        break
                if all_sender_paths_guarded:
                    continue

            if not has_guard:
                # Report the first sender_read in this context
                first_sender = sender_instrs[0]
                # Check if sender read is on a tainted path
                sender_on_tainted_path = first_sender.index in tainted_set
                extra = {"detection_method": "tasir"}
                if sender_on_tainted_path:
                    extra["on_tainted_path"] = True
                findings.append(
                    self._build_vuln(
                        message=f"Block {ctx_id} reads sender without guards (TASIR analysis)",
                        instruction=first_sender,
                        severity="low",
                        remediation="Add sender validation with IF/THROW instructions.",
                        extra=extra,
                    )
                )

        return findings

    def _context_has_guard_tasir(
        self,
        module: TVMModule,
        context_id: str,
    ) -> bool:
        """
        Check if a context has guard instructions using TASIR InstructionKind.

        Uses centralized guard opcode definitions to keep TASIR behavior aligned
        with event-based/fallback guard semantics (including IFRET/IFNOTRET and
        THROW* families).

        Args:
            module: The TVMModule containing all blocks and instructions
            context_id: The context ID to check ("main" or continuation_id)

        Returns:
            True if at least one guard instruction exists in the context
        """
        # Iterate through all blocks and check context_id match
        for block in module.all_blocks():
            # TVMBasicBlock has context_id field matching "main" or continuation_id
            if block.context_id != context_id:
                continue

            # Check instructions in this block for guard opcodes.
            # This stays consistent with config.GUARD_OPCODES used by other paths.
            for tasir_inst in block.instructions:
                if tasir_inst.opcode in GUARD_OPCODES:
                    return True

        return False
