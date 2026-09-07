"""Unused variable detector

Detects values loaded from messages or storage that are immediately dropped
without being used. In Tact-compiled contracts, unused context fields
(like $ctx'bounced, $ctx'value) produce LD* followed by DROP/NIP patterns.

This complements unchecked_sender by catching a broader class of unused values,
not just sender addresses.

Corresponds to TONScanner's "UncheckReturn" vulnerability class
(which actually means "unused local variable", not "unchecked return value").
"""
from typing import List, Set

from ..analyzer.facts import AnalysisFacts
from .base import Detector
from .results import Confidence, Vulnerability


class UnusedVariableDetector(Detector):
    """Detects loaded values that are immediately dropped (unused variables)."""

    name = "unused_variable"
    category = "code_quality"
    enabled_by_default = True
    default_severity = "low"
    description = "Detects values loaded from messages/storage that are unused"
    tags = ("unused", "variable", "tonscanner-compat")

    LOAD_OPCODES: Set[str] = {
        "LDU", "LDI", "LDSLICE", "LDSLICEX", "LDREF", "LDDICT", "LDOPTREF",
        "LDGRAMS", "LDVARUINT16", "LDVARUINT32", "LDMSGADDR", "LDBITS",
        "PLDU", "PLDI", "PLDREF", "PLDSLICE",
    }

    DROP_OPCODES: Set[str] = {
        "DROP", "DROP2", "NIP", "BLKDROP", "BLKDROP2", "DROPX",
    }

    # Minimum LD->DROP patterns to report (avoid noise)
    MIN_PATTERNS = 3

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        module = self.get_tasir(facts)

        if module is not None:
            findings.extend(self._detect_tasir(module, facts))

        if not findings:
            findings.extend(self._detect_opcode(facts))

        return findings

    def _detect_tasir(self, module, facts) -> List[Vulnerability]:
        """TASIR-level: per-function LD->DROP analysis."""
        findings = []

        for func_name, func in module.functions.items():
            all_insts = list(func.all_instructions())
            ld_drops = self._count_ld_drops(all_insts)

            if ld_drops >= self.MIN_PATTERNS:
                instruction = self._find_original_instruction(facts,
                    all_insts[0].index if all_insts else 0)
                findings.append(self._build_vuln(
                    message=f"Function '{func_name}' has {ld_drops} loaded values "
                            f"that are immediately dropped (unused variables)",
                    instruction=instruction,
                    severity="low",
                    confidence=Confidence.MEDIUM,
                    extra={"function": func_name, "ld_drop_count": ld_drops,
                           "detection_method": "tasir"},
                ))

        return findings

    def _detect_opcode(self, facts) -> List[Vulnerability]:
        """Opcode-level: count LD->DROP patterns across entire contract."""
        findings = []
        instructions = facts.instructions
        ld_drops = self._count_ld_drops_facts(instructions)

        if ld_drops >= self.MIN_PATTERNS:
            findings.append(self._build_vuln(
                message=f"Contract has {ld_drops} loaded values that are immediately "
                        f"dropped (potential unused variables)",
                instruction=instructions[0] if instructions else None,
                severity="low",
                confidence=Confidence.LOW,
                extra={"ld_drop_count": ld_drops, "detection_method": "opcode"},
            ))

        return findings

    def _count_ld_drops(self, all_insts) -> int:
        """Count LD->DROP patterns in TASIR instruction list."""
        count = 0
        for i, inst in enumerate(all_insts):
            if inst.opcode in self.LOAD_OPCODES:
                for j in range(i + 1, min(i + 4, len(all_insts))):
                    next_inst = all_insts[j]
                    if next_inst.opcode in self.DROP_OPCODES:
                        count += 1
                        break
                    if next_inst.opcode in self.LOAD_OPCODES or next_inst.opcode.startswith("ST"):
                        break
        return count

    def _count_ld_drops_facts(self, instructions) -> int:
        """Count LD->DROP patterns in flat instruction list."""
        count = 0
        for i, inst in enumerate(instructions):
            if inst.opcode in self.LOAD_OPCODES:
                for j in range(i + 1, min(i + 4, len(instructions))):
                    if instructions[j].opcode in self.DROP_OPCODES:
                        count += 1
                        break
                    if instructions[j].opcode in self.LOAD_OPCODES or \
                       instructions[j].opcode.startswith("ST"):
                        break
        return count

    def _find_original_instruction(self, facts, index):
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        return facts.instructions[0] if facts.instructions else None
