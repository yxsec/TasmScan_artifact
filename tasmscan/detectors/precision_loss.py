"""Precision loss detector

Detects patterns that may lose integer precision:
1. Division followed by multiplication (div-then-mul)
2. Any arithmetic sequence where truncation occurs before scaling

At bytecode level, FunC/Tact compilers may optimize div/mul into different
opcodes or inline them. We also detect at TASIR level using InstructionKind.

Corresponds to TONScanner's "PrecisionLoss" vulnerability class.
"""
from typing import Dict, List, Set

from ..analyzer.facts import AnalysisFacts
from ..ir.tasir_types import InstructionKind
from .base import Detector
from .results import Confidence, Vulnerability


class PrecisionLossDetector(Detector):
    """Detects division-before-multiplication patterns that lose precision."""

    name = "precision_loss"
    category = "security"
    enabled_by_default = True
    default_severity = "medium"
    description = "Detects integer division followed by multiplication that may lose precision"
    tags = ("arithmetic", "tonscanner-compat")

    # Division opcodes (arithmetic division only)
    # Excludes RSHIFT (serialization) and DIVMOD (compiler-generated quotient+remainder)
    DIV_OPCODES: Set[str] = {
        "DIV", "DIVC", "DIVR",
        "MODPOW2",
    }

    # DIVMOD variants: used by compilers for quotient+remainder decomposition,
    # included in absence heuristic but excluded from Pattern 1 proximity check
    DIVMOD_OPCODES: Set[str] = {
        "DIVMOD", "DIVMODC", "DIVMODR",
    }

    # Shift-based division (often compiler-generated for serialization)
    SHIFT_DIV_OPCODES: Set[str] = {
        "RSHIFT", "RSHIFTR", "RSHIFTC",
        "RSHIFTMOD",
    }

    # Multiplication opcodes (arithmetic only, excludes LSHIFT)
    MUL_OPCODES: Set[str] = {
        "MUL", "MULCONST",
    }

    # Shift-based multiplication
    SHIFT_MUL_OPCODES: Set[str] = {
        "LSHIFT", "LSHIFTADDDIVMOD",
    }

    # All division-like opcodes (for absence heuristic)
    ALL_DIV_OPCODES: Set[str] = DIV_OPCODES | DIVMOD_OPCODES | SHIFT_DIV_OPCODES

    # All multiplication-like opcodes
    ALL_MUL_OPCODES: Set[str] = MUL_OPCODES | SHIFT_MUL_OPCODES

    # Safe fused opcodes
    SAFE_FUSED_OPCODES: Set[str] = {
        "MULDIV", "MULDIVR", "MULDIVC",
        "MULDIVMOD", "MULDIVMODR", "MULDIVMODC",
        "MULRSHIFT", "MULRSHIFTR", "MULRSHIFTC",
    }

    MAX_DISTANCE = 12

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        module = self.get_tasir(facts)

        if module is not None:
            # TASIR-level: use InstructionKind.ARITHMETIC
            for func in module.functions.values():
                findings.extend(self._check_tasir_function(func, facts))
            for block in getattr(module, "global_blocks", []):
                findings.extend(self._check_tasir_insts(
                    sorted(block.instructions, key=lambda i: i.index), facts
                ))

        # Always also run opcode-level (catches cases TASIR misses)
        findings.extend(self._detect_opcode_level(facts))

        # Deduplicate by instruction index
        seen = set()
        deduped = []
        for f in findings:
            key = f.extra.get("div_index", id(f))
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    def _check_tasir_function(self, func, facts):
        return self._check_tasir_insts(list(func.all_instructions()), facts)

    def _check_tasir_insts(self, all_insts, facts):
        """Check TASIR instructions for div-then-mul using InstructionKind.

        Only matches arithmetic DIV→MUL pairs (excludes RSHIFT→LSHIFT
        which are typically compiler-generated serialization patterns).
        """
        findings = []

        arith_insts = [i for i in all_insts if getattr(i, 'kind', None) == InstructionKind.ARITHMETIC]

        # Only arithmetic division (DIV/DIVMOD), not shifts
        div_insts = [i for i in arith_insts if i.opcode in self.DIV_OPCODES and i.opcode not in self.SAFE_FUSED_OPCODES]
        # Arithmetic multiplication + LSHIFT (DIV→LSHIFT is still a precision pattern)
        mul_insts = [i for i in arith_insts if i.opcode in self.ALL_MUL_OPCODES and i.opcode not in self.SAFE_FUSED_OPCODES]

        # Also check non-ARITHMETIC tagged but division-like opcodes
        all_div = [i for i in all_insts if i.opcode in self.DIV_OPCODES and i.opcode not in self.SAFE_FUSED_OPCODES]
        all_mul = [i for i in all_insts if i.opcode in self.ALL_MUL_OPCODES and i.opcode not in self.SAFE_FUSED_OPCODES]

        div_insts = list({i.index: i for i in div_insts + all_div}.values())
        mul_insts = list({i.index: i for i in mul_insts + all_mul}.values())

        reported = set()
        for div_inst in div_insts:
            if div_inst.index in reported:
                continue
            for mul_inst in mul_insts:
                dist = mul_inst.index - div_inst.index
                if 0 < dist <= self.MAX_DISTANCE:
                    reported.add(div_inst.index)
                    instruction = self._find_original_instruction(facts, div_inst.index)
                    findings.append(self._build_vuln(
                        message=f"Potential precision loss: '{div_inst.opcode}' (index {div_inst.index}) "
                                f"followed by '{mul_inst.opcode}' (index {mul_inst.index})",
                        instruction=instruction,
                        remediation="Use fused MULDIV/MULDIVR or reorder to multiply before dividing",
                        severity="medium",
                        confidence=Confidence.MEDIUM,
                        extra={"div_opcode": div_inst.opcode, "div_index": div_inst.index,
                               "mul_opcode": mul_inst.opcode, "mul_index": mul_inst.index},
                    ))
                    break
        return findings

    def _detect_opcode_level(self, facts):
        """Opcode-level detection: arithmetic DIV→MUL pairs + per-function absence check."""
        findings = []
        instructions = facts.instructions

        # Pattern 1: Arithmetic DIV followed by MUL/LSHIFT (excludes RSHIFT→LSHIFT)
        divs = [i for i in instructions if i.opcode in self.DIV_OPCODES and i.opcode not in self.SAFE_FUSED_OPCODES]
        muls = [i for i in instructions if i.opcode in self.ALL_MUL_OPCODES and i.opcode not in self.SAFE_FUSED_OPCODES]

        reported = set()
        for d in divs:
            if d.index in reported:
                continue
            for m in muls:
                dist = m.index - d.index
                if 0 < dist <= self.MAX_DISTANCE:
                    reported.add(d.index)
                    findings.append(self._build_vuln(
                        message=f"Potential precision loss: '{d.opcode}' followed by '{m.opcode}'",
                        instruction=d,
                        severity="medium",
                        confidence=Confidence.MEDIUM,
                        extra={"div_opcode": d.opcode, "div_index": d.index,
                               "mul_opcode": m.opcode, "detection_method": "opcode"},
                    ))
                    break

        # Pattern 2: Per-continuation MULDIV absence check
        # Only flag when the SAME function/continuation has both arithmetic DIV
        # and MUL but no MULDIV — avoids false positives from compiler-generated
        # DIV in unrelated functions (e.g., stdlib, Jetton boilerplate)
        module = self.get_tasir(facts)
        if module is not None and not findings:
            # Collect all code regions: named functions + global blocks
            regions = list(module.functions.values())
            for block in getattr(module, "global_blocks", []):
                regions.append(block)
            for region in regions:
                region_insts = list(region.all_instructions()) if hasattr(region, 'all_instructions') else getattr(region, 'instructions', [])
                region_has_div = any(i.opcode in self.DIV_OPCODES for i in region_insts)
                region_has_mul = any(i.opcode in self.ALL_MUL_OPCODES for i in region_insts)
                region_has_muldiv = any(i.opcode in self.SAFE_FUSED_OPCODES for i in region_insts)
                if region_has_div and region_has_mul and not region_has_muldiv:
                    first_div = next((i for i in region_insts if i.opcode in self.DIV_OPCODES), None)
                    if first_div:
                        instruction = self._find_original_instruction(facts, first_div.index)
                        findings.append(self._build_vuln(
                            message="Contract uses both division and multiplication but never fused MULDIV — potential precision loss",
                            instruction=instruction,
                            severity="low",
                            confidence=Confidence.LOW,
                            extra={"detection_method": "absence_heuristic"},
                        ))
                        break  # One finding per contract is sufficient

        # Fallback: contract-level check using only arithmetic DIV (no shifts)
        # Catches cases where TASIR functions/global_blocks don't cover all
        # instructions (e.g., main entry continuation not lifted as a function)
        if not findings:
            has_arith_div = any(i.opcode in self.DIV_OPCODES for i in instructions)
            has_arith_mul = any(i.opcode in self.ALL_MUL_OPCODES for i in instructions)
            has_muldiv = any(i.opcode in self.SAFE_FUSED_OPCODES for i in instructions)
            if has_arith_div and has_arith_mul and not has_muldiv:
                first_div = next((i for i in instructions if i.opcode in self.DIV_OPCODES), None)
                if first_div:
                    findings.append(self._build_vuln(
                        message="Contract uses both division and multiplication but never fused MULDIV — potential precision loss",
                        instruction=first_div,
                        severity="low",
                        confidence=Confidence.LOW,
                        extra={"detection_method": "absence_heuristic"},
                    ))

        return findings

    def _find_original_instruction(self, facts, index):
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        return facts.instructions[0] if facts.instructions else None
