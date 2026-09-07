"""Inconsistent data detector

Detects contracts where the cell layout read from storage differs from what is
written back. Uses multiple detection strategies:
1. TASIR SemanticLabel STORAGE_READ/STORAGE_WRITE comparison
2. LD* vs ST* opcode count imbalance (broad heuristic)
3. Per-function load/store signature comparison

Corresponds to TONScanner's "InconsistentData" vulnerability class.
"""
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

from ..analyzer.facts import AnalysisFacts
from ..ir.tasir_types import SemanticLabel
from .base import Detector
from .results import Confidence, Vulnerability


class InconsistentDataDetector(Detector):
    """Detects inconsistent get_data/set_data cell layouts."""

    name = "inconsistent_data"
    category = "security"
    enabled_by_default = False  # Disabled: paired-function rewrite too strict (1.7% recall), global counting 0% precision
    default_severity = "medium"
    description = "Detects inconsistent cell layouts between get_data() and set_data()"
    tags = ("storage", "data-integrity", "tonscanner-compat")

    LOAD_OPCODES: Set[str] = {
        "LDU", "LDI", "LDSLICE", "LDSLICEX", "LDREF", "LDDICT", "LDOPTREF",
        "LDGRAMS", "LDVARUINT16", "LDVARUINT32", "LDMSGADDR", "LDBITS",
        "PLDREF", "PLDU", "PLDI", "PLDSLICE", "PLDDICT",
    }
    STORE_OPCODES: Set[str] = {
        "STU", "STI", "STSLICE", "STSLICEX", "STREF", "STDICT", "STOPTREF",
        "STGRAMS", "STVARUINT16", "STVARUINT32", "STSLICER", "STBITS",
        "STZEROES", "STONES",
    }
    STORAGE_READ_OPS: Set[str] = {"PUSHCTR", "GETGLOB"}
    STORAGE_WRITE_OPS: Set[str] = {"POPCTR", "SETGLOB"}
    CELL_OPS: Set[str] = {"CTOS", "ENDC", "NEWC"}

    # Map load opcodes to their store counterparts
    LD_TO_ST = {
        "LDU": "STU", "LDI": "STI", "LDSLICE": "STSLICE",
        "LDREF": "STREF", "LDDICT": "STDICT", "LDOPTREF": "STOPTREF",
        "LDGRAMS": "STGRAMS", "LDVARUINT16": "STVARUINT16",
        "LDVARUINT32": "STVARUINT32", "LDMSGADDR": "STSLICER",
    }

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        module = self.get_tasir(facts)
        if module is not None:
            result = self._detect_paired_functions(module, facts)
            if result:
                return result

        # Fallback: opcode-level paired segment detection
        return self._detect_opcode_paired(facts)

    def _detect_paired_functions(self, module, facts) -> List[Vulnerability]:
        """Compare load_data and save_data functions directly via TASIR."""
        # Find load_data-like functions (read c4 via PUSHCTR then CTOS)
        # and save_data-like functions (build cell then write c4 via POPCTR)
        load_funcs = {}  # func_name -> load signature
        save_funcs = {}  # func_name -> store signature

        for func_name, func in module.functions.items():
            all_insts = list(func.all_instructions())
            opcodes = [i.opcode for i in all_insts]

            # Detect load_data pattern: PUSHCTR (c4) -> CTOS -> LD* sequence
            has_c4_read = "PUSHCTR" in opcodes and "CTOS" in opcodes
            # Detect save_data pattern: ST* sequence -> ENDC -> POPCTR (c4)
            has_c4_write = "POPCTR" in opcodes and ("ENDC" in opcodes or "NEWC" in opcodes)

            if has_c4_read and not has_c4_write:
                # Pure load_data function
                sig = self._extract_load_signature(all_insts)
                if sig:
                    load_funcs[func_name] = sig

            if has_c4_write and not has_c4_read:
                # Pure save_data function
                sig = self._extract_store_signature(all_insts)
                if sig:
                    save_funcs[func_name] = sig

            if has_c4_read and has_c4_write:
                # Combined function (both load and save) — extract both
                load_sig = self._extract_load_signature(all_insts)
                store_sig = self._extract_store_signature(all_insts)
                if load_sig:
                    load_funcs[func_name] = load_sig
                if store_sig:
                    save_funcs[func_name] = store_sig

        if not load_funcs or not save_funcs:
            return []

        # Compare signatures: normalize LD* to ST* equivalents
        for lf_name, load_sig in load_funcs.items():
            normalized_load = []
            for op in load_sig:
                st_equiv = self.LD_TO_ST.get(op, op)
                normalized_load.append(st_equiv)

            for sf_name, store_sig in save_funcs.items():
                if normalized_load == store_sig:
                    continue  # Match — no inconsistency

                # Check for significant structural mismatch (not just count)
                if len(normalized_load) == 0 or len(store_sig) == 0:
                    continue

                # Only flag if both are non-trivial and differ structurally
                if len(normalized_load) >= 2 and len(store_sig) >= 2:
                    if normalized_load != store_sig:
                        instruction = self._find_original_instruction(
                            facts, list(module.functions.values())[0]
                            .all_instructions().__next__().index
                            if module.functions else 0)
                        return [self._build_vuln(
                            message=f"Inconsistent data in function '{sf_name}': "
                                    f"load signature ({len(normalized_load)} ops) "
                                    f"differs from store signature ({len(store_sig)} ops)",
                            instruction=instruction,
                            remediation="Ensure get_data() and set_data() use the same cell layout",
                            severity="medium",
                            confidence=Confidence.MEDIUM,
                            extra={"load_func": str(lf_name), "save_func": str(sf_name),
                                   "load_ops": len(normalized_load), "store_ops": len(store_sig),
                                   "detection_method": "tasir_paired"},
                        )]
        return []

    def _extract_load_signature(self, insts) -> list:
        """Extract ordered load opcode sequence after CTOS."""
        sig = []
        after_ctos = False
        for i in insts:
            if i.opcode == "CTOS":
                after_ctos = True
                continue
            if after_ctos:
                if i.opcode in self.LOAD_OPCODES:
                    sig.append(i.opcode)
                elif i.opcode in ("ENDC", "NEWC", "POPCTR", "RET", "RETALT",
                                   "SETGLOB", "CALLDICT", "CALLREF"):
                    break  # End of load sequence
        return sig

    def _extract_store_signature(self, insts) -> list:
        """Extract ordered store opcode sequence before ENDC."""
        sig = []
        after_newc = False
        for i in insts:
            if i.opcode == "NEWC":
                after_newc = True
                sig = []  # Reset on each NEWC (last one before POPCTR matters)
                continue
            if after_newc:
                if i.opcode in self.STORE_OPCODES:
                    sig.append(i.opcode)
                elif i.opcode in ("ENDC", "POPCTR"):
                    if i.opcode == "POPCTR":
                        return sig  # This is the save_data ENDC->POPCTR
                    # After ENDC, may have POPCTR next
                    continue
        return sig

    def _detect_opcode_paired(self, facts) -> List[Vulnerability]:
        """Fallback: find PUSHCTR c4 -> CTOS and NEWC -> POPCTR c4 segments."""
        instructions = facts.instructions
        if len(instructions) < 10:
            return []

        # Find c4-read segment (PUSHCTR followed by CTOS)
        load_sig = []
        save_sig = []

        i = 0
        while i < len(instructions):
            inst = instructions[i]
            # Detect c4 read: PUSHCTR c4 -> CTOS -> LD* sequence
            if inst.opcode == "CTOS" and i > 0:
                prev = instructions[i - 1]
                if prev.opcode == "PUSHCTR":
                    j = i + 1
                    while j < len(instructions) and instructions[j].opcode in self.LOAD_OPCODES:
                        load_sig.append(instructions[j].opcode)
                        j += 1
                    if load_sig:
                        break
            i += 1

        # Find c4 write: NEWC -> ST* -> ENDC -> POPCTR c4
        for i, inst in enumerate(instructions):
            if inst.opcode == "NEWC":
                seg = []
                for j in range(i + 1, min(i + 30, len(instructions))):
                    if instructions[j].opcode in self.STORE_OPCODES:
                        seg.append(instructions[j].opcode)
                    elif instructions[j].opcode == "POPCTR":
                        if seg:
                            save_sig = seg
                        break
                    elif instructions[j].opcode in ("ENDC",):
                        continue
                    elif instructions[j].opcode not in ("STREF", "NEWC"):
                        break

        if not load_sig or not save_sig:
            return []

        # Normalize and compare
        normalized = [self.LD_TO_ST.get(op, op) for op in load_sig]
        if normalized != save_sig and len(normalized) >= 2 and len(save_sig) >= 2:
            return [self._build_vuln(
                message=f"Inconsistent data in function '0': "
                        f"{len(load_sig)} load ops vs {len(save_sig)} store ops",
                instruction=instructions[0],
                remediation="Ensure get_data() and set_data() use the same cell layout",
                severity="medium",
                confidence=Confidence.LOW,
                extra={"loads": len(load_sig), "stores": len(save_sig),
                       "detection_method": "opcode_paired"},
            )]
        return []

    def _find_original_instruction(self, facts, index):
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        return facts.instructions[0] if facts.instructions else None
