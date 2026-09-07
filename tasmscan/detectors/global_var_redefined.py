"""Global variable redefined detector

Detects global variables that may be unintentionally redefined:
1. Cross-function: same global written by multiple functions without reading first
2. Intra-function: same global written multiple times in the same function
3. Pattern: SETGLOB count significantly exceeds GETGLOB for same register

Corresponds to TONScanner's "GobalVarRedefined" vulnerability class.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from ..analyzer.facts import AnalysisFacts
from .base import Detector
from .results import Confidence, Vulnerability


class GlobalVarRedefinedDetector(Detector):
    """Detects potentially unsafe global variable redefinitions."""

    name = "global_var_redefined"
    category = "code_quality"
    enabled_by_default = False  # Disabled: 0% precision on Full-Corpus audit (all FP from save_data pattern)
    default_severity = "medium"
    description = "Detects global variables that may be redefined unsafely"
    tags = ("state-management", "tonscanner-compat")

    GETGLOB_OPCODES: Set[str] = {"GETGLOB"}
    SETGLOB_OPCODES: Set[str] = {"SETGLOB"}
    # c4=data, c5=actions are expected to be written frequently
    SKIP_REGISTERS: Set[str] = {"c0", "c1", "c4", "c5"}

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        findings = []
        module = self.get_tasir(facts)

        if module is not None:
            findings.extend(self._check_tasir(module, facts))

        # Always also run opcode-level
        findings.extend(self._detect_opcode(facts))

        # Deduplicate by global id
        seen = set()
        deduped = []
        for f in findings:
            key = f.extra.get("global_id", id(f))
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped

    def _check_tasir(self, module, facts) -> List[Vulnerability]:
        """Check across and within TASIR functions."""
        findings = []

        # Track per-function global access
        glob_writers: Dict[str, List[Tuple[str, object]]] = defaultdict(list)
        glob_readers: Dict[str, Set[str]] = defaultdict(set)

        for func_name, func in module.functions.items():
            all_insts = list(func.all_instructions())
            func_writes: Dict[str, List] = defaultdict(list)
            func_reads: Set[str] = set()

            for inst in all_insts:
                glob_id = self._extract_glob_id(inst)
                if glob_id is None or glob_id in self.SKIP_REGISTERS:
                    continue
                if inst.opcode in self.GETGLOB_OPCODES:
                    func_reads.add(glob_id)
                elif inst.opcode in self.SETGLOB_OPCODES:
                    func_writes[glob_id].append(inst)

            # Intra-function: 3+ writes to same global (2 is normal: init + save)
            for glob_id, writes in func_writes.items():
                if len(writes) >= 3:
                    instruction = self._find_original_instruction(facts, writes[0].index)
                    findings.append(self._build_vuln(
                        message=f"Global variable '{glob_id}' written {len(writes)} times in "
                                f"function '{func_name}'",
                        instruction=instruction,
                        severity="medium",
                        confidence=Confidence.MEDIUM,
                        extra={"global_id": glob_id, "function": func_name,
                               "write_count": len(writes), "detection_method": "tasir_intra"},
                    ))

            # Collect for cross-function analysis
            for glob_id, writes in func_writes.items():
                glob_writers[glob_id].append((func_name, writes[0]))
                if glob_id in func_reads:
                    glob_readers[glob_id].add(func_name)

        # Cross-function: written by 3+ functions (2 is normal: handler + save_data)
        for glob_id, writers in glob_writers.items():
            if len(writers) >= 3:
                blind = [(fn, inst) for fn, inst in writers if fn not in glob_readers.get(glob_id, set())]
                if blind:
                    instruction = self._find_original_instruction(facts, writers[0][1].index)
                    func_names = [fn for fn, _ in writers]
                    findings.append(self._build_vuln(
                        message=f"Global '{glob_id}' written by {len(writers)} functions "
                                f"({', '.join(str(fn) for fn in func_names[:3])})",
                        instruction=instruction,
                        severity="medium",
                        confidence=Confidence.MEDIUM,
                        extra={"global_id": glob_id, "functions": func_names,
                               "detection_method": "tasir_cross"},
                    ))

        return findings

    def _detect_opcode(self, facts) -> List[Vulnerability]:
        """Opcode-level detection: SETGLOB imbalance."""
        findings = []
        instructions = facts.instructions

        # Count per-register
        get_counts: Counter = Counter()
        set_counts: Counter = Counter()
        set_insts: Dict[str, list] = defaultdict(list)

        for inst in instructions:
            if inst.opcode == "GETGLOB":
                gid = self._extract_glob_id_from_fact(inst)
                if gid and gid not in self.SKIP_REGISTERS:
                    get_counts[gid] += 1
            elif inst.opcode == "SETGLOB":
                gid = self._extract_glob_id_from_fact(inst)
                if gid and gid not in self.SKIP_REGISTERS:
                    set_counts[gid] += 1
                    set_insts[gid].append(inst)

        for gid, sc in set_counts.items():
            gc = get_counts.get(gid, 0)
            # Multiple writes: potential redefinition (5+ to reduce FP)
            if sc >= 5:
                findings.append(self._build_vuln(
                    message=f"Global variable '{gid}' written {sc} times (read {gc} times)",
                    instruction=set_insts[gid][0],
                    severity="medium" if sc > gc else "low",
                    confidence=Confidence.LOW,
                    extra={"global_id": gid, "set_count": sc, "get_count": gc,
                           "detection_method": "opcode"},
                ))

        return findings

    def _extract_glob_id(self, inst) -> str | None:
        if inst.opcode in ("GETGLOB", "SETGLOB"):
            args = getattr(inst, "original_args", []) or getattr(inst, "args", [])
            if args:
                val = getattr(args[0], "value", args[0])
                return f"g{val}"
        return None

    def _extract_glob_id_from_fact(self, inst) -> str | None:
        args = getattr(getattr(inst, "instruction", None), "args", [])
        if args:
            val = getattr(args[0], "value", args[0])
            return f"g{val}"
        return None

    def _find_original_instruction(self, facts, index):
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        return facts.instructions[0] if facts.instructions else None
