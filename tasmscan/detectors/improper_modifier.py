"""Improper function modifier detector.

This detector follows the broader TONScanner notion of IFM:

- resolve a called function,
- confirm that the callee directly or transitively performs one of the
  supported side effects (throw, message send, storage write/code update,
  global write),
- and only then report when the call result is immediately dropped.

The previous opcode-only fallback treated any CALL+DROP inside a contract with
unrelated side effects as IFM. That was the main source of rq5 false positives,
especially on parser helpers and builder-heavy contracts.
"""
from collections import defaultdict
from typing import Dict, List, Set

from ..analyzer.facts import AnalysisFacts
from .base import Detector
from .results import Confidence, Vulnerability


class ImproperModifierDetector(Detector):
    """Detects functions with side effects that may lack impure modifier."""

    name = "improper_modifier"
    category = "code_quality"
    enabled_by_default = True
    default_severity = "medium"
    description = "Detects functions with side effects that may lack the impure modifier"
    tags = ("modifier", "side-effects", "tonscanner-compat")

    CALL_OPCODES: Set[str] = {
        "CALLDICT", "EXECUTE", "CALLXARGS", "CALLCC", "CALLCCARGS",
    }

    DROP_OPCODES: Set[str] = {
        "DROP", "DROP2", "DROPX", "NIP", "BLKDROP", "BLKDROP2",
    }

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        if self._is_trivial_contract(facts):
            return []

        module = self.get_tasir(facts)
        if module is None:
            return []

        from ..ir.tasir_types import InstructionKind

        aliases, transitive_effects = self._build_transitive_effect_index(module, InstructionKind)
        findings = self._check_tasir(module, facts, aliases, transitive_effects)
        findings.extend(self._check_raw_call_drop(facts, aliases, transitive_effects))

        # Deduplicate
        seen = set()
        deduped = []
        for f in findings:
            key = f.extra.get("call_index", f.extra.get("drop_index", id(f)))
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped

    def _check_tasir(self, module, facts, aliases, transitive_effects) -> List[Vulnerability]:
        """TASIR-level IFM check with transitive side-effect summaries."""
        findings = []
        if not transitive_effects:
            return findings

        analysis_incomplete = self._analysis_incomplete(facts, module)

        for func_name, func in module.functions.items():
            all_insts = list(func.all_instructions())
            for i, inst in enumerate(all_insts):
                if inst.opcode not in self.CALL_OPCODES:
                    continue

                drop_inst = self._find_drop_after_call(all_insts, i)
                if drop_inst is None:
                    continue

                callee_id = self._resolve_callee(inst)
                canonical = aliases.get(callee_id) if callee_id is not None else None
                if canonical is None:
                    continue

                effects = transitive_effects.get(canonical, set())
                if not effects:
                    continue

                # Reviewed rq5 samples show that TASIR-only global-write effects under
                # incomplete analysis are dominated by false positives. Keep stronger
                # send/storage/throw patterns, and let raw_call_drop handle the send+throw
                # fallback separately.
                if analysis_incomplete and effects == {"global"}:
                    continue

                instruction = self._find_original_instruction(facts, inst.index)
                findings.append(self._build_vuln(
                    message=f"Function '{canonical}' has broad TONScanner side effects "
                            f"({', '.join(sorted(effects))}) but the call result is dropped",
                    instruction=instruction,
                    severity="medium",
                    confidence=Confidence.MEDIUM,
                    extra={
                        "callee": canonical,
                        "call_index": inst.index,
                        "drop_index": drop_inst.index,
                        "effects": sorted(effects),
                        "detection_method": "tasir",
                    },
                ))

        return findings

    def _check_raw_call_drop(
        self,
        facts: AnalysisFacts,
        aliases: Dict[str, str],
        transitive_effects: Dict[str, Set[str]],
    ) -> List[Vulnerability]:
        """Fallback to raw CALLDICT-like sites when TASIR function layout hides the call."""
        findings = []
        instructions = facts.instructions

        for i, inst in enumerate(instructions):
            if inst.opcode not in self.CALL_OPCODES:
                continue

            callee_id = self._resolve_callee_from_fact(inst)
            canonical = aliases.get(callee_id) if callee_id is not None else None
            if canonical is None:
                continue

            effects = transitive_effects.get(canonical, set())
            if not effects:
                continue

            drop_inst = self._find_drop_after_call(instructions, i)
            if drop_inst is None:
                continue

            # Keep the raw fallback narrow: require the TONScanner-style
            # send+throw mismatch, avoid storage-heavy helper patterns, and
            # suppress noisy stack-juggling drops that dominated reviewed rq5
            # false positives.
            if not {"send", "throw"}.issubset(effects) or "storage" in effects:
                continue
            if not self._is_supported_raw_drop_shape(inst.index, drop_inst):
                continue

            findings.append(self._build_vuln(
                message=f"Function '{canonical}' has broad TONScanner side effects "
                        f"({', '.join(sorted(effects))}) but the call result is dropped",
                instruction=inst,
                severity="medium",
                confidence=Confidence.MEDIUM,
                extra={
                    "callee": canonical,
                    "call_index": inst.index,
                    "drop_index": drop_inst.index,
                    "effects": sorted(effects),
                    "detection_method": "raw_call_drop",
                },
            ))

        return findings

    def _build_transitive_effect_index(self, module, instruction_kind) -> tuple[Dict[str, str], Dict[str, Set[str]]]:
        """Build canonical-name aliases and transitive side-effect sets."""
        canonical_names: Dict[int, str] = {}
        direct_effects: Dict[str, Set[str]] = {}
        edges: Dict[str, Set[str]] = defaultdict(set)
        aliases: Dict[str, str] = {}

        for method_id, func in module.functions.items():
            canonical = func.name or f"method_{method_id}"
            canonical_names[method_id] = canonical
            aliases[canonical] = canonical
            aliases[str(method_id)] = canonical
            direct_effects[canonical] = self._direct_effects(func, instruction_kind)

        for caller_mid, callee_mid in getattr(module, "call_graph", []):
            caller = canonical_names.get(caller_mid)
            callee = canonical_names.get(callee_mid)
            if caller and callee:
                edges[caller].add(callee)

        memo: Dict[str, Set[str]] = {}
        visiting: Set[str] = set()

        def visit(func_id: str) -> Set[str]:
            if func_id in memo:
                return memo[func_id]
            if func_id in visiting:
                return set(direct_effects.get(func_id, set()))
            visiting.add(func_id)
            effects = set(direct_effects.get(func_id, set()))
            for callee in edges.get(func_id, ()):
                effects.update(visit(callee))
            visiting.remove(func_id)
            memo[func_id] = effects
            return effects

        transitive = {func_id: visit(func_id) for func_id in direct_effects}
        return aliases, transitive

    def _direct_effects(self, func, instruction_kind) -> Set[str]:
        """Return direct TONScanner-broad side effects for a function."""
        effects: Set[str] = set()
        for inst in func.all_instructions():
            kind = getattr(inst, "kind", None)
            if kind == instruction_kind.THROW or str(inst.opcode).startswith("THROW"):
                effects.add("throw")
            if kind == instruction_kind.SEND_MESSAGE:
                effects.add("send")
            if kind == instruction_kind.SET_CODE or inst.opcode == "COMMIT":
                effects.add("storage")
            if kind == instruction_kind.GLOBAL_STORE or inst.opcode in {"SETGLOB", "SETGLOBVAR"}:
                effects.add("global")
            if inst.opcode == "POPCTR":
                operands = getattr(inst, "original_args", []) or []
                first = operands[0] if operands else None
                if hasattr(first, "value"):
                    first = first.value
                if first == 4:
                    effects.add("storage")
        return effects

    def _analysis_incomplete(self, facts: AnalysisFacts, module) -> bool:
        facts_meta = facts.metadata if isinstance(facts.metadata, dict) else {}
        if facts_meta.get("analysis_incomplete"):
            return True
        module_meta = getattr(module, "analysis_metadata", None)
        if isinstance(module_meta, dict) and module_meta.get("analysis_incomplete"):
            return True
        graph = getattr(facts, "dataflow_graph", None)
        graph_meta = getattr(graph, "analysis_metadata", None) if graph is not None else None
        return bool(isinstance(graph_meta, dict) and graph_meta.get("analysis_incomplete"))

    def _is_supported_raw_drop_shape(self, call_index: int, drop_inst) -> bool:
        distance = drop_inst.index - call_index
        return drop_inst.opcode in {"DROP", "DROP2"} or distance == 3

    def _find_drop_after_call(self, instructions, call_pos):
        for pos in range(call_pos + 1, min(call_pos + 4, len(instructions))):
            inst = instructions[pos]
            if inst.opcode in self.DROP_OPCODES:
                return inst
            if inst.opcode in self.CALL_OPCODES:
                break
        return None

    def _resolve_callee(self, call_inst) -> str | None:
        args = getattr(call_inst, "original_args", []) or getattr(call_inst, "args", [])
        if args:
            arg0 = args[0]
            if hasattr(arg0, "value"):
                arg0 = arg0.value
            return str(arg0)
        return None

    def _resolve_callee_from_fact(self, fact_inst) -> str | None:
        args = getattr(fact_inst.instruction, "args", []) or []
        if args:
            arg0 = args[0]
            if hasattr(arg0, "value"):
                arg0 = arg0.value
            return str(arg0)
        return None

    def _find_original_instruction(self, facts, index):
        for inst in facts.instructions:
            if inst.index == index:
                return inst
        return facts.instructions[0] if facts.instructions else None
