"""Bad randomness detector"""
from typing import List

from ..analyzer.facts import AnalysisFacts
from ..config import CORE_SENSITIVE_OPCODES
from ..ir.tasir_types import SemanticLabel
from .base import Detector
from .results import Vulnerability


class BadRandomnessDetector(Detector):
    """Detects potentially insecure randomness usage.

    Detection Patterns:
    1. Random value generation followed by sensitive operations (SENDRAWMSG, etc.)
       - Randomness may influence fund distribution and can be predicted/manipulated

    2. Weak random seeds from predictable sources (NOW, BLOCKLT, LTIME)
       - These values are known to validators and can be manipulated
    """

    name = "bad_randomness"
    category = "security"
    enabled_by_default = True
    default_severity = "high"
    description = "Detects insecure randomness that could be exploited"

    # Random number generation opcodes
    RANDOM_OPCODES = {"RANDU256", "RAND", "SETRAND", "ADDRAND", "RANDSEED"}

    # Weak random seed sources (predictable by validators/miners)
    WEAK_SEED_OPCODES = {"NOW", "BLOCKLT", "LTIME"}

    # Sensitive operations that should not depend on predictable randomness
    # Derived from CORE_SENSITIVE_OPCODES (config.py) - single source of truth
    SENSITIVE_OPCODES = set(CORE_SENSITIVE_OPCODES)

    def detect(self, facts: AnalysisFacts) -> List[Vulnerability]:
        findings = []
        module = self.get_tasir(facts)

        if module is None:
            return self._detect_simple(facts)

        for func in module.functions.values():
            findings.extend(self._check_function(func))

        # Continuation blocks not mapped to any function are stored in global_blocks.
        # Scan them as independent contexts to avoid TASIR-only false negatives.
        context_to_insts = {}
        for block in getattr(module, "global_blocks", []):
            context_to_insts.setdefault(block.context_id, []).extend(block.instructions)
        for insts in context_to_insts.values():
            findings.extend(self._check_instruction_list(sorted(insts, key=lambda i: i.index)))

        return findings

    def _check_function(self, func) -> List[Vulnerability]:
        """Check a single function for randomness issues."""
        return self._check_instruction_list(list(func.all_instructions()))

    def _check_instruction_list(self, all_insts) -> List[Vulnerability]:
        """Check an instruction list (function or global continuation context)."""
        findings = []

        # Find random number generation points
        random_insts = [inst for inst in all_insts if inst.opcode in self.RANDOM_OPCODES]

        # Use SemanticLabel for more precise detection
        random_insts_by_label = [inst for inst in all_insts
                                 if SemanticLabel.RANDOM_GENERATE in getattr(inst, 'semantic_labels', [])]
        seed_insts_by_label = [inst for inst in all_insts
                               if SemanticLabel.RANDOM_SEED in getattr(inst, 'semantic_labels', [])]
        # Merge label-based results with opcode-based (union, deduplicated by index)
        seen_indices = {inst.index for inst in random_insts}
        for inst in random_insts_by_label:
            if inst.index not in seen_indices:
                random_insts.append(inst)
                seen_indices.add(inst.index)

        # Find sensitive operations (exclude POPCTR — register management, not security-sensitive in randomness context)
        RANDOMNESS_SENSITIVE = self.SENSITIVE_OPCODES - {"POPCTR"}
        sensitive_insts = [inst for inst in all_insts if inst.opcode in RANDOMNESS_SENSITIVE]

        # Find weak seed sources
        weak_seed_insts = [inst for inst in all_insts if inst.opcode in self.WEAK_SEED_OPCODES]
        # Merge seed label results
        seen_seed_indices = {inst.index for inst in weak_seed_insts}
        for inst in seed_insts_by_label:
            if inst.index not in seen_seed_indices:
                weak_seed_insts.append(inst)
                seen_seed_indices.add(inst.index)

        # Pattern 1: Random + Sensitive operation
        if random_insts and sensitive_insts:
            for rand_inst in random_insts:
                # Simple heuristic: if random value is generated before sensitive operation
                for sens_inst in sensitive_insts:
                    if rand_inst.index < sens_inst.index:
                        findings.append(
                            self._build_vuln(
                                message=f"Random value (index {rand_inst.index}) may influence "
                                f"sensitive operation '{sens_inst.opcode}' (index {sens_inst.index})",
                                instruction=self._make_instruction(rand_inst),
                                remediation="Ensure randomness cannot be predicted or manipulated by miners/validators",
                                severity="high",
                            )
                        )
                        break  # Report each random instruction only once

        # Pattern 2: Weak random seed
        setrand_indices = {inst.index for inst in all_insts if inst.opcode.upper() in {"SETRAND", "ADDRAND"}}
        for weak_inst in weak_seed_insts:
            if any(idx > weak_inst.index for idx in setrand_indices):
                findings.append(
                    self._build_vuln(
                        message=f"Weak randomness: '{weak_inst.opcode}' (index {weak_inst.index}) "
                        f"may be used as random seed",
                        instruction=self._make_instruction(weak_inst),
                        remediation="Avoid using predictable values (NOW, BLOCKLT) as random seeds",
                        severity="medium",
                    )
                )

        return findings

    def _detect_simple(self, facts: AnalysisFacts) -> List[Vulnerability]:
        """Simple detection without TASIR."""
        findings = []

        random_insts = [inst for inst in facts.instructions if inst.opcode in self.RANDOM_OPCODES]
        sensitive_insts = [inst for inst in facts.instructions if inst.opcode in self.SENSITIVE_OPCODES]
        weak_seed_insts = [inst for inst in facts.instructions if inst.opcode in self.WEAK_SEED_OPCODES]

        # Pattern 1: Random + Sensitive operation
        if random_insts and sensitive_insts:
            findings.append(
                self._build_vuln(
                    message="Random value generation found with sensitive operations - verify randomness security",
                    instruction=random_insts[0],
                    remediation="Ensure randomness cannot be predicted or manipulated",
                    severity="high",
                )
            )

        # Pattern 2: Weak random seed
        for weak_inst in weak_seed_insts:
            has_setrand = any(
                inst.opcode in {"SETRAND", "ADDRAND"} and inst.index > weak_inst.index
                for inst in facts.instructions
            )
            if has_setrand:
                findings.append(
                    self._build_vuln(
                        message=f"Weak randomness: '{weak_inst.opcode}' may be used as random seed",
                        instruction=weak_inst,
                        remediation="Avoid using predictable values as random seeds",
                        severity="medium",
                    )
                )

        return findings

    def _make_instruction(self, tasir_inst):
        """Create a compatible instruction object from TVMInstruction."""
        from ..analyzer.facts import InstructionFact

        # Create a mock instruction
        class MockInstruction:
            def __init__(self, name, args):
                self.name = name
                self.args = args

        return InstructionFact(
            instruction=MockInstruction(
                tasir_inst.opcode,
                list(getattr(tasir_inst, "original_args", []) or []),
            ),
            index=tasir_inst.index,
        )
