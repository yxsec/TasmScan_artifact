"""
Tests for BadRandomnessDetector.

Tests cover:
- Pattern 1: Random value generation + sensitive operations
- Pattern 2: Weak random seeds (NOW, BLOCKLT, LTIME)
- Both TASIR-based and simple detection paths
"""
from unittest.mock import MagicMock

from tasmscan.analyzer.facts import AnalysisFacts, InstructionFact
from tasmscan.detectors.bad_randomness import BadRandomnessDetector
from tasmscan.ir.tasir_types import (
    InstructionKind,
    TVMBasicBlock,
    TVMFunction,
    TVMInstruction,
    TVMModule,
)


class MockInstruction:
    """Mock instruction."""

    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


def make_instruction_fact(opcode: str, index: int) -> InstructionFact:
    """Create an InstructionFact with given opcode and index."""
    return InstructionFact(
        instruction=MockInstruction(opcode),
        index=index,
    )


def make_facts(instructions: list, metadata=None) -> AnalysisFacts:
    """Create AnalysisFacts with given instructions."""
    return AnalysisFacts(
        instructions=instructions,
        metadata=metadata or {},
    )


class TestBadRandomnessDetectorSimple:
    """Tests for simple detection path (no TASIR)."""

    def test_no_random_opcodes_returns_empty(self):
        """Contract without random opcodes should return no findings."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("SETCP0", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("SENDRAWMSG", 2),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_random_without_sensitive_returns_empty(self):
        """Random opcodes without sensitive operations should return no findings."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("SETCP0", 0),
            make_instruction_fact("RANDU256", 1),
            make_instruction_fact("NOP", 2),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_random_with_sendrawmsg_reports_vulnerability(self):
        """Random + SENDRAWMSG should report high severity vulnerability."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("SETCP0", 0),
            make_instruction_fact("RANDU256", 1),
            make_instruction_fact("SENDRAWMSG", 2),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "high"
        assert "random" in findings[0].message.lower()

    def test_random_with_sendmsg_reports_vulnerability(self):
        """Random + SENDMSG should report vulnerability."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("RAND", 0),
            make_instruction_fact("SENDMSG", 1),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_random_with_rawreserve_reports_vulnerability(self):
        """Random + RAWRESERVE should report vulnerability."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("RANDU256", 0),
            make_instruction_fact("RAWRESERVE", 1),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1

    def test_now_with_setrand_reports_weak_seed(self):
        """NOW followed by SETRAND should report weak seed vulnerability."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("NOW", 0),
            make_instruction_fact("SETRAND", 1),
            make_instruction_fact("NOP", 2),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "medium"
        assert "weak" in findings[0].message.lower()
        assert "NOW" in findings[0].message

    def test_blocklt_with_addrand_reports_weak_seed(self):
        """BLOCKLT followed by ADDRAND should report weak seed vulnerability."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("BLOCKLT", 0),
            make_instruction_fact("ADDRAND", 1),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "BLOCKLT" in findings[0].message

    def test_ltime_with_setrand_reports_weak_seed(self):
        """LTIME followed by SETRAND should report weak seed vulnerability."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("LTIME", 0),
            make_instruction_fact("SETRAND", 1),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert "LTIME" in findings[0].message

    def test_weak_seed_without_setrand_returns_empty(self):
        """Weak seed source without SETRAND/ADDRAND should not report."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("NOW", 0),
            make_instruction_fact("NOP", 1),
            make_instruction_fact("NOP", 2),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_setrand_before_weak_seed_returns_empty(self):
        """SETRAND before weak seed source should not report (wrong order)."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("SETRAND", 0),
            make_instruction_fact("NOW", 1),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        assert findings == []

    def test_multiple_random_opcodes(self):
        """Multiple random opcodes with sensitive ops should report once."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("RANDU256", 0),
            make_instruction_fact("RAND", 1),
            make_instruction_fact("SENDRAWMSG", 2),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        # Simple detection reports once for the pattern
        assert len(findings) == 1

    def test_both_patterns_reported(self):
        """Both random+sensitive and weak seed patterns should be reported."""
        detector = BadRandomnessDetector()
        instructions = [
            make_instruction_fact("NOW", 0),
            make_instruction_fact("SETRAND", 1),
            make_instruction_fact("RANDU256", 2),
            make_instruction_fact("SENDRAWMSG", 3),
        ]
        facts = make_facts(instructions)

        findings = detector.detect(facts)
        # Should have both: weak seed (medium) and random+sensitive (high)
        assert len(findings) == 2
        severities = {f.severity for f in findings}
        assert "high" in severities
        assert "medium" in severities


class TestBadRandomnessDetectorTASIR:
    """Tests for TASIR-based detection path."""

    def _make_tasir_inst(self, opcode: str, index: int):
        """Create a mock TASIR instruction."""
        inst = MagicMock()
        inst.opcode = opcode
        inst.index = index
        inst.operands = []
        inst.offset = 0
        return inst

    def _make_tasir_func(self, instructions):
        """Create a mock TASIR function."""
        func = MagicMock()
        func.all_instructions = MagicMock(return_value=iter(instructions))
        return func

    def _make_tasir_module(self, functions):
        """Create a mock TASIR module."""
        module = MagicMock()
        module.functions = {f"func_{i}": f for i, f in enumerate(functions)}
        return module

    def test_tasir_random_with_sendrawmsg(self):
        """TASIR: Random + SENDRAWMSG should report vulnerability."""
        detector = BadRandomnessDetector()

        insts = [
            self._make_tasir_inst("NOP", 0),
            self._make_tasir_inst("RANDU256", 1),
            self._make_tasir_inst("SENDRAWMSG", 2),
        ]
        func = self._make_tasir_func(insts)
        module = self._make_tasir_module([func])

        facts = make_facts([], metadata={"tasir_module": module})

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_tasir_weak_seed_pattern(self):
        """TASIR: Weak seed pattern should be detected."""
        detector = BadRandomnessDetector()

        insts = [
            self._make_tasir_inst("NOW", 0),
            self._make_tasir_inst("SETRAND", 1),
        ]
        func = self._make_tasir_func(insts)
        module = self._make_tasir_module([func])

        facts = make_facts([], metadata={"tasir_module": module})

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].severity == "medium"

    def test_tasir_sensitive_before_random_no_report(self):
        """TASIR: Sensitive op before random should not report (random doesn't influence it)."""
        detector = BadRandomnessDetector()

        insts = [
            self._make_tasir_inst("SENDRAWMSG", 0),
            self._make_tasir_inst("RANDU256", 1),
        ]
        func = self._make_tasir_func(insts)
        module = self._make_tasir_module([func])

        facts = make_facts([], metadata={"tasir_module": module})

        findings = detector.detect(facts)
        # Random comes after sensitive op, so no influence
        assert findings == []

    def test_tasir_multiple_functions(self):
        """TASIR: Should check all functions in module."""
        detector = BadRandomnessDetector()

        func1_insts = [
            self._make_tasir_inst("NOP", 0),
        ]
        func2_insts = [
            self._make_tasir_inst("RANDU256", 1),
            self._make_tasir_inst("SENDRAWMSG", 2),
        ]

        func1 = self._make_tasir_func(func1_insts)
        func2 = self._make_tasir_func(func2_insts)
        module = self._make_tasir_module([func1, func2])

        facts = make_facts([], metadata={"tasir_module": module})

        findings = detector.detect(facts)
        assert len(findings) == 1

    def test_tasir_with_real_types_does_not_crash(self):
        """TASIR path should work with real TVMFunction/TVMInstruction types."""
        detector = BadRandomnessDetector()

        block = TVMBasicBlock(
            id=0,
            context_id="recv_internal",
            instructions=[
                TVMInstruction(index=0, kind=InstructionKind.UNKNOWN, opcode="RANDU256"),
                TVMInstruction(index=1, kind=InstructionKind.UNKNOWN, opcode="SENDRAWMSG"),
            ],
        )
        module = TVMModule(
            functions={
                0: TVMFunction(method_id=0, blocks={0: block}, is_recv_internal=True),
            }
        )
        facts = make_facts([], metadata={"tasir_module": module})

        findings = detector.detect(facts)
        assert len(findings) == 1
        assert findings[0].instruction is not None
        assert findings[0].instruction.opcode == "RANDU256"


class TestBadRandomnessDetectorAttributes:
    """Tests for detector attributes and configuration."""

    def test_detector_name(self):
        """Detector should have correct name."""
        detector = BadRandomnessDetector()
        assert detector.name == "bad_randomness"

    def test_detector_category(self):
        """Detector should be in security category."""
        detector = BadRandomnessDetector()
        assert detector.category == "security"

    def test_enabled_by_default(self):
        """Detector should be enabled by default."""
        detector = BadRandomnessDetector()
        assert detector.enabled_by_default is True

    def test_default_severity(self):
        """Default severity should be high."""
        detector = BadRandomnessDetector()
        assert detector.default_severity == "high"

    def test_description_not_empty(self):
        """Detector should have a description."""
        detector = BadRandomnessDetector()
        assert detector.description
        assert len(detector.description) > 0


class TestBadRandomnessDetectorEdgeCases:
    """Edge case tests."""

    def test_empty_instructions(self):
        """Empty instruction list should be handled gracefully."""
        detector = BadRandomnessDetector()
        facts = make_facts([])

        findings = detector.detect(facts)
        assert findings == []

    def test_all_random_opcodes_detected(self):
        """All random opcodes in RANDOM_OPCODES should be recognized."""
        detector = BadRandomnessDetector()

        for opcode in BadRandomnessDetector.RANDOM_OPCODES:
            instructions = [
                make_instruction_fact(opcode, 0),
                make_instruction_fact("SENDRAWMSG", 1),
            ]
            facts = make_facts(instructions)

            findings = detector.detect(facts)
            assert len(findings) >= 1, f"Failed to detect {opcode}"

    def test_all_weak_seed_opcodes_detected(self):
        """All weak seed opcodes should be recognized."""
        detector = BadRandomnessDetector()

        for opcode in BadRandomnessDetector.WEAK_SEED_OPCODES:
            instructions = [
                make_instruction_fact(opcode, 0),
                make_instruction_fact("SETRAND", 1),
            ]
            facts = make_facts(instructions)

            findings = detector.detect(facts)
            assert len(findings) >= 1, f"Failed to detect weak seed from {opcode}"

    def test_all_sensitive_opcodes_detected(self):
        """All sensitive opcodes should trigger detection with random."""
        detector = BadRandomnessDetector()

        for opcode in BadRandomnessDetector.SENSITIVE_OPCODES:
            instructions = [
                make_instruction_fact("RANDU256", 0),
                make_instruction_fact(opcode, 1),
            ]
            facts = make_facts(instructions)

            findings = detector.detect(facts)
            assert len(findings) >= 1, f"Failed to detect sensitive op {opcode}"
