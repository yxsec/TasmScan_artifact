from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.dynamic_call import DynamicCallDetector
from tasmscan.ir.ir_builder import IRBuilder


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


def _build_facts_with_unknown_try(opcode: str):
    instructions = [
        MockInstruction("NOP"),
        MockInstruction("NOP"),
        MockInstruction(opcode),
        MockInstruction("NOP"),
        MockInstruction("NOP"),
        MockInstruction("NOP"),
    ]
    return ProgramAnalyzer().analyze(instructions, cell=None)


def _run_detector_with_and_without_tasir(facts):
    detector = DynamicCallDetector(min_complexity_threshold=0)

    fallback_findings = detector.detect(facts)

    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)

    return fallback_findings, tasir_findings


def test_dynamic_call_detector_tasir_path_keeps_try_unknown_target_finding():
    facts = _build_facts_with_unknown_try("TRY")
    fallback_findings, tasir_findings = _run_detector_with_and_without_tasir(facts)

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert fallback_findings[0].instruction.index == tasir_findings[0].instruction.index
    assert "Dynamic call to unknown target" in tasir_findings[0].message


def test_dynamic_call_detector_tasir_path_keeps_tryargs_unknown_target_finding():
    facts = _build_facts_with_unknown_try("TRYARGS")
    fallback_findings, tasir_findings = _run_detector_with_and_without_tasir(facts)

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert fallback_findings[0].instruction.index == tasir_findings[0].instruction.index
    assert "Dynamic call to unknown target" in tasir_findings[0].message
