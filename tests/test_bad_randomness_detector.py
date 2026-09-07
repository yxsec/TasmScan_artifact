from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.bad_randomness import BadRandomnessDetector
from tasmscan.ir.ir_builder import IRBuilder


class MockArg:
    def __init__(self, arg_type, value):
        self.type = arg_type
        self.value = value


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


class MockContinuation:
    def __init__(self, instructions):
        self.instructions = instructions


def test_bad_randomness_tasir_detects_issues_in_global_continuation_blocks():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction(
                "PUSHCONT",
                [MockArg("code", MockContinuation([MockInstruction("RANDU256"), MockInstruction("SENDRAWMSG")]))],
            ),
            MockInstruction("CALLX"),
        ],
        cell=None,
    )

    detector = BadRandomnessDetector()
    fallback_findings = detector.detect(facts)
    assert len(fallback_findings) >= 1

    facts.metadata["tasir_module"] = IRBuilder().build_tasir(facts)
    tasir_findings = detector.detect(facts)
    assert len(tasir_findings) >= 1
