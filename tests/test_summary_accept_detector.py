from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.summary_accept import SummarySendRequiresAcceptDetector
from tasmscan.ir.ir_builder import IRBuilder


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


class MockContinuation:
    def __init__(self, instructions):
        self.instructions = instructions


class MockArg:
    def __init__(self, value):
        self.value = value


def _run_detector_with_and_without_tasir(instructions):
    facts = ProgramAnalyzer().analyze(instructions, cell=None)
    detector = SummarySendRequiresAcceptDetector(min_complexity_threshold=0)

    fallback_findings = detector.detect(facts)

    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)

    return fallback_findings, tasir_findings


def test_summary_accept_tasir_respects_accept_in_concrete_caller_context():
    fallback_findings, tasir_findings = _run_detector_with_and_without_tasir(
        [
            MockInstruction("ACCEPT"),
            MockInstruction(
                "PUSHCONT",
                [MockArg(MockContinuation([MockInstruction("SENDRAWMSG")]))],
            ),
            MockInstruction("EXECUTE"),
        ]
    )

    assert fallback_findings == []
    assert tasir_findings == []


def test_summary_accept_tasir_keeps_unguarded_send_detection():
    fallback_findings, tasir_findings = _run_detector_with_and_without_tasir(
        [
            MockInstruction(
                "PUSHCONT",
                [MockArg(MockContinuation([MockInstruction("SENDRAWMSG")]))],
            ),
            MockInstruction("EXECUTE"),
        ]
    )

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
