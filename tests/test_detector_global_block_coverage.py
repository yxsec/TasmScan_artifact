from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.dict_type_mismatch import DictTypeMismatchDetector
from tasmscan.detectors.lack_end_parse import LackEndParseDetector
from tasmscan.detectors.tlb_structure import TLBStructureViolationDetector
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


def _detect_with_and_without_tasir(detector, instructions):
    facts = ProgramAnalyzer().analyze(instructions, cell=None)
    fallback_findings = detector.detect(facts)

    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)
    return fallback_findings, tasir_findings


def _detect_with_tasir(detector, instructions):
    facts = ProgramAnalyzer().analyze(instructions, cell=None)
    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    return detector.detect(facts)


def test_lack_end_parse_tasir_scans_continuation_global_blocks():
    detector = LackEndParseDetector(min_complexity_threshold=0)
    fallback_findings, tasir_findings = _detect_with_and_without_tasir(
        detector,
        [
            MockInstruction(
                "PUSHCONT",
                [MockArg(MockContinuation([MockInstruction("CTOS"), MockInstruction("LDU")]))],
            ),
            MockInstruction("EXECUTE"),
        ],
    )

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert "end_parse" in tasir_findings[0].message.lower()


def test_tlb_structure_tasir_scans_continuation_global_blocks():
    detector = TLBStructureViolationDetector(min_complexity_threshold=0)
    continuation_body = [MockInstruction("CTOS")] + [MockInstruction("LDU") for _ in range(8)]
    fallback_findings, tasir_findings = _detect_with_and_without_tasir(
        detector,
        [
            MockInstruction("PUSHCONT", [MockArg(MockContinuation(continuation_body))]),
            MockInstruction("EXECUTE"),
        ],
    )

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert "tl-b structure violation" in tasir_findings[0].message.lower()


def test_dict_type_mismatch_tasir_scans_continuation_global_blocks():
    detector = DictTypeMismatchDetector(min_complexity_threshold=0)

    main_findings = _detect_with_tasir(
        detector,
        [
            MockInstruction("CTOS"),
            MockInstruction("DICTGET"),
        ],
    )
    cont_findings = _detect_with_tasir(
        detector,
        [
            MockInstruction(
                "PUSHCONT",
                [MockArg(MockContinuation([MockInstruction("CTOS"), MockInstruction("DICTGET")]))],
            ),
            MockInstruction("EXECUTE"),
        ],
    )

    assert len(main_findings) == 1
    assert len(cont_findings) == 1
    assert "dictionary operation dictget" in cont_findings[0].message.lower()
    assert cont_findings[0].extra.get("context_id") == "cont_0"
