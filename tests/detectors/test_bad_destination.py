from tasmscan.analyzer.facts import AnalysisFacts, BasicBlock, Event, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.bad_destination import BadDestinationAddressDetector
from tasmscan.ir.ir_builder import IRBuilder


class MockInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class MockArg:
    def __init__(self, value):
        self.value = value


def make_instruction_fact(opcode: str, index: int) -> InstructionFact:
    return InstructionFact(instruction=MockInstruction(opcode), index=index)


def test_guard_between_linear_indices_but_off_path_is_not_treated_as_guarded():
    detector = BadDestinationAddressDetector()
    source = make_instruction_fact("LDMSGADDR", 0)
    guard = make_instruction_fact("THROWIFNOT", 5)
    sink = make_instruction_fact("SENDRAWMSG", 10)

    facts = AnalysisFacts(
        instructions=[source, guard, sink],
        events=[Event(type="guard", instruction=guard)],
        basic_blocks=[
            BasicBlock(id=0, instruction_indices=[0], successors=[1, 2]),
            BasicBlock(id=1, instruction_indices=[10], successors=[]),  # sink path
            BasicBlock(id=2, instruction_indices=[5], successors=[]),   # off-path guard
        ],
    )

    assert detector._is_guarded_between(facts, source_idx=0, sink_idx=10) is False


def test_guard_detection_falls_back_to_linear_when_cfg_missing():
    detector = BadDestinationAddressDetector()
    source = make_instruction_fact("LDMSGADDR", 0)
    guard = make_instruction_fact("THROWIFNOT", 5)
    sink = make_instruction_fact("SENDRAWMSG", 10)

    facts = AnalysisFacts(
        instructions=[source, guard, sink],
        events=[Event(type="guard", instruction=guard)],
        basic_blocks=[],
    )

    assert detector._is_guarded_between(facts, source_idx=0, sink_idx=10) is True


def test_bad_destination_ignores_cross_context_builder_chains():
    detector = BadDestinationAddressDetector(min_complexity_threshold=0)

    class _Cont:
        def __init__(self, instructions):
            self.instructions = instructions

    class _Arg:
        def __init__(self, value):
            self.value = value

    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("NEWC"),
            MockInstruction("STREF2CONST"),
            MockInstruction("STREF2CONST"),
            MockInstruction("STREF2CONST"),
            MockInstruction("ENDC"),
            MockInstruction("PUSHCONT", [_Arg(_Cont([MockInstruction("SENDRAWMSG")]))]),
            MockInstruction("EXECUTE"),
        ],
        cell=None,
    )

    fallback_findings = detector.detect(facts)
    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)

    assert fallback_findings == []
    assert tasir_findings == []


def test_bad_destination_reports_single_overflow_finding_per_send_chain():
    detector = BadDestinationAddressDetector(min_complexity_threshold=0)

    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("NEWC"),
            MockInstruction("STREF2CONST"),
            MockInstruction("STREF2CONST"),
            MockInstruction("STREF2CONST"),
            MockInstruction("ENDC"),
            MockInstruction("SENDRAWMSG"),
        ],
        cell=None,
    )

    fallback_findings = detector.detect(facts)
    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert fallback_findings[0].extra.get("signal") == "cell_overflow"
    assert tasir_findings[0].extra.get("signal") == "cell_overflow"


def test_bad_destination_legacy_parses_structured_immediate_for_overflow():
    detector = BadDestinationAddressDetector(min_complexity_threshold=0)

    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("NEWC"),
            MockInstruction("STU", [MockArg(2048)]),
            MockInstruction("ENDC"),
            MockInstruction("SENDRAWMSG"),
        ],
        cell=None,
    )

    # Legacy path: do not attach tasir_module
    fallback_findings = detector.detect(facts)

    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert fallback_findings[0].extra.get("total_bits") == 2048
    assert tasir_findings[0].extra.get("total_bits") == 2048
