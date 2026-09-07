from tasmscan.analyzer.facts import AnalysisFacts, BasicBlock, Event, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.summary_sender_guard import SummarySenderGuardDetector
from tasmscan.ir.ir_builder import IRBuilder


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


class FakeTasirInstruction:
    def __init__(self, opcode, index):
        self.opcode = opcode
        self.index = index
        self.outputs = []


class FakeTasirBlock:
    def __init__(self, context_id, instructions):
        self.context_id = context_id
        self.instructions = instructions


class FakeTasirModule:
    def __init__(self, blocks):
        self._blocks = blocks

    def get_tainted_instructions(self):
        return []

    def all_instructions(self):
        return [inst for block in self._blocks for inst in block.instructions]

    def all_blocks(self):
        return self._blocks


def _detect_with_and_without_tasir(instructions):
    facts = ProgramAnalyzer().analyze(instructions, cell=None)
    detector = SummarySenderGuardDetector(min_complexity_threshold=0)

    fallback_findings = detector.detect(facts)

    module = IRBuilder().build_tasir(facts)
    facts.metadata["tasir_module"] = module
    tasir_findings = detector.detect(facts)

    return fallback_findings, tasir_findings


def test_summary_sender_guard_tasir_path_treats_ifret_as_guard():
    fallback_findings, tasir_findings = _detect_with_and_without_tasir(
        [
            MockInstruction("INMSG_SRC"),
            MockInstruction("IFRET"),
            MockInstruction("NOP"),
        ]
    )

    assert fallback_findings == []
    assert tasir_findings == []


def test_summary_sender_guard_tasir_path_keeps_unguarded_detection():
    fallback_findings, tasir_findings = _detect_with_and_without_tasir(
        [
            MockInstruction("INMSG_SRC"),
            MockInstruction("NOP"),
            MockInstruction("NOP"),
        ]
    )

    assert len(fallback_findings) == 1
    assert len(tasir_findings) == 1
    assert fallback_findings[0].instruction.index == tasir_findings[0].instruction.index


def test_summary_sender_guard_tasir_path_suppresses_sender_with_guarded_predecessor():
    detector = SummarySenderGuardDetector(min_complexity_threshold=0)

    instructions = [
        InstructionFact(MockInstruction("SDEQ"), 0),
        InstructionFact(MockInstruction("THROWIFNOT"), 1),
        InstructionFact(MockInstruction("LDMSGADDR"), 2, continuation_id="cont_0"),
    ]
    events = [Event(type="sender_read", instruction=instructions[2])]
    basic_blocks = [
        BasicBlock(id=0, instruction_indices=[0, 1], successors=[1], context="main"),
        BasicBlock(id=1, instruction_indices=[2], successors=[], context="cont_0"),
    ]
    facts = AnalysisFacts(instructions=instructions, events=events, basic_blocks=basic_blocks)
    facts.metadata["tasir_module"] = FakeTasirModule(
        [
            FakeTasirBlock(
                "main",
                [FakeTasirInstruction("SDEQ", 0), FakeTasirInstruction("THROWIFNOT", 1)],
            ),
            FakeTasirBlock(
                "cont_0",
                [FakeTasirInstruction("LDMSGADDR", 2)],
            ),
        ]
    )

    findings = detector.detect(facts)

    assert findings == []
