from tasmscan.analyzer.facts import AnalysisFacts, Event, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.no_accept import NoAcceptBeforeSendDetector


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


def _build_cfg(cfg_builder, instructions):
    return cfg_builder.build_cfg_with_continuations(
        instructions,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        continuations={},
        cont_fact_map={},
    )


def _build_blocks(cfg_builder, instructions, edges):
    return cfg_builder.build_basic_blocks_with_continuations(
        instructions,
        edges,
        continuations={},
    )


def test_entry_block_selection_with_unknown_branch():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=4),
        InstructionFact(instruction=MockInstruction("IF"), index=2, offset=8),
        InstructionFact(instruction=MockInstruction("SENDRAWMSG"), index=3, offset=12),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)
    blocks = _build_blocks(analyzer._cfg_builder, instructions, edges)

    block_map = {block.id: block for block in blocks}
    entry_id = min(block_map) if block_map else None
    assert entry_id is not None
    assert entry_id >= 0

    events = [Event(type="send", instruction=instructions[3])]
    facts = AnalysisFacts(instructions=instructions, events=events, basic_blocks=blocks)

    detector = NoAcceptBeforeSendDetector()
    findings = detector.detect(facts)

    assert any("accept" in f.message.lower() for f in findings)
