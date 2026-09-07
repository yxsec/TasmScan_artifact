from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.ir.dataflow import DataFlowAnalyzer


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args if args is not None else []


class MockContinuation:
    def __init__(self, instructions):
        self.instructions = instructions


class MockArg:
    def __init__(self, value):
        self.value = value


def test_taint_propagates_through_callx_continuation():
    cont_instructions = [
        MockInstruction("LDMSGADDR"),
    ]

    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_instructions))]),
        MockInstruction("EXECUTE"),
        MockInstruction("SENDRAWMSG"),
    ]

    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(main_instructions, cell=None)

    graph = DataFlowAnalyzer().analyze(facts, path_sensitive=True)

    send_idx = next(
        f.index for f in facts.instructions
        if f.opcode == "SENDRAWMSG" and f.continuation_id is None
    )

    assert any(to_idx == send_idx for _, to_idx in graph.tainted_propagation)
