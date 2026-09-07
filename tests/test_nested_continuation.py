from tasmscan.analyzer.program_analyzer import ProgramAnalyzer


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


def test_nested_continuation_call_edges():
    inner_cont = [
        MockInstruction("PUSHINT"),
    ]

    outer_cont = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(inner_cont))]),
        MockInstruction("PUSHINT"),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(outer_cont))]),
        MockInstruction("PUSHINT"),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(main_instructions, cell=None)

    outer_if = next(
        (f for f in facts.instructions if f.instruction.name == "IF" and f.continuation_id),
        None,
    )
    assert outer_if is not None

    call_edges = [e for e in facts.cfg_edges if e.kind == "call_cont" and e.source == outer_if.index]
    assert call_edges
