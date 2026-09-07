from tasmscan.analyzer.program_analyzer import ProgramAnalyzer


class MockInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class MockArg:
    def __init__(self, value):
        self.value = value


class MockContinuation:
    def __init__(self, instructions):
        self.instructions = instructions


def _facts_by_opcode(facts, opcode, *, continuation_id=None):
    return [
        fact
        for fact in facts.instructions
        if fact.opcode == opcode and fact.continuation_id == continuation_id
    ]


def test_againend_tail_is_materialized_as_synthetic_continuation():
    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(
        [
            MockInstruction("PUSHINT"),
            MockInstruction("AGAINEND"),
            MockInstruction("NOP"),
            MockInstruction("RET"),
        ],
        cell=None,
    )

    main_opcodes = [f.opcode for f in facts.instructions if f.continuation_id is None]
    assert main_opcodes == ["PUSHINT", "AGAINEND"]

    continuations = facts.metadata["continuations"]
    synthetic = [cont for cont in continuations.values() if cont.kind == "end_loop_body"]
    assert len(synthetic) == 1
    synthetic_id = synthetic[0].id

    synthetic_opcodes = [f.opcode for f in facts.instructions if f.continuation_id == synthetic_id]
    assert synthetic_opcodes == ["NOP", "RET"]

    againend = _facts_by_opcode(facts, "AGAINEND", continuation_id=None)[0]
    entry_idx = min(f.index for f in facts.instructions if f.continuation_id == synthetic_id)
    out_edges = [e for e in facts.cfg_edges if e.source == againend.index]
    assert any(e.kind == "call_cont" and e.target == entry_idx for e in out_edges)
    assert not any(e.kind == "fallthrough" for e in out_edges)


def test_againend_at_context_end_does_not_create_synthetic_continuation():
    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(
        [MockInstruction("PUSHINT"), MockInstruction("AGAINEND")],
        cell=None,
    )

    continuations = facts.metadata["continuations"]
    assert not any(cont.kind == "end_loop_body" for cont in continuations.values())

    againend = _facts_by_opcode(facts, "AGAINEND", continuation_id=None)[0]
    out_edges = [e for e in facts.cfg_edges if e.source == againend.index]
    assert not any(e.kind == "fallthrough" for e in out_edges)


def test_nested_pushcont_inside_end_loop_body_keeps_resolved_mapping():
    analyzer = ProgramAnalyzer()
    nested_ret = MockContinuation([MockInstruction("RET")])
    facts = analyzer.analyze(
        [
            MockInstruction("AGAINEND"),
            MockInstruction("PUSHCONT", args=[MockArg(nested_ret)]),
            MockInstruction("PUSHINT"),
            MockInstruction("IF"),
        ],
        cell=None,
    )

    continuations = facts.metadata["continuations"]
    end_body = next(cont for cont in continuations.values() if cont.kind == "end_loop_body")
    child_conts = [
        cont
        for cont in continuations.values()
        if cont.parent_context == end_body.id and cont.parent_local_index == 0
    ]
    assert len(child_conts) == 1
    child_id = child_conts[0].id

    if_fact = _facts_by_opcode(facts, "IF", continuation_id=end_body.id)[0]
    child_entry = min(f.index for f in facts.instructions if f.continuation_id == child_id)
    if_edges = [e for e in facts.cfg_edges if e.source == if_fact.index]
    assert any(e.kind == "call_cont" and e.target == child_entry for e in if_edges)
