from tasmscan.analyzer.facts import AnalysisFacts, Continuation, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.ir.dataflow import DataFlowAnalyzer, DataFlowValue, ValueSource


class MockInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class MockArg:
    def __init__(self, value):
        self.value = value


def make_fact(
    index: int,
    opcode: str,
    *,
    args=None,
    continuation_id=None,
) -> InstructionFact:
    return InstructionFact(
        instruction=MockInstruction(opcode, args=args or []),
        index=index,
        continuation_id=continuation_id,
    )


def _build_cfg(
    analyzer: ProgramAnalyzer,
    all_facts,
    *,
    push_map=None,
    inline_map=None,
    continuations=None,
    cont_fact_map=None,
):
    return analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=push_map or {},
        inline_cont_map=inline_map or {},
        continuations=continuations or {},
        cont_fact_map=cont_fact_map or {},
    )


def test_dict_dispatch_opcodes_emit_unknown_cfg_edges():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "DICTIGETJMP"),
        make_fact(1, "NOP"),
    ]

    edges = _build_cfg(analyzer, facts)

    assert any(e.source == 0 and e.kind == "fallthrough" and e.target == 1 for e in edges)
    assert any(
        e.source == 0
        and e.kind == "branch"
        and e.target is None
        and bool((e.metadata or {}).get("target_unknown"))
        for e in edges
    )


def test_try_opcode_targets_body_inline_continuation_only():
    analyzer = ProgramAnalyzer()
    all_facts = [
        make_fact(0, "TRY"),
        make_fact(1, "NOP"),
        make_fact(10, "NOP", continuation_id="cont_body"),
        make_fact(11, "RET", continuation_id="cont_body"),
        make_fact(20, "NOP", continuation_id="cont_handler"),
        make_fact(21, "RET", continuation_id="cont_handler"),
    ]
    continuations = {
        "cont_body": Continuation(
            id="cont_body",
            instructions=[MockInstruction("NOP")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_handler": Continuation(
            id="cont_handler",
            instructions=[MockInstruction("NOP")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
    }
    edges = _build_cfg(
        analyzer,
        all_facts,
        inline_map={0: ["cont_body", "cont_handler"]},
        continuations=continuations,
        cont_fact_map={("cont_body", 0): 10, ("cont_handler", 0): 20},
    )

    call_targets = {e.target for e in edges if e.source == 0 and e.kind == "call_cont"}
    assert call_targets == {10}
    assert any(e.source == 11 and e.kind == "return_cont" and e.target == 1 for e in edges)
    assert not any(e.source == 21 and e.kind == "return_cont" for e in edges)


def test_tryargs_targets_body_continuation_only():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "TRYARGS", args=[MockArg(1), MockArg(0)]),
        make_fact(1, "NOP"),
    ]

    branch_map, returning_branches, _, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts,
        pushcont_to_cont_ids={},
        inline_cont_map={0: ["cont_body", "cont_handler"]},
    )

    assert branch_map[0] == ["cont_body"]
    assert 0 in returning_branches
    assert 0 not in uncertain


def test_bless_then_execute_keeps_unknown_call_and_call_return():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "BLESS"),
        make_fact(1, "EXECUTE"),
        make_fact(2, "NOP"),
    ]

    edges = _build_cfg(analyzer, facts)

    assert any(e.source == 1 and e.kind == "call" and e.target is None for e in edges)
    assert any(e.source == 1 and e.kind == "call_return" and e.target == 2 for e in edges)


def test_atexit_invalidates_c0_based_dispatch_resolution():
    analyzer = ProgramAnalyzer()
    all_facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "POPCTR", args=[MockArg(0)]),  # c0 = cont_a
        make_fact(2, "PUSHCONT"),
        make_fact(3, "ATEXIT"),  # mutates c0
        make_fact(4, "PUSHCTR", args=[MockArg(0)]),
        make_fact(5, "JMPX"),
        make_fact(6, "NOP"),
        make_fact(10, "NOP", continuation_id="cont_a"),
    ]
    continuations = {
        "cont_a": Continuation(
            id="cont_a",
            instructions=[MockInstruction("NOP")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_b": Continuation(
            id="cont_b",
            instructions=[MockInstruction("NOP")],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
    }
    edges = _build_cfg(
        analyzer,
        all_facts,
        push_map={0: ["cont_a"], 2: ["cont_b"]},
        continuations=continuations,
        cont_fact_map={("cont_a", 0): 10},
    )

    assert any(e.source == 5 and e.kind == "jump" and e.target is None for e in edges)
    assert not any(e.source == 5 and e.kind == "jump" and e.target == 10 for e in edges)


def test_setcontctr_output_keeps_unknown_call_and_call_return():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[MockArg(7)]),
        make_fact(2, "SETCONTCTR", args=[MockArg(0)]),
        make_fact(3, "EXECUTE"),
        make_fact(4, "NOP"),
    ]

    edges = _build_cfg(analyzer, facts, push_map={0: ["cont_a"]})

    assert any(e.source == 3 and e.kind == "call" and e.target is None for e in edges)
    assert any(e.source == 3 and e.kind == "call_return" and e.target == 4 for e in edges)


def test_again_does_not_emit_return_to_next_instruction():
    analyzer = ProgramAnalyzer()
    all_facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "AGAIN"),
        make_fact(2, "NOP"),
        make_fact(10, "RET", continuation_id="loop_body"),
    ]
    continuations = {
        "loop_body": Continuation(
            id="loop_body",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
    }

    edges = _build_cfg(
        analyzer,
        all_facts,
        push_map={0: ["loop_body"]},
        continuations=continuations,
        cont_fact_map={("loop_body", 0): 10},
    )

    assert any(e.source == 1 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert not any(e.source == 1 and e.kind == "fallthrough" and e.target == 2 for e in edges)
    assert not any(e.source == 10 and e.kind == "return_cont" and e.target == 2 for e in edges)
    assert any(e.source == 10 and e.kind == "return_cont" and e.target == 1 for e in edges)


def test_again_retalt_exits_loop_instead_of_continue():
    analyzer = ProgramAnalyzer()
    all_facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "AGAIN"),
        make_fact(2, "NOP"),
        make_fact(10, "RETALT", continuation_id="loop_body"),
    ]
    continuations = {
        "loop_body": Continuation(
            id="loop_body",
            instructions=[MockInstruction("RETALT")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
    }

    edges = _build_cfg(
        analyzer,
        all_facts,
        push_map={0: ["loop_body"]},
        continuations=continuations,
        cont_fact_map={("loop_body", 0): 10},
    )

    assert any(e.source == 10 and e.kind == "return_cont" and e.target == 2 for e in edges)
    assert not any(e.source == 10 and e.kind == "return_cont" and e.target == 1 for e in edges)


def test_again_unresolved_target_does_not_fake_exit_edge():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "AGAIN"),
        make_fact(1, "NOP"),
    ]

    edges = _build_cfg(analyzer, facts)

    assert any(e.source == 0 and e.kind == "call" and e.target is None for e in edges)
    assert not any(e.source == 0 and e.kind == "fallthrough" and e.target == 1 for e in edges)
    assert not any(e.source == 0 and e.kind == "call_return" and e.target == 1 for e in edges)


def test_againbrk_unresolved_target_does_not_fake_exit_edge():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "AGAINBRK"),
        make_fact(1, "NOP"),
    ]

    edges = _build_cfg(analyzer, facts)

    assert any(e.source == 0 and e.kind == "call" and e.target is None for e in edges)
    assert not any(e.source == 0 and e.kind == "fallthrough" and e.target == 1 for e in edges)
    assert not any(e.source == 0 and e.kind == "call_return" and e.target == 1 for e in edges)


def test_repeat_models_continue_backedge_and_possible_exit():
    analyzer = ProgramAnalyzer()
    all_facts = [
        make_fact(0, "PUSHINT", args=[MockArg(2)]),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "REPEAT"),
        make_fact(3, "NOP"),
        make_fact(10, "RET", continuation_id="loop_body"),
    ]
    continuations = {
        "loop_body": Continuation(
            id="loop_body",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=1,
        ),
    }

    edges = _build_cfg(
        analyzer,
        all_facts,
        push_map={1: ["loop_body"]},
        continuations=continuations,
        cont_fact_map={("loop_body", 0): 10},
    )

    assert any(e.source == 2 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 10 and e.kind == "return_cont" and e.target == 2 for e in edges)
    assert any(e.source == 10 and e.kind == "return_cont" and e.target == 3 for e in edges)


def test_popctr_pushctr_preserve_taint_through_register():
    analyzer = DataFlowAnalyzer()
    facts = AnalysisFacts(
        instructions=[
            make_fact(0, "LDMSGADDR"),
            make_fact(1, "POPCTR", args=[MockArg(0)]),
            make_fact(2, "PUSHCTR", args=[MockArg(0)]),
        ]
    )

    analyzer.analyze(facts)

    reg0 = analyzer.current_state.registers.get(0)
    assert reg0 is not None and reg0.tainted
    assert analyzer.current_state.stack
    assert analyzer.current_state.stack[0].tainted


def test_medium_confidence_guard_does_not_propagate_all_taint_origins():
    analyzer = DataFlowAnalyzer()
    analyzer.current_state.stack = [
        DataFlowValue(
            source=ValueSource.MESSAGE_VALUE,
            definition_site=10,
            tainted=True,
            metadata={"taint_origins": [1, 2]},
        )
    ]

    analyzer._process_guard_instruction("THROWIFNOT", analyzer.GUARD_OPCODES)

    assert 10 in analyzer.current_state.guarded_values
    assert 1 not in analyzer.current_state.guarded_values
    assert 2 not in analyzer.current_state.guarded_values
