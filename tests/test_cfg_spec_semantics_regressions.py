from tasmscan.analyzer.facts import Continuation, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer


class MockInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class MockArg:
    def __init__(self, value):
        self.value = value


def make_fact(index: int, opcode: str, *, args=None, continuation_id=None) -> InstructionFact:
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


def test_retalt_in_callee_does_not_fake_return_to_caller_next():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "IF"),
        make_fact(2, "NOP"),
        make_fact(10, "RETALT", continuation_id="cont_callee"),
    ]
    continuations = {
        "cont_callee": Continuation(
            id="cont_callee",
            instructions=[MockInstruction("RETALT")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        )
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_callee"]},
        continuations=continuations,
        cont_fact_map={("cont_callee", 0): 10},
    )

    assert not any(e.source == 10 and e.kind == "return_cont" and e.target == 2 for e in edges)
    assert any(
        e.source == 10
        and e.kind == "jump"
        and e.target is None
        and bool((e.metadata or {}).get("return_target_unknown"))
        for e in edges
    )


def test_retbool_in_callee_keeps_c0_and_marks_c1_unknown():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "IF"),
        make_fact(2, "NOP"),
        make_fact(10, "RETBOOL", continuation_id="cont_callee"),
    ]
    continuations = {
        "cont_callee": Continuation(
            id="cont_callee",
            instructions=[MockInstruction("RETBOOL")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        )
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_callee"]},
        continuations=continuations,
        cont_fact_map={("cont_callee", 0): 10},
    )

    assert any(e.source == 10 and e.kind == "return_cont" and e.target == 2 for e in edges)
    assert any(
        e.source == 10
        and e.kind == "jump"
        and e.target is None
        and bool((e.metadata or {}).get("return_target_unknown"))
        for e in edges
    )


def test_retbool_in_callee_uses_known_c1_target_when_propagated():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # cont_alt
        make_fact(1, "SETALTCTR"),
        make_fact(2, "PUSHCONT"),  # cont_callee
        make_fact(3, "IF"),
        make_fact(4, "NOP"),
        make_fact(10, "RETBOOL", continuation_id="cont_callee"),
        make_fact(20, "RET", continuation_id="cont_alt"),
    ]
    continuations = {
        "cont_alt": Continuation(
            id="cont_alt",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_callee": Continuation(
            id="cont_callee",
            instructions=[MockInstruction("RETBOOL")],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_alt"], 2: ["cont_callee"]},
        continuations=continuations,
        cont_fact_map={("cont_alt", 0): 20, ("cont_callee", 0): 10},
    )

    retbool_edges = [e for e in edges if e.source == 10]
    assert any(e.kind == "return_cont" and e.target == 4 for e in retbool_edges)
    assert any(
        e.kind == "return_cont"
        and e.target == 20
        and bool((e.metadata or {}).get("via_c1"))
        for e in retbool_edges
    )
    assert not any(
        e.kind == "jump"
        and e.target is None
        and bool((e.metadata or {}).get("return_target_unknown"))
        for e in retbool_edges
    )


def test_retalt_in_callee_uses_known_c1_target_without_unknown_edge():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # cont_alt
        make_fact(1, "SETALTCTR"),
        make_fact(2, "PUSHCONT"),  # cont_callee
        make_fact(3, "IF"),
        make_fact(4, "NOP"),
        make_fact(10, "RETALT", continuation_id="cont_callee"),
        make_fact(20, "RET", continuation_id="cont_alt"),
    ]
    continuations = {
        "cont_alt": Continuation(
            id="cont_alt",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_callee": Continuation(
            id="cont_callee",
            instructions=[MockInstruction("RETALT")],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_alt"], 2: ["cont_callee"]},
        continuations=continuations,
        cont_fact_map={("cont_alt", 0): 20, ("cont_callee", 0): 10},
    )

    retalt_edges = [e for e in edges if e.source == 10]
    assert any(
        e.kind == "return_cont"
        and e.target == 20
        and bool((e.metadata or {}).get("via_c1"))
        for e in retalt_edges
    )
    assert not any(
        e.kind == "jump"
        and e.target is None
        and bool((e.metadata or {}).get("return_target_unknown"))
        for e in retalt_edges
    )


def test_samealtsave_copies_c0_into_c1_for_retalt():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # cont_alt
        make_fact(1, "POPCTR", args=[MockArg(0)]),  # c0 = cont_alt
        make_fact(2, "SAMEALTSAVE"),  # c1 = c0
        make_fact(3, "PUSHCONT"),  # cont_callee
        make_fact(4, "EXECUTE"),
        make_fact(5, "NOP"),
        make_fact(10, "RETALT", continuation_id="cont_callee"),
        make_fact(20, "RET", continuation_id="cont_alt"),
    ]
    continuations = {
        "cont_alt": Continuation(
            id="cont_alt",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_callee": Continuation(
            id="cont_callee",
            instructions=[MockInstruction("RETALT")],
            entry_index=0,
            parent_context="main",
            parent_local_index=3,
        ),
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_alt"], 3: ["cont_callee"]},
        continuations=continuations,
        cont_fact_map={("cont_alt", 0): 20, ("cont_callee", 0): 10},
    )

    retalt_edges = [e for e in edges if e.source == 10]
    assert any(
        e.kind == "return_cont"
        and e.target == 20
        and bool((e.metadata or {}).get("via_c1"))
        for e in retalt_edges
    )
    assert not any(
        e.kind == "jump"
        and e.target is None
        and bool((e.metadata or {}).get("return_target_unknown"))
        for e in retalt_edges
    )


def test_setexitalt_invalidates_stale_known_c1_target():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # cont_alt
        make_fact(1, "SETALTCTR"),  # c1 = cont_alt
        make_fact(2, "PUSHCONT"),  # cont_exit_wrapper input
        make_fact(3, "SETEXITALT"),  # c1 becomes composed continuation (unknown id)
        make_fact(4, "PUSHCONT"),  # cont_callee
        make_fact(5, "EXECUTE"),
        make_fact(6, "NOP"),
        make_fact(10, "RETALT", continuation_id="cont_callee"),
        make_fact(20, "RET", continuation_id="cont_alt"),
        make_fact(30, "RET", continuation_id="cont_exit_wrapper"),
    ]
    continuations = {
        "cont_alt": Continuation(
            id="cont_alt",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_exit_wrapper": Continuation(
            id="cont_exit_wrapper",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
        "cont_callee": Continuation(
            id="cont_callee",
            instructions=[MockInstruction("RETALT")],
            entry_index=0,
            parent_context="main",
            parent_local_index=4,
        ),
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={
            0: ["cont_alt"],
            2: ["cont_exit_wrapper"],
            4: ["cont_callee"],
        },
        continuations=continuations,
        cont_fact_map={
            ("cont_alt", 0): 20,
            ("cont_exit_wrapper", 0): 30,
            ("cont_callee", 0): 10,
        },
    )

    retalt_edges = [e for e in edges if e.source == 10]
    assert not any(
        e.kind == "return_cont"
        and e.target == 20
        and bool((e.metadata or {}).get("via_c1"))
        for e in retalt_edges
    )
    assert any(
        e.kind == "jump"
        and e.target is None
        and bool((e.metadata or {}).get("return_target_unknown"))
        for e in retalt_edges
    )


def test_repeat_keeps_direct_fallthrough_even_when_loop_body_resolves():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[MockArg(-1)]),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "REPEAT"),
        make_fact(3, "NOP"),
        make_fact(10, "THROW", continuation_id="loop_body"),
    ]
    continuations = {
        "loop_body": Continuation(
            id="loop_body",
            instructions=[MockInstruction("THROW")],
            entry_index=0,
            parent_context="main",
            parent_local_index=1,
        )
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={1: ["loop_body"]},
        continuations=continuations,
        cont_fact_map={("loop_body", 0): 10},
    )

    assert any(e.source == 2 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 2 and e.kind == "fallthrough" and e.target == 3 for e in edges)


def test_callxargs_var_return_count_does_not_resolve_stale_below_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "CALLXARGS", args=[MockArg(0), MockArg(-1)]),
        make_fact(3, "IFJMP"),
    ]

    branch_map, _returning, _nested, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts,
        pushcont_to_cont_ids={0: ["cont_below"], 1: ["cont_call"]},
        inline_cont_map={},
    )

    assert branch_map.get(2) == ["cont_call"]
    assert 3 not in branch_map
    assert 3 in uncertain


def test_loop_calls_form_basic_block_boundaries():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[MockArg(1)]),
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
        )
    }
    edges = _build_cfg(
        analyzer,
        facts,
        push_map={1: ["loop_body"]},
        continuations=continuations,
        cont_fact_map={("loop_body", 0): 10},
    )
    blocks = analyzer._cfg_builder.build_basic_blocks_with_continuations(facts, edges, continuations)

    main_blocks = [b for b in blocks if b.context == "main"]
    block_with_repeat = next(b for b in main_blocks if 2 in b.instruction_indices)
    block_with_next = next(b for b in main_blocks if 3 in b.instruction_indices)
    assert block_with_repeat.id != block_with_next.id


def test_jmpdict_with_unknown_c3_emits_unknown_jump_not_stack_target():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "JMPDICT"),
        make_fact(2, "NOP"),
        make_fact(10, "RET", continuation_id="cont_stack"),
    ]
    continuations = {
        "cont_stack": Continuation(
            id="cont_stack",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        )
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_stack"]},
        continuations=continuations,
        cont_fact_map={("cont_stack", 0): 10},
    )

    assert not any(e.source == 1 and e.kind == "jump" and e.target == 10 for e in edges)
    assert any(e.source == 1 and e.kind == "jump" and e.target is None for e in edges)
    assert not any(e.source == 1 and e.kind == "fallthrough" and e.target == 2 for e in edges)


def test_jmpdict_uses_c3_target_instead_of_top_of_stack_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "POPCTR", args=[MockArg(3)]),  # c3 = cont_c3
        make_fact(2, "PUSHCONT"),  # unrelated continuation on stack
        make_fact(3, "JMPDICT"),
        make_fact(10, "RET", continuation_id="cont_c3"),
        make_fact(20, "RET", continuation_id="cont_stack"),
    ]
    continuations = {
        "cont_c3": Continuation(
            id="cont_c3",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_stack": Continuation(
            id="cont_stack",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
    }

    edges = _build_cfg(
        analyzer,
        facts,
        push_map={0: ["cont_c3"], 2: ["cont_stack"]},
        continuations=continuations,
        cont_fact_map={("cont_c3", 0): 10, ("cont_stack", 0): 20},
    )

    assert any(e.source == 3 and e.kind == "jump" and e.target == 10 for e in edges)
    assert not any(e.source == 3 and e.kind == "jump" and e.target == 20 for e in edges)


def test_dictigetexec_models_call_and_success_return():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "DICTIGETEXEC"),
        make_fact(1, "NOP"),
        make_fact(10, "RET", continuation_id="method_cont"),
    ]
    continuations = {
        "method_cont": Continuation(
            id="method_cont",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            kind="method",
            method_id=1,
        )
    }

    edges = _build_cfg(
        analyzer,
        facts,
        continuations=continuations,
        cont_fact_map={("method_cont", 0): 10},
    )

    assert any(e.source == 0 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 0 and e.kind == "call_return" and e.target == 1 for e in edges)
    assert any(e.source == 0 and e.kind == "fallthrough" and e.target == 1 for e in edges)


def test_pfxdictgetexec_has_no_direct_fallthrough_but_keeps_call_return():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PFXDICTGETEXEC"),
        make_fact(1, "NOP"),
        make_fact(10, "RET", continuation_id="method_cont"),
    ]
    continuations = {
        "method_cont": Continuation(
            id="method_cont",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            kind="method",
            method_id=1,
        )
    }

    edges = _build_cfg(
        analyzer,
        facts,
        continuations=continuations,
        cont_fact_map={("method_cont", 0): 10},
    )

    assert any(e.source == 0 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 0 and e.kind == "call_return" and e.target == 1 for e in edges)
    assert not any(e.source == 0 and e.kind == "fallthrough" and e.target == 1 for e in edges)
