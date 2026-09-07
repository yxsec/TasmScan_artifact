from tasmscan.analyzer.facts import InstructionFact
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


def _build_facts(main_instructions, continuations):
    main_facts = [
        InstructionFact(instruction=inst, index=i, offset=0, length=0, hash="test",
                        continuation_id=None, parent_index=None)
        for i, inst in enumerate(main_instructions)
    ]

    cont_facts = []
    cont_fact_map = {}
    next_idx = len(main_facts)

    for cont_id, cont in continuations.items():
        for local_idx, cont_inst in enumerate(cont.instructions):
            fact = InstructionFact(
                instruction=cont_inst, index=next_idx, offset=0, length=0,
                hash=f"test:{cont_id}", continuation_id=cont_id,
                parent_index=None
            )
            cont_facts.append(fact)
            cont_fact_map[(cont_id, local_idx)] = next_idx
            next_idx += 1

    return main_facts, cont_facts, cont_fact_map


def _build_push_map(push_map_by_context, main_facts, cont_facts):
    pushcont_to_cont_ids = {}
    for (ctx, local_idx), cont_ids in push_map_by_context.items():
        if ctx == "main":
            pushcont_to_cont_ids[main_facts[local_idx].index] = list(cont_ids)
        else:
            ctx_facts = [f for f in cont_facts if f.continuation_id == ctx]
            if local_idx < len(ctx_facts):
                pushcont_to_cont_ids[ctx_facts[local_idx].index] = list(cont_ids)
    return pushcont_to_cont_ids


def test_continuation_with_unresolved_branch():
    cont = [
        MockInstruction("PUSHINT"),
        MockInstruction("IF"),  # Unresolved branch in continuation
        MockInstruction("PUSHINT"),
    ]

    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont))]),
        MockInstruction("PUSHINT"),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(main_instructions)

    main_facts, cont_facts, cont_fact_map = _build_facts(main_instructions, continuations)
    all_facts = main_facts + cont_facts

    pushcont_to_cont_ids = _build_push_map(push_map, main_facts, cont_facts)

    cfg_edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts, pushcont_to_cont_ids, inline_cont_map={}, continuations=continuations, cont_fact_map=cont_fact_map
    )

    cont_indices = {f.index for f in cont_facts}
    branch_edges = [e for e in cfg_edges if e.kind == "branch" and e.source in cont_indices]
    assert branch_edges


def test_empty_continuation_is_extracted_for_stack_semantics():
    empty_cont = []
    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(empty_cont))]),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(main_instructions)

    # Empty continuation should still be extracted so PUSHCONT semantics are preserved.
    assert len(continuations) == 1
    assert ("main", 0) in push_map
    cont_ids = push_map[("main", 0)]
    assert len(cont_ids) == 1
    assert continuations[cont_ids[0]].instructions == []
    assert inline_map == {}


def test_continuation_ending_with_ret_creates_return_edge():
    cont = [
        MockInstruction("PUSHINT"),
        MockInstruction("RET"),
    ]

    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont))]),
        MockInstruction("PUSHINT"),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(main_instructions)

    main_facts, cont_facts, cont_fact_map = _build_facts(main_instructions, continuations)
    all_facts = main_facts + cont_facts

    pushcont_to_cont_ids = _build_push_map(push_map, main_facts, cont_facts)

    cfg_edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts, pushcont_to_cont_ids, inline_cont_map={}, continuations=continuations, cont_fact_map=cont_fact_map
    )

    return_cont_edges = [e for e in cfg_edges if e.kind == "return_cont"]
    assert return_cont_edges


def test_pushrefcont_is_treated_as_stack_continuation():
    cont = [
        MockInstruction("PUSHINT"),
    ]

    main_instructions = [
        MockInstruction("PUSHREFCONT", [MockArg(MockContinuation(cont))]),
        MockInstruction("PUSHINT"),
        MockInstruction("IFJMP"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(main_instructions)

    assert ("main", 0) in push_map
    assert inline_map == {}

    main_facts, cont_facts, cont_fact_map = _build_facts(main_instructions, continuations)
    all_facts = main_facts + cont_facts
    pushcont_to_cont_ids = _build_push_map(push_map, main_facts, cont_facts)

    cfg_edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts, pushcont_to_cont_ids, inline_cont_map={}, continuations=continuations, cont_fact_map=cont_fact_map
    )

    assert any(e.kind == "jump" and e.source == 2 and e.target is not None for e in cfg_edges)
    assert not any(e.kind == "branch" and e.source == 2 and e.target is None for e in cfg_edges)


def test_interprocedural_entry_shape_resolves_callee_entry_branch():
    cont_target = [
        MockInstruction("PUSHINT"),
    ]
    cont_callee = [
        MockInstruction("IFJMP"),
        MockInstruction("PUSHINT"),
    ]

    # Main calls cont_callee via IF while leaving cont_target on stack.
    # Interprocedural entry-shape propagation should allow IFJMP at callee
    # entry to resolve cont_target instead of emitting unknown branch.
    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_target))]),
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_callee))]),
        MockInstruction("PUSHINT"),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(main_instructions, cell=None)

    callee_ifjmp = next(
        f for f in facts.instructions
        if f.opcode == "IFJMP" and f.continuation_id is not None
    )

    assert any(
        e.source == callee_ifjmp.index and e.kind == "jump" and e.target is not None
        for e in facts.cfg_edges
    )
    assert not any(
        e.source == callee_ifjmp.index
        and e.target is None
        and e.kind in {"branch", "jump", "call"}
        for e in facts.cfg_edges
    )
