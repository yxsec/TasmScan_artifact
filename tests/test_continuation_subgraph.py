from tasmscan.analyzer.facts import AnalysisFacts, Event, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.no_accept import NoAcceptBeforeSendDetector


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


def test_continuation_extraction():
    cont_instructions = [
        MockInstruction("ACCEPT"),
        MockInstruction("SENDRAWMSG"),
    ]

    main_instructions = [
        MockInstruction("PUSHINT"),
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_instructions))]),
        MockInstruction("IF"),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(main_instructions)

    assert len(continuations) == 1
    cont = next(iter(continuations.values()))
    assert cont.parent_context == "main"
    assert cont.parent_local_index == 1
    assert ("main", 1) in push_map
    assert inline_map == {}


def test_continuation_cfg_edges():
    cont_instructions = [
        MockInstruction("ACCEPT"),
        MockInstruction("SENDRAWMSG"),
    ]

    main_instructions = [
        MockInstruction("PUSHINT"),
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_instructions))]),
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

    assert any(e.kind == "fallthrough" for e in cfg_edges)
    assert any(e.kind in {"call_cont", "branch"} for e in cfg_edges)


def test_continuation_blocks():
    cont_instructions = [
        MockInstruction("ACCEPT"),
        MockInstruction("SENDRAWMSG"),
    ]

    main_instructions = [
        MockInstruction("PUSHINT"),
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_instructions))]),
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
    blocks = analyzer._cfg_builder.build_basic_blocks_with_continuations(all_facts, cfg_edges, continuations)

    cont_blocks = [b for b in blocks if b.context != "main"]
    assert cont_blocks


def test_detector_with_continuation():
    cont_instructions = [
        MockInstruction("SENDRAWMSG"),
    ]

    main_instructions = [
        MockInstruction("PUSHINT"),
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_instructions))]),
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
    blocks = analyzer._cfg_builder.build_basic_blocks_with_continuations(all_facts, cfg_edges, continuations)

    events = [Event(type="send", instruction=cont_facts[0])]
    facts = AnalysisFacts(instructions=all_facts, events=events, basic_blocks=blocks)

    detector = NoAcceptBeforeSendDetector()
    findings = detector.detect(facts)

    assert len(findings) >= 1


def test_ifelse_inline_continuations_cfg():
    true_cont = [
        MockInstruction("PUSHINT"),
        MockInstruction("ACCEPT"),
    ]
    false_cont = [
        MockInstruction("PUSHINT"),
        MockInstruction("SENDRAWMSG"),
    ]

    main_instructions = [
        MockInstruction("PUSHINT"),
        MockInstruction("IFELSE", [MockArg(MockContinuation(true_cont)), MockArg(MockContinuation(false_cont))]),
        MockInstruction("PUSHINT"),
    ]

    analyzer = ProgramAnalyzer()
    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(main_instructions)

    assert push_map == {}
    assert ("main", 1) in inline_map
    assert len(inline_map[("main", 1)]) == 2

    facts = analyzer.analyze(main_instructions, cell=None)

    ifelse_fact = next(
        f for f in facts.instructions if f.instruction.name == "IFELSE" and f.continuation_id is None
    )

    cont_entry_indices = []
    for cont_id in inline_map[("main", 1)]:
        cont_indices = [f.index for f in facts.instructions if f.continuation_id == cont_id]
        cont_entry_indices.append(min(cont_indices))

    call_edges = [
        e for e in facts.cfg_edges if e.kind == "call_cont" and e.source == ifelse_fact.index
    ]
    call_targets = sorted(e.target for e in call_edges)
    assert sorted(cont_entry_indices) == call_targets

    # IFELSE should not fall through directly.
    assert not any(
        e.kind == "fallthrough" and e.source == ifelse_fact.index for e in facts.cfg_edges
    )

    # Both continuations should return to the next main instruction.
    return_targets = [e.target for e in facts.cfg_edges if e.kind == "return_cont"]
    assert return_targets.count(ifelse_fact.index + 1) >= 2


def test_reachable_non_method_continuation_not_classified_as_entry_point():
    cont_instructions = [
        MockInstruction("NOP"),
    ]
    main_instructions = [
        MockInstruction("PUSHCONT", [MockArg(MockContinuation(cont_instructions))]),
        MockInstruction("PUSHINT", [MockArg(1)]),
        MockInstruction("IF"),
        MockInstruction("NOP"),
    ]

    analyzer = ProgramAnalyzer()
    facts = analyzer.analyze(main_instructions, cell=None)

    # Main entry should exist.
    assert any(ep.get("kind") == "main" for ep in facts.entry_points)
    # Reachable non-method continuation should not be treated as module/function entry.
    assert not any(ep.get("kind") == "continuation" for ep in facts.entry_points)
