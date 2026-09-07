from tasmscan.analyzer.facts import (
    AnalysisFacts,
    Continuation,
    ControlFlowEdge,
    Event,
    InstructionFact,
)
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.no_accept import NoAcceptBeforeSendDetector


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class MockArg:
    def __init__(self, value):
        self.value = value


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


def test_unknown_branch_handling():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=4),
        InstructionFact(instruction=MockInstruction("IF"), index=2, offset=8),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 2}), index=3, offset=12),
        InstructionFact(instruction=MockInstruction("ACCEPT"), index=4, offset=16),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 3}), index=5, offset=20),
        InstructionFact(instruction=MockInstruction("SENDRAWMSG"), index=6, offset=24),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)
    blocks = _build_blocks(analyzer._cfg_builder, instructions, edges)

    if_block = next((b for b in blocks if 2 in b.instruction_indices), None)
    assert if_block is not None

    # No negative-ID blocks should exist
    assert all(b.id >= 0 for b in blocks)

    # Entry block should be valid
    entry_id = min(b.id for b in blocks)
    assert entry_id >= 0

    # Unknown continuation should be represented with has_unknown_successor flag
    # (not -1 in successors - that was the old behavior)
    assert if_block.has_unknown_successor is True
    # All successor IDs should be valid (non-negative)
    assert all(s >= 0 for s in if_block.successors)


def test_basic_blocks_split_on_jump_target_leader():
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("NOP"), index=2, offset=2),
    ]
    edges = [
        ControlFlowEdge(source=0, target=1, kind="fallthrough"),
        ControlFlowEdge(source=1, target=2, kind="fallthrough"),
        ControlFlowEdge(source=2, target=1, kind="jump"),
    ]

    analyzer = ProgramAnalyzer()
    blocks = _build_blocks(analyzer._cfg_builder, instructions, edges)
    block_with_zero = next(b for b in blocks if 0 in b.instruction_indices)
    block_with_one = next(b for b in blocks if 1 in b.instruction_indices)

    assert block_with_zero.id != block_with_one.id
    assert block_with_zero.instruction_indices == [0]
    assert block_with_one.instruction_indices[0] == 1


def test_guard_throw_unknown_marks_block_unknown_successor():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("THROWIF"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("NOP"), index=2, offset=2),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)
    blocks = _build_blocks(analyzer._cfg_builder, instructions, edges)
    guard_block = next((b for b in blocks if 1 in b.instruction_indices), None)

    assert guard_block is not None
    assert guard_block.has_unknown_successor is True


def test_guard_return_unknown_marks_block_unknown_successor():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("IFRET"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("NOP"), index=2, offset=2),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)
    blocks = _build_blocks(analyzer._cfg_builder, instructions, edges)
    guard_block = next((b for b in blocks if 1 in b.instruction_indices), None)

    assert guard_block is not None
    assert guard_block.has_unknown_successor is True


def test_dictigetjmpz_expands_to_method_entries_without_unknown_edge():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("DICTIGETJMPZ"), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1, continuation_id=None),
        InstructionFact(instruction=MockInstruction("RET"), index=10, offset=10, continuation_id="cont_m0"),
        InstructionFact(instruction=MockInstruction("RET"), index=20, offset=20, continuation_id="cont_m1"),
    ]
    continuations = {
        "cont_m0": Continuation(
            id="cont_m0",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            kind="method",
            method_id=0,
        ),
        "cont_m1": Continuation(
            id="cont_m1",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            kind="method",
            method_id=1,
        ),
    }
    cont_fact_map = {
        ("cont_m0", 0): 10,
        ("cont_m1", 0): 20,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(e.source == 0 and e.kind == "fallthrough" and e.target == 1 for e in edges)
    assert any(e.source == 0 and e.kind == "jump" and e.target == 10 for e in edges)
    assert any(e.source == 0 and e.kind == "jump" and e.target == 20 for e in edges)
    assert not any(
        e.source == 0 and e.target is None and e.kind in {"branch", "jump", "call"}
        for e in edges
    )


def test_accept_send_detector_with_unknown_branch():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=4),
        InstructionFact(instruction=MockInstruction("IF"), index=2, offset=8),
        InstructionFact(instruction=MockInstruction("ACCEPT"), index=3, offset=12),
        InstructionFact(instruction=MockInstruction("SENDRAWMSG"), index=4, offset=16),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)
    blocks = _build_blocks(analyzer._cfg_builder, instructions, edges)

    events = [
        Event(type="accept", instruction=instructions[3]),
        Event(type="send", instruction=instructions[4]),
    ]

    facts = AnalysisFacts(
        instructions=instructions,
        events=events,
        basic_blocks=blocks,
    )

    detector = NoAcceptBeforeSendDetector()
    findings = detector.detect(facts)

    # Should not flag "send before accept" on the known fallthrough path.
    assert not any("send before accept" in f.message.lower() for f in findings)


def test_stack_uncertain_creates_marked_edge():
    """Test that stack_uncertain branches create edges with stack_uncertain metadata."""
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("ROLL"), index=1, offset=4),  # Unmodeled shuffle
        InstructionFact(instruction=MockInstruction("IF"), index=2, offset=8),    # Cannot resolve continuation
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=12),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)

    # Find the unknown branch edge from IF
    unknown_edges = [e for e in edges if e.source == 2 and e.target is None]
    assert len(unknown_edges) == 1

    edge = unknown_edges[0]
    assert edge.metadata is not None
    assert edge.metadata.get("target_unknown") is True
    assert edge.metadata.get("stack_uncertain") is True  # Important invariant: marked as stack_uncertain


def test_calldict_without_resolved_c3_keeps_unknown_call_and_call_return():
    instructions = [
        InstructionFact(instruction=MockInstruction("CALLDICT"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=4),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)

    call_edges = [e for e in edges if e.source == 0 and e.kind == "call"]
    call_return_edges = [e for e in edges if e.source == 0 and e.kind == "call_return"]

    assert any(e.target is None for e in call_edges)
    assert len(call_return_edges) == 1
    assert call_return_edges[0].target == 1


def test_execute_without_known_cont_keeps_unknown_call_and_call_return():
    instructions = [
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=4),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)

    call_edges = [e for e in edges if e.source == 0 and e.kind == "call"]
    call_return_edges = [e for e in edges if e.source == 0 and e.kind == "call_return"]

    assert any(e.target is None for e in call_edges)
    assert len(call_return_edges) == 1
    assert call_return_edges[0].target == 1


def test_try_with_unknown_body_does_not_misresolve_top_known_handler():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(0)]), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(0)]), index=1, offset=1, continuation_id=None),
        # Produces an internal unknown continuation placeholder on stack.
        InstructionFact(instruction=MockInstruction("SETCONTCTR", args=[MockArg(1)]), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("TRY"), index=4, offset=4, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=5, offset=5, continuation_id=None),
        InstructionFact(instruction=MockInstruction("RET"), index=10, offset=10, continuation_id="cont_body"),
    ]
    pushcont_to_cont_ids = {3: ["cont_body"]}
    continuations = {
        "cont_body": Continuation(
            id="cont_body",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=3,
        ),
    }
    cont_fact_map = {("cont_body", 0): 10}

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert not any(e.source == 4 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 4 and e.kind == "call" and e.target is None for e in edges)
    assert any(e.source == 4 and e.kind == "call_return" and e.target == 5 for e in edges)


def test_try_propagates_handler_into_c2_for_body_context():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),  # body
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=1, continuation_id=None),  # handler
        InstructionFact(instruction=MockInstruction("TRY"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCTR", args=[MockArg(2)]), index=10, offset=10, continuation_id="cont_body"),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=11, offset=11, continuation_id="cont_body"),
        InstructionFact(instruction=MockInstruction("RET"), index=12, offset=12, continuation_id="cont_body"),
        InstructionFact(instruction=MockInstruction("RET"), index=20, offset=20, continuation_id="cont_handler"),
    ]
    continuations = {
        "cont_body": Continuation(
            id="cont_body",
            instructions=[MockInstruction("PUSHCTR"), MockInstruction("EXECUTE"), MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "cont_handler": Continuation(
            id="cont_handler",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=1,
        ),
    }
    cont_fact_map = {
        ("cont_body", 0): 10,
        ("cont_handler", 0): 20,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={0: ["cont_body"], 1: ["cont_handler"]},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(e.source == 2 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 11 and e.kind == "call_cont" and e.target == 20 for e in edges)


def test_tryargs_entry_shape_does_not_leak_non_passed_stack_values_into_body():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),  # stale
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=1, continuation_id=None),  # body
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=2, offset=2, continuation_id=None),  # handler
        InstructionFact(instruction=MockInstruction("TRYARGS", args=[MockArg(0), MockArg(0)]), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(1)]), index=10, offset=10, continuation_id="cont_body"),
        InstructionFact(instruction=MockInstruction("IF"), index=11, offset=11, continuation_id="cont_body"),
        InstructionFact(instruction=MockInstruction("RET"), index=12, offset=12, continuation_id="cont_body"),
        InstructionFact(instruction=MockInstruction("RET"), index=20, offset=20, continuation_id="cont_handler"),
        InstructionFact(instruction=MockInstruction("RET"), index=30, offset=30, continuation_id="cont_stale"),
    ]
    continuations = {
        "cont_body": Continuation(
            id="cont_body",
            instructions=[MockInstruction("PUSHINT"), MockInstruction("IF"), MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=1,
        ),
        "cont_handler": Continuation(
            id="cont_handler",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
        "cont_stale": Continuation(
            id="cont_stale",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
    }
    cont_fact_map = {
        ("cont_body", 0): 10,
        ("cont_handler", 0): 20,
        ("cont_stale", 0): 30,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={0: ["cont_stale"], 1: ["cont_body"], 2: ["cont_handler"]},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(e.source == 3 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert not any(e.source == 11 and e.kind == "call_cont" and e.target == 30 for e in edges)


def test_execute_with_unknown_placeholder_keeps_unknown_call_and_call_return():
    analyzer = ProgramAnalyzer()
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(0)]), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(0)]), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("SETCONTCTR", args=[MockArg(1)]), index=2, offset=2),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=3, offset=3),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4),
    ]

    edges = _build_cfg(analyzer._cfg_builder, instructions)

    assert any(e.source == 3 and e.kind == "call" and e.target is None for e in edges)
    assert any(e.source == 3 and e.kind == "call_return" and e.target == 4 for e in edges)


def test_execute_with_unknown_placeholder_overapproximates_to_method_entries():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(0)]), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT", args=[MockArg(0)]), index=1, offset=1, continuation_id=None),
        InstructionFact(instruction=MockInstruction("SETCONTCTR", args=[MockArg(1)]), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4, continuation_id=None),
        InstructionFact(instruction=MockInstruction("RET"), index=10, offset=10, continuation_id="cont_m0"),
        InstructionFact(instruction=MockInstruction("RET"), index=20, offset=20, continuation_id="cont_m1"),
    ]
    continuations = {
        "cont_m0": Continuation(
            id="cont_m0",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            kind="method",
            method_id=0,
        ),
        "cont_m1": Continuation(
            id="cont_m1",
            instructions=[MockInstruction("RET")],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            kind="method",
            method_id=1,
        ),
    }
    cont_fact_map = {
        ("cont_m0", 0): 10,
        ("cont_m1", 0): 20,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(e.source == 3 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert any(e.source == 3 and e.kind == "call_cont" and e.target == 20 for e in edges)
    assert not any(e.source == 3 and e.kind == "call" and e.target is None for e in edges)


def test_while_with_stack_uncertainty_emits_unresolved_call_edge():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("ROLL"), index=1, offset=4),  # Unmodeled shuffle
        InstructionFact(instruction=MockInstruction("WHILE"), index=2, offset=8),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=12),
    ]

    analyzer = ProgramAnalyzer()
    edges = _build_cfg(analyzer._cfg_builder, instructions)

    unresolved_calls = [e for e in edges if e.source == 2 and e.kind == "call" and e.target is None]
    fallthroughs = [e for e in edges if e.source == 2 and e.kind == "fallthrough" and e.target == 3]

    assert len(unresolved_calls) == 1
    assert unresolved_calls[0].metadata.get("stack_uncertain") is True
    assert len(fallthroughs) == 1


def test_while_with_empty_and_nonempty_conts_avoids_false_unresolved_call():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=1, continuation_id=None),
        InstructionFact(instruction=MockInstruction("WHILE"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=10, offset=10, continuation_id="cont_body"),
    ]
    pushcont_to_cont_ids = {
        0: ["cont_empty"],
        1: ["cont_body"],
    }
    continuations = {
        "cont_empty": Continuation(
            id="cont_empty", instructions=[], entry_index=0, parent_context="main", parent_local_index=0
        ),
        "cont_body": Continuation(
            id="cont_body", instructions=[MockInstruction("NOP")], entry_index=0, parent_context="main", parent_local_index=1
        ),
    }
    cont_fact_map = {
        ("cont_body", 0): 10,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(e.source == 2 and e.kind == "call_cont" and e.target == 10 for e in edges)
    assert not any(e.source == 2 and e.kind == "call" and e.target is None for e in edges)


def test_ctrl_regs_propagate_from_parent_continuation_to_child_context():
    analyzer = ProgramAnalyzer()
    all_facts = [
        # main: jump into cont_parent
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("JMPX"), index=1, offset=1),
        # cont_parent: c3 = cont_target, then jump to cont_child
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=10, offset=10, continuation_id="cont_parent"),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(3)]),
            index=11,
            offset=11,
            continuation_id="cont_parent",
        ),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=12, offset=12, continuation_id="cont_parent"),
        InstructionFact(instruction=MockInstruction("JMPX"), index=13, offset=13, continuation_id="cont_parent"),
        # cont_child: should read inherited c3 and resolve JMPX target
        InstructionFact(
            instruction=MockInstruction("PUSHCTR", args=[MockArg(3)]),
            index=20,
            offset=20,
            continuation_id="cont_child",
        ),
        InstructionFact(instruction=MockInstruction("JMPX"), index=21, offset=21, continuation_id="cont_child"),
        # cont_target entry
        InstructionFact(instruction=MockInstruction("NOP"), index=30, offset=30, continuation_id="cont_target"),
    ]
    pushcont_to_cont_ids = {
        0: ["cont_parent"],
        10: ["cont_target"],
        12: ["cont_child"],
    }
    continuations = {
        "cont_parent": Continuation(
            id="cont_parent", instructions=[], entry_index=0, parent_context="main", parent_local_index=0
        ),
        "cont_child": Continuation(
            id="cont_child", instructions=[], entry_index=0, parent_context="cont_parent", parent_local_index=2
        ),
        "cont_target": Continuation(
            id="cont_target", instructions=[], entry_index=0, parent_context="cont_parent", parent_local_index=0
        ),
    }
    cont_fact_map = {
        ("cont_parent", 0): 10,
        ("cont_child", 0): 20,
        ("cont_target", 0): 30,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(
        e.source == 21 and e.kind == "jump" and e.target == 30
        for e in edges
    )
    assert not any(
        e.source == 21 and e.kind in {"branch", "jump", "call"} and e.target is None
        for e in edges
    )


def test_unreachable_continuation_unresolved_edge_is_pruned():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("NOP"), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT"), index=10, offset=10, continuation_id="cont_dead"),
        InstructionFact(instruction=MockInstruction("IFELSE"), index=11, offset=11, continuation_id="cont_dead"),
    ]
    continuations = {
        "cont_dead": Continuation(
            id="cont_dead", instructions=[], entry_index=0, parent_context="main", parent_local_index=99
        ),
    }
    cont_fact_map = {("cont_dead", 0): 10}

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert not any(
        e.source == 11 and e.target is None and e.kind in {"branch", "jump", "call"}
        for e in edges
    )


def test_reachable_unknown_edge_uses_local_protection_and_still_prunes_other_components():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("IF"), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT"), index=10, offset=10, continuation_id="cont_dead"),
        InstructionFact(instruction=MockInstruction("IFELSE"), index=11, offset=11, continuation_id="cont_dead"),
    ]
    continuations = {
        "cont_dead": Continuation(
            id="cont_dead", instructions=[], entry_index=0, parent_context="main", parent_local_index=99
        ),
    }
    cont_fact_map = {("cont_dead", 0): 10}

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert not any(
        e.source == 11 and e.target is None and e.kind in {"branch", "jump", "call"}
        for e in edges
    )
    assert analyzer._cfg_builder.last_build_metadata.get("pruning_skipped_due_reachable_unknown") is not True
    assert analyzer._cfg_builder.last_build_metadata.get("unbounded_reachable_unknown_edge_count") == 1


def test_reachable_bounded_unknown_edge_prunes_unreachable_unresolved_in_other_component():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT"), index=1, offset=1, continuation_id=None),
        InstructionFact(instruction=MockInstruction("IF"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHINT"), index=10, offset=10, continuation_id="cont_dead"),
        InstructionFact(instruction=MockInstruction("IFELSE"), index=11, offset=11, continuation_id="cont_dead"),
    ]
    continuations = {
        "cont_dead": Continuation(
            id="cont_dead", instructions=[], entry_index=0, parent_context="main", parent_local_index=99
        ),
    }
    cont_fact_map = {("cont_dead", 0): 10}

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={0: ["cont_missing"]},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(
        e.source == 2 and e.target is None and e.kind in {"branch", "jump", "call"}
        for e in edges
    )
    assert not any(
        e.source == 11 and e.target is None and e.kind in {"branch", "jump", "call"}
        for e in edges
    )
    assert analyzer._cfg_builder.last_build_metadata.get("pruning_skipped_due_reachable_unknown") is not True


def test_ctrl_regs_propagation_uses_callsite_snapshot_not_final_context_state():
    analyzer = ProgramAnalyzer()
    all_facts = [
        # main: c3=cont_a; CALLX child; then overwrite c3=cont_b
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(3)]),
            index=1,
            offset=1,
            continuation_id=None,
        ),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=4, offset=4, continuation_id=None),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(3)]),
            index=5,
            offset=5,
            continuation_id=None,
        ),
        InstructionFact(instruction=MockInstruction("RET"), index=6, offset=6, continuation_id=None),
        # child: reads c3 and jumps through it
        InstructionFact(
            instruction=MockInstruction("PUSHCTR", args=[MockArg(3)]),
            index=7,
            offset=7,
            continuation_id="child",
        ),
        InstructionFact(instruction=MockInstruction("JMPX"), index=8, offset=8, continuation_id="child"),
        # continuation entries
        InstructionFact(instruction=MockInstruction("NOP"), index=9, offset=9, continuation_id="cont_a"),
        InstructionFact(instruction=MockInstruction("NOP"), index=10, offset=10, continuation_id="cont_b"),
    ]
    pushcont_to_cont_ids = {
        0: ["cont_a"],
        2: ["child"],
        4: ["cont_b"],
    }
    continuations = {
        "child": Continuation(
            id="child", instructions=[], entry_index=0, parent_context="main", parent_local_index=2
        ),
        "cont_a": Continuation(
            id="cont_a", instructions=[], entry_index=0, parent_context="main", parent_local_index=0
        ),
        "cont_b": Continuation(
            id="cont_b", instructions=[], entry_index=0, parent_context="main", parent_local_index=4
        ),
    }
    cont_fact_map = {
        ("child", 0): 7,
        ("cont_a", 0): 9,
        ("cont_b", 0): 10,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(
        e.source == 8 and e.kind == "jump" and e.target == 9
        for e in edges
    )
    assert not any(
        e.source == 8 and e.kind == "jump" and e.target == 10
        for e in edges
    )


def test_guard_throw_uses_known_c2_handler_target_when_propagated():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(2)]),
            index=1,
            offset=1,
            continuation_id=None,
        ),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4, continuation_id=None),
        InstructionFact(instruction=MockInstruction("THROWIF"), index=10, offset=10, continuation_id="cont_worker"),
        InstructionFact(instruction=MockInstruction("RET"), index=11, offset=11, continuation_id="cont_worker"),
        InstructionFact(instruction=MockInstruction("RET"), index=20, offset=20, continuation_id="cont_handler"),
    ]
    continuations = {
        "cont_worker": Continuation(
            id="cont_worker",
            instructions=[],
            entry_index=0,
            parent_context="main",
            parent_local_index=2,
        ),
        "cont_handler": Continuation(
            id="cont_handler",
            instructions=[],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
    }
    cont_fact_map = {
        ("cont_worker", 0): 10,
        ("cont_handler", 0): 20,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={0: ["cont_handler"], 2: ["cont_worker"]},
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(
        e.source == 10
        and e.kind == "guard_throw"
        and e.target == 20
        and bool((e.metadata or {}).get("via_c2"))
        for e in edges
    )
    assert not any(
        e.source == 10
        and e.kind == "guard_throw"
        and e.target is None
        for e in edges
    )


def test_non_c3_ctrl_regs_are_not_propagated_intercontext():
    analyzer = ProgramAnalyzer()
    all_facts = [
        # main: c0 = cont_a, then call child
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(0)]),
            index=1,
            offset=1,
            continuation_id=None,
        ),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("RET"), index=4, offset=4, continuation_id=None),
        # child: PUSHCTR c0; JMPX should remain unresolved
        InstructionFact(
            instruction=MockInstruction("PUSHCTR", args=[MockArg(0)]),
            index=7,
            offset=7,
            continuation_id="child",
        ),
        InstructionFact(instruction=MockInstruction("JMPX"), index=8, offset=8, continuation_id="child"),
        InstructionFact(instruction=MockInstruction("NOP"), index=9, offset=9, continuation_id="cont_a"),
    ]
    pushcont_to_cont_ids = {
        0: ["cont_a"],
        2: ["child"],
    }
    continuations = {
        "child": Continuation(
            id="child", instructions=[], entry_index=0, parent_context="main", parent_local_index=2
        ),
        "cont_a": Continuation(
            id="cont_a", instructions=[], entry_index=0, parent_context="main", parent_local_index=0
        ),
    }
    cont_fact_map = {
        ("child", 0): 7,
        ("cont_a", 0): 9,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    assert any(
        e.source == 8 and e.kind == "jump" and e.target is None
        for e in edges
    )
    assert not any(
        e.source == 8 and e.kind == "jump" and e.target == 9
        for e in edges
    )


def test_mixed_callsite_c3_presence_does_not_force_resolved_target():
    analyzer = ProgramAnalyzer()
    all_facts = [
        # callsite A: c3 is known
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(3)]),
            index=1,
            offset=1,
            continuation_id=None,
        ),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=3, offset=3, continuation_id=None),
        # callsite B: same callee, but c3 is overwritten with a non-continuation value
        InstructionFact(instruction=MockInstruction("PUSHINT"), index=4, offset=4, continuation_id=None),
        InstructionFact(
            instruction=MockInstruction("POPCTR", args=[MockArg(3)]),
            index=5,
            offset=5,
            continuation_id=None,
        ),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=6, offset=6, continuation_id=None),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=7, offset=7, continuation_id=None),
        InstructionFact(instruction=MockInstruction("RET"), index=8, offset=8, continuation_id=None),
        # callee continuation: uses PUSHCTR c3 + JMPX
        InstructionFact(
            instruction=MockInstruction("PUSHCTR", args=[MockArg(3)]),
            index=10,
            offset=10,
            continuation_id="child",
        ),
        InstructionFact(instruction=MockInstruction("JMPX"), index=11, offset=11, continuation_id="child"),
        InstructionFact(instruction=MockInstruction("NOP"), index=20, offset=20, continuation_id="cont_a"),
    ]
    pushcont_to_cont_ids = {
        0: ["cont_a"],
        2: ["child"],
        6: ["child"],
    }
    continuations = {
        "child": Continuation(
            id="child", instructions=[], entry_index=0, parent_context="main", parent_local_index=2
        ),
        "cont_a": Continuation(
            id="cont_a", instructions=[], entry_index=0, parent_context="main", parent_local_index=0
        ),
    }
    cont_fact_map = {
        ("child", 0): 10,
        ("cont_a", 0): 20,
    }

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    # c3 is not known across all callsites, so child JMPX must remain unresolved.
    assert any(
        e.source == 11 and e.kind == "jump" and e.target is None
        for e in edges
    )
    assert not any(
        e.source == 11 and e.kind == "jump" and e.target == 20
        for e in edges
    )


def test_callref_generates_call_cont_edge():
    """CALLREF has its continuation as an inline cell reference.

    Adding CALLREF to BRANCH_CONT_ARITY allows the resolver to pick up the
    inline continuation and produce a call_cont edge to the continuation entry,
    plus a return_cont edge back from the continuation.
    """
    analyzer = ProgramAnalyzer()
    all_facts = [
        # main context: CALLREF with inline continuation, then NOP
        InstructionFact(instruction=MockInstruction("CALLREF"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=4),
        # cont_target: the inline continuation body
        InstructionFact(instruction=MockInstruction("NOP"), index=10, offset=10, continuation_id="cont_0"),
        InstructionFact(instruction=MockInstruction("RET"), index=11, offset=11, continuation_id="cont_0"),
    ]
    inline_cont_map = {0: ["cont_0"]}
    continuations = {
        "cont_0": Continuation(
            id="cont_0", instructions=[], entry_index=0,
            parent_context="main", parent_local_index=0,
        ),
    }
    cont_fact_map = {("cont_0", 0): 10}

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids={},
        inline_cont_map=inline_cont_map,
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )

    # A call_cont edge must exist from CALLREF (index 0) to the continuation entry (index 10)
    call_cont_edges = [e for e in edges if e.source == 0 and e.kind == "call_cont"]
    assert len(call_cont_edges) == 1, (
        f"Expected 1 call_cont edge from CALLREF, got {len(call_cont_edges)}; "
        f"all edges from 0: {[(e.kind, e.target) for e in edges if e.source == 0]}"
    )
    assert call_cont_edges[0].target == 10

    # No unresolved fallthrough: when call is resolved, fallthrough is suppressed
    fallthrough_edges = [e for e in edges if e.source == 0 and e.kind == "fallthrough"]
    assert len(fallthrough_edges) == 0, (
        f"CALLREF with resolved call_cont should not have fallthrough, got {fallthrough_edges}"
    )

    # No unresolved call_return edge (only emitted when call is NOT resolved)
    call_return_edges = [e for e in edges if e.source == 0 and e.kind == "call_return"]
    assert len(call_return_edges) == 0, (
        f"CALLREF with resolved call_cont should not have call_return fallback, got {call_return_edges}"
    )

    # A return_cont edge should exist from cont_0's RET back to the instruction after CALLREF
    return_cont_edges = [e for e in edges if e.source == 11 and e.kind == "return_cont"]
    assert len(return_cont_edges) == 1, (
        f"Expected return_cont from RET (11) back to post-CALLREF, got "
        f"{[(e.kind, e.target) for e in edges if e.source == 11]}"
    )
    assert return_cont_edges[0].target == 1


def test_callref_continuation_resolver_maps_inline_cont():
    """Unit test: map_continuations correctly resolves CALLREF's inline continuation."""
    analyzer = ProgramAnalyzer()
    facts = [
        InstructionFact(instruction=MockInstruction("CALLREF"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=4),
    ]
    inline_cont_map = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = \
        analyzer._continuation_resolver.map_continuations(
            facts,
            pushcont_to_cont_ids={},
            inline_cont_map=inline_cont_map,
        )

    # CALLREF should resolve cont_0 from inline_cont_map
    assert 0 in branch_map, f"CALLREF not in branch_map: {branch_map}"
    assert branch_map[0] == ["cont_0"]

    # CALLREF is a returning call (execution resumes after the call)
    assert 0 in returning_branches

    # Return target should be the next instruction (index 1)
    assert "cont_0" in cont_targets
    assert cont_targets["cont_0"] == [1]

    # Should not be uncertain
    assert 0 not in uncertain


def test_entry_shape_widening_recovers_from_non_convergence():
    analyzer = ProgramAnalyzer()
    all_facts = [
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=0, offset=0, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=1, offset=1, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=2, offset=2, continuation_id=None),
        InstructionFact(instruction=MockInstruction("SWAP"), index=3, offset=3, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=4, offset=4, continuation_id=None),
        InstructionFact(instruction=MockInstruction("RET"), index=5, offset=5, continuation_id=None),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=6, offset=6, continuation_id="c0"),
        InstructionFact(instruction=MockInstruction("CALLCC"), index=7, offset=7, continuation_id="c0"),
        InstructionFact(instruction=MockInstruction("PUSHCONT"), index=8, offset=8, continuation_id="c0"),
        InstructionFact(instruction=MockInstruction("CALLCC"), index=9, offset=9, continuation_id="c0"),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=10, offset=10, continuation_id="c0"),
        InstructionFact(instruction=MockInstruction("EXECUTE"), index=11, offset=11, continuation_id="c0"),
        InstructionFact(
            instruction=MockInstruction("CALLXARGS", args=[MockArg(1), MockArg(0)]),
            index=12,
            offset=12,
            continuation_id="c0",
        ),
        InstructionFact(instruction=MockInstruction("RET"), index=13, offset=13, continuation_id="c0"),
        InstructionFact(instruction=MockInstruction("SWAP"), index=14, offset=14, continuation_id="c1"),
        InstructionFact(
            instruction=MockInstruction("PUSHINT", args=[MockArg(0)]),
            index=15,
            offset=15,
            continuation_id="c1",
        ),
        InstructionFact(instruction=MockInstruction("NOP"), index=16, offset=16, continuation_id="c1"),
        InstructionFact(
            instruction=MockInstruction("PUSHINT", args=[MockArg(0)]),
            index=17,
            offset=17,
            continuation_id="c1",
        ),
        InstructionFact(instruction=MockInstruction("RET"), index=18, offset=18, continuation_id="c1"),
    ]
    pushcont_to_cont_ids = {
        0: ["c0"],
        1: ["c0"],
        2: ["c0"],
        4: ["c0"],
        6: ["c1"],
        8: ["c0"],
    }
    continuations = {
        "c0": Continuation(
            id="c0",
            instructions=[],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
        "c1": Continuation(
            id="c1",
            instructions=[],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
        ),
    }
    cont_fact_map = {("c0", 0): 6, ("c1", 0): 14}

    edges = analyzer._cfg_builder.build_cfg_with_continuations(
        all_facts,
        pushcont_to_cont_ids=pushcont_to_cont_ids,
        inline_cont_map={},
        continuations=continuations,
        cont_fact_map=cont_fact_map,
    )
    meta = analyzer._cfg_builder.last_build_metadata

    assert edges
    assert meta.get("entry_shape_widened") is True
    assert meta.get("entry_shape_widened_contexts") == ["c0", "c1"]
    assert meta.get("entry_shape_converged") is True
    assert meta.get("entry_shape_pending_contexts") == []
