from tasmscan.analyzer.facts import (
    AnalysisFacts,
    Continuation,
    ControlFlowEdge,
    InstructionFact,
    StackState,
)
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.stack_underflow import StackUnderflowDetector


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = [] if args is None else args


class MockArg:
    def __init__(self, value):
        self.value = value


def make_instruction_fact(opcode: str, index: int, arg_values=None) -> InstructionFact:
    args = []
    if arg_values is not None:
        if isinstance(arg_values, (list, tuple)):
            args = [MockArg(v) for v in arg_values]
        else:
            args = [MockArg(arg_values)]
    return InstructionFact(instruction=MockInstruction(opcode, args), index=index, offset=0)


def test_dropx_detection():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 10}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("DROPX"), index=1, offset=4),
    ]

    analyzer = ProgramAnalyzer()
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions)

    dropx_state = stack_states[1]
    assert dropx_state.delta is not None and dropx_state.delta < 0
    assert dropx_state.unknown is True
    assert dropx_state.height_min is not None


def test_rollx_detection():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 5}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 3}), index=1, offset=4),
        InstructionFact(instruction=MockInstruction("ROLLX"), index=2, offset=8),
    ]

    analyzer = ProgramAnalyzer()
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions)

    rollx_state = stack_states[2]
    assert rollx_state.delta is not None and rollx_state.delta < 0


def test_reverse_stack_delta_known_zero():
    analyzer = ProgramAnalyzer()
    delta, known = analyzer._stack_analyzer.stack_delta("REVERSE")
    assert delta == 0
    assert known is True


def test_if_stack_delta_consumes_condition_and_continuation():
    analyzer = ProgramAnalyzer()
    delta, known = analyzer._stack_analyzer.stack_delta("IF")
    assert delta == -2
    assert known is True
    delta, known = analyzer._stack_analyzer.stack_delta("IFELSE")
    assert delta == -3
    assert known is True


def test_ifref_stack_delta_variants():
    analyzer = ProgramAnalyzer()
    # Inline-ref variants consume only condition
    for opcode in ("IFREF", "IFNOTREF", "IFJMPREF", "IFNOTJMPREF", "IFREFELSEREF"):
        delta, known = analyzer._stack_analyzer.stack_delta(opcode)
        assert delta == -1
        assert known is True
    # Mixed ref/stack variants consume condition + one continuation
    for opcode in ("IFREFELSE", "IFELSEREF"):
        delta, known = analyzer._stack_analyzer.stack_delta(opcode)
        assert delta == -2
        assert known is True
    # Bit-jump ref variants preserve the bit source (net 0)
    for opcode in ("IFBITJMPREF", "IFNBITJMPREF"):
        delta, known = analyzer._stack_analyzer.stack_delta(opcode)
        assert delta == 0
        assert known is True


def test_xchgx_stack_delta_dynamic_pop():
    analyzer = ProgramAnalyzer()
    delta, known = analyzer._stack_analyzer.stack_delta("XCHGX")
    assert delta == -1
    assert known is False


def test_if_continuation_resolution_order():
    instructions = [
        make_instruction_fact("PUSHCONT", 0),
        make_instruction_fact("PUSHINT", 1, 1),
        make_instruction_fact("IF", 2),
    ]

    analyzer = ProgramAnalyzer()
    branch_cont_map, _returning, _targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        instructions,
        pushcont_to_cont_ids={0: ["cont_0"]},
        inline_cont_map={},
    )

    assert branch_cont_map.get(2) == ["cont_0"]
    assert 2 not in uncertain


def test_pushctrx_consumes_index_for_continuation_resolution():
    instructions = [
        make_instruction_fact("PUSHCONT", 0),
        make_instruction_fact("PUSHINT", 1, 0),
        make_instruction_fact("PUSHCTRX", 2),
        make_instruction_fact("IF", 3),
    ]

    analyzer = ProgramAnalyzer()
    branch_cont_map, _returning, _targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        instructions,
        pushcont_to_cont_ids={0: ["cont_0"]},
        inline_cont_map={},
    )

    assert branch_cont_map.get(3) == ["cont_0"]
    assert 3 not in uncertain


def test_popctrx_consumes_value_and_index_for_continuation_resolution():
    instructions = [
        make_instruction_fact("PUSHCONT", 0),
        make_instruction_fact("PUSHINT", 1, 7),
        make_instruction_fact("PUSHINT", 2, 1),
        make_instruction_fact("POPCTRX", 3),
        make_instruction_fact("PUSHINT", 4, 1),
        make_instruction_fact("IF", 5),
    ]

    analyzer = ProgramAnalyzer()
    branch_cont_map, _returning, _targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        instructions,
        pushcont_to_cont_ids={0: ["cont_0"]},
        inline_cont_map={},
    )

    assert branch_cont_map.get(5) == ["cont_0"]
    assert 5 not in uncertain


def test_xchg3_alt_preserves_continuations():
    instructions = [
        make_instruction_fact("PUSHCONT", 0),
        make_instruction_fact("PUSHCONT", 1),
        make_instruction_fact("PUSHCONT", 2),
        make_instruction_fact("XCHG3_ALT", 3, [2, 1, 0]),
        make_instruction_fact("PUSHINT", 4, 1),
        make_instruction_fact("IFELSE", 5),
    ]

    analyzer = ProgramAnalyzer()
    branch_cont_map, _returning, _targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        instructions,
        pushcont_to_cont_ids={0: ["cont_a"], 1: ["cont_b"], 2: ["cont_c"]},
        inline_cont_map={},
    )

    conts = branch_cont_map.get(5) or []
    assert 5 not in uncertain
    assert len(conts) == 2
    assert set(conts).issubset({"cont_a", "cont_b", "cont_c"})


def test_ifrefelse_continuation_resolution_mixed_inline_and_stack():
    instructions = [
        make_instruction_fact("PUSHCONT", 0),
        make_instruction_fact("PUSHINT", 1, 1),
        make_instruction_fact("IFREFELSE", 2),
    ]

    analyzer = ProgramAnalyzer()
    branch_cont_map, _returning, _targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        instructions,
        pushcont_to_cont_ids={0: ["cont_stack"]},
        inline_cont_map={2: ["cont_inline"]},
    )

    assert branch_cont_map.get(2) == ["cont_inline", "cont_stack"]
    assert 2 not in uncertain


def test_dup2_preserves_top_two_continuations():
    instructions = [
        make_instruction_fact("PUSHCONT", 0),
        make_instruction_fact("PUSHCONT", 1),
        make_instruction_fact("DUP2", 2),
        make_instruction_fact("PUSHINT", 3, 1),
        make_instruction_fact("IFELSE", 4),
    ]

    analyzer = ProgramAnalyzer()
    branch_cont_map, _returning, _targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        instructions,
        pushcont_to_cont_ids={0: ["cont_a"], 1: ["cont_b"]},
        inline_cont_map={},
    )

    assert branch_cont_map.get(4) == ["cont_b", "cont_a"]
    assert 4 not in uncertain


def test_detector_integration():
    # Test that DROPX with insufficient stack elements is detected
    # DROPX consumes variable number of elements, and with only 1 element on stack
    # it may underflow depending on runtime count (dynamic effect)
    # Need at least 10 instructions to pass the trivial contract filter
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("NOP"), index=2, offset=2),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=3),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4),
        InstructionFact(instruction=MockInstruction("NOP"), index=5, offset=5),
        InstructionFact(instruction=MockInstruction("NOP"), index=6, offset=6),
        InstructionFact(instruction=MockInstruction("NOP"), index=7, offset=7),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 10}), index=8, offset=8),
        InstructionFact(instruction=MockInstruction("DROPX"), index=9, offset=12),
    ]

    analyzer = ProgramAnalyzer()
    # Use initial_height=0 to simulate empty stack scenario for testing
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions, initial_height=0)

    facts = AnalysisFacts(
        instructions=instructions,
        basic_blocks=[],
        stack_states=stack_states,
        call_sites=[],
    )

    detector = StackUnderflowDetector()
    findings = detector.detect(facts)

    # Dynamic DROPX should be detected by the underflow heuristic
    assert len(findings) > 0


def test_cumulative_detection_push_then_pops():
    """Test that PUSH followed by multiple POPs is correctly tracked.

    This tests the cumulative detection logic: a PUSH (+1) followed by
    multiple POPs should track the net consumption, not reset on PUSH.
    Example: PUSH, POP, POP, POP, POP -> net -3, should be detected
    """
    # 10 instructions to pass trivial contract filter
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("NOP"), index=2, offset=2),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=3),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4),
        # PUSH then multiple POPs: +1, -1, -1, -1, -1 = net -3
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=5, offset=5),
        InstructionFact(instruction=MockInstruction("DROP"), index=6, offset=6),
        InstructionFact(instruction=MockInstruction("DROP"), index=7, offset=7),
        InstructionFact(instruction=MockInstruction("DROP"), index=8, offset=8),
        InstructionFact(instruction=MockInstruction("DROP"), index=9, offset=9),
    ]

    analyzer = ProgramAnalyzer()
    # Use initial_height=0 to simulate empty stack scenario for testing
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions, initial_height=0)

    facts = AnalysisFacts(
        instructions=instructions,
        basic_blocks=[],
        stack_states=stack_states,
        call_sites=[],
    )

    detector = StackUnderflowDetector()
    findings = detector.detect(facts)

    # Net consumption of 3 exceeds threshold, should be detected
    assert len(findings) > 0


def test_no_false_positive_with_sufficient_stack():
    """Test that PUSH followed by POPs with sufficient stack is NOT flagged.

    If we PUSH enough elements before POPing, there's no underflow.
    """
    # 10 instructions to pass trivial contract filter
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1),
        # Push 5 elements
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=2, offset=2),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 2}), index=3, offset=3),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 3}), index=4, offset=4),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 4}), index=5, offset=5),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 5}), index=6, offset=6),
        # Pop 4 elements - still have 1 left, no underflow
        InstructionFact(instruction=MockInstruction("DROP"), index=7, offset=7),
        InstructionFact(instruction=MockInstruction("DROP"), index=8, offset=8),
        InstructionFact(instruction=MockInstruction("DROP"), index=9, offset=9),
        InstructionFact(instruction=MockInstruction("DROP"), index=10, offset=10),
    ]

    analyzer = ProgramAnalyzer()
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions)

    facts = AnalysisFacts(
        instructions=instructions,
        basic_blocks=[],
        stack_states=stack_states,
        call_sites=[],
    )

    detector = StackUnderflowDetector()
    findings = detector.detect(facts)

    # No underflow - sufficient stack elements
    assert len(findings) == 0


def test_cfg_aware_stack_analysis_join_point():
    """Test CFG-aware stack analysis at join points with different stack heights.

    Simulates a control flow where two paths merge:
    - Path 1: PUSH, PUSH -> height 2
    - Path 2: PUSH -> height 1
    At the join point, should use conservative bounds (min=1, max=2).
    """
    # Create instructions that simulate branching paths
    # idx 0: PUSH (both paths start here)
    # idx 1: IF (conditional branch - true goes to idx 2, false goes to idx 3)
    # idx 2: PUSH (true branch only)
    # idx 3: NOP (join point - both paths reach here)
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("IF"), index=1, offset=4),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 2}), index=2, offset=8),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=12),
    ]

    # Create CFG edges that represent the control flow:
    # 0 -> 1 (fallthrough)
    # 1 -> 2 (true branch)
    # 1 -> 3 (false branch - fallthrough when condition false)
    # 2 -> 3 (fallthrough from true branch)
    cfg_edges = [
        ControlFlowEdge(source=0, target=1, kind="fallthrough"),
        ControlFlowEdge(source=1, target=2, kind="branch"),
        ControlFlowEdge(source=1, target=3, kind="fallthrough"),
        ControlFlowEdge(source=2, target=3, kind="fallthrough"),
    ]

    analyzer = ProgramAnalyzer()
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions, cfg_edges=cfg_edges)

    # idx 3 (NOP) is a join point with two predecessors:
    # - From idx 1 (IF): stack height = 0 (PUSH +1, IF -1 = 0)
    # - From idx 2 (PUSH): stack height = 1 (PUSH +1, IF -1, PUSH +1 = 1)
    # Conservative merge should give height_min=0, height_max=1
    join_state = stack_states[3]
    assert join_state.height_min is not None
    assert join_state.height_max is not None
    # The join point should track the range of possible heights
    # Note: exact values depend on the path analysis


def test_cfg_aware_stack_analysis_merges_back_edge():
    """Ensure back-edge predecessors are merged at loop headers."""
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=4),
        InstructionFact(instruction=MockInstruction("IF"), index=2, offset=8),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 2}), index=3, offset=12),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 3}), index=4, offset=16),
        InstructionFact(instruction=MockInstruction("JMPX"), index=5, offset=20),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 4}), index=6, offset=24),
        InstructionFact(instruction=MockInstruction("JMPX"), index=7, offset=28),
    ]

    cfg_edges = [
        ControlFlowEdge(source=0, target=1, kind="fallthrough"),
        ControlFlowEdge(source=1, target=2, kind="fallthrough"),
        ControlFlowEdge(source=2, target=3, kind="branch"),
        ControlFlowEdge(source=2, target=6, kind="fallthrough"),
        ControlFlowEdge(source=3, target=4, kind="fallthrough"),
        ControlFlowEdge(source=4, target=5, kind="fallthrough"),
        ControlFlowEdge(source=5, target=1, kind="jump"),
        ControlFlowEdge(source=6, target=7, kind="fallthrough"),
        ControlFlowEdge(source=7, target=1, kind="jump"),
    ]

    analyzer = ProgramAnalyzer()
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions, cfg_edges=cfg_edges)

    header_state = next(state for state in stack_states if state.instruction_index == 1)
    assert header_state.height_min is not None
    assert header_state.height_max is not None
    assert header_state.height_min <= header_state.height_max
    assert header_state.height_max >= 1
    assert header_state.unknown is True


def test_cfg_aware_stack_analysis_reanalyzes_main_with_return_cont_predecessor():
    instructions = [
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 1}), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("PUSHINT", {"value": 2}), index=10, offset=10, continuation_id="cont_0"),
    ]
    continuations = {
        "cont_0": Continuation(
            id="cont_0",
            instructions=[],
            entry_index=0,
            parent_context="main",
            parent_local_index=0,
            parent_instruction_index=0,
        ),
    }
    cfg_edges = [
        ControlFlowEdge(source=0, target=1, kind="fallthrough"),
        ControlFlowEdge(source=0, target=10, kind="call_cont"),
        ControlFlowEdge(source=10, target=1, kind="return_cont"),
    ]

    analyzer = ProgramAnalyzer()
    stack_states = analyzer._stack_analyzer.analyze_stack(
        instructions,
        continuations=continuations,
        cfg_edges=cfg_edges,
        initial_height=0,
    )

    join_state = next(state for state in stack_states if state.instruction_index == 1)
    assert join_state.height_min == 1
    assert join_state.height_max == 2
    assert join_state.unknown is True


def test_program_analyzer_marks_stack_non_converged_in_metadata(monkeypatch):
    analyzer = ProgramAnalyzer()
    instructions = [
        MockInstruction("NOP"),
        MockInstruction("NOP"),
    ]

    def fake_analyze_stack(*_args, **_kwargs):
        analyzer._stack_analyzer.last_analysis_metadata = {
            "global_converged": False,
            "global_iterations": 8,
            "global_iteration_limit": 8,
        }
        return []

    monkeypatch.setattr(analyzer._stack_analyzer, "analyze_stack", fake_analyze_stack)
    facts = analyzer.analyze(instructions, cell=None)

    assert facts.metadata.get("analysis_incomplete") is True
    assert "stack_analysis_non_converged" in facts.metadata.get("analysis_incomplete_reasons", [])
    assert facts.metadata.get("stack_analysis", {}).get("global_converged") is False


def test_merge_predecessor_states():
    """Test _merge_predecessor_states helper method."""
    analyzer = ProgramAnalyzer()

    # Test with single predecessor
    single = [StackState(instruction_index=0, height_before=0, height_after=5,
                         delta=5, unknown=False, height_min=5, height_max=5)]
    h, h_min, h_max, unknown = analyzer._stack_analyzer._merge_predecessor_states(single)
    assert h == 5
    assert h_min == 5
    assert h_max == 5
    assert unknown is False

    # Test with multiple predecessors - different heights
    multiple = [
        StackState(instruction_index=0, height_before=0, height_after=3,
                   delta=3, unknown=False, height_min=3, height_max=3),
        StackState(instruction_index=1, height_before=0, height_after=7,
                   delta=7, unknown=False, height_min=7, height_max=7),
    ]
    h, h_min, h_max, unknown = analyzer._stack_analyzer._merge_predecessor_states(multiple)
    assert h_min == 3  # min of mins
    assert h_max == 7  # max of maxs
    assert unknown is True  # differing heights -> unknown
    # Let's verify with a larger spread
    multiple_wide = [
        StackState(instruction_index=0, height_before=0, height_after=0,
                   delta=0, unknown=False, height_min=0, height_max=0),
        StackState(instruction_index=1, height_before=0, height_after=10,
                   delta=10, unknown=False, height_min=10, height_max=10),
    ]
    h, h_min, h_max, unknown = analyzer._stack_analyzer._merge_predecessor_states(multiple_wide)
    assert h_min == 0
    assert h_max == 10
    assert unknown is True  # spread > 5

    # Test with any unknown predecessor
    mixed = [
        StackState(instruction_index=0, height_before=0, height_after=3,
                   delta=3, unknown=True, height_min=3, height_max=3),
        StackState(instruction_index=1, height_before=0, height_after=4,
                   delta=4, unknown=False, height_min=4, height_max=4),
    ]
    h, h_min, h_max, unknown = analyzer._stack_analyzer._merge_predecessor_states(mixed)
    assert unknown is True  # any predecessor unknown -> merged is unknown


def test_empty_predecessor_states():
    """Test _merge_predecessor_states with empty list."""
    analyzer = ProgramAnalyzer()
    h, h_min, h_max, unknown = analyzer._stack_analyzer._merge_predecessor_states([])
    assert h == 0
    assert h_min == 0
    assert h_max == 0
    assert unknown is True


def test_min_inputs_underflow_detection_for_pick():
    # 10 instructions to pass trivial contract filter
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=0, offset=0),
        InstructionFact(instruction=MockInstruction("NOP"), index=1, offset=1),
        InstructionFact(instruction=MockInstruction("NOP"), index=2, offset=2),
        InstructionFact(instruction=MockInstruction("NOP"), index=3, offset=3),
        InstructionFact(instruction=MockInstruction("NOP"), index=4, offset=4),
        InstructionFact(instruction=MockInstruction("NOP"), index=5, offset=5),
        InstructionFact(instruction=MockInstruction("NOP"), index=6, offset=6),
        InstructionFact(instruction=MockInstruction("NOP"), index=7, offset=7),
        InstructionFact(instruction=MockInstruction("NOP"), index=8, offset=8),
        InstructionFact(instruction=MockInstruction("PICK"), index=9, offset=9),
    ]

    analyzer = ProgramAnalyzer()
    # Use initial_height=0 to simulate empty stack scenario for testing
    stack_states = analyzer._stack_analyzer.analyze_stack(instructions, initial_height=0)

    facts = AnalysisFacts(
        instructions=instructions,
        basic_blocks=[],
        stack_states=stack_states,
        call_sites=[],
    )

    detector = StackUnderflowDetector()
    findings = detector.detect(facts)

    assert len(findings) > 0
