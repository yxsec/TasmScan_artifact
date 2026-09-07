from tasmscan.analyzer.facts import AnalysisFacts, BasicBlock, InstructionFact
from tasmscan.ir.path_sensitive_dataflow import PathSensitiveDataFlowAnalyzer, PathState
from tasmscan.ir.dataflow import DataFlowState, DataFlowValue
from tasmscan.ir.dataflow.types import ValueSource


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


def _facts_with_self_loop(opcode: str) -> AnalysisFacts:
    inst = InstructionFact(instruction=MockInstruction(opcode), index=0)
    block = BasicBlock(id=0, instruction_indices=[0], successors=[0], context="main")
    return AnalysisFacts(instructions=[inst], basic_blocks=[block])


def _facts_with_linear_chain(length: int) -> AnalysisFacts:
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=i)
        for i in range(length)
    ]
    blocks = []
    for i in range(length):
        successors = [i + 1] if i + 1 < length else []
        blocks.append(
            BasicBlock(
                id=i,
                instruction_indices=[i],
                successors=successors,
                context="main",
            )
        )
    return AnalysisFacts(instructions=instructions, basic_blocks=blocks)


def _facts_with_diamond_cfg() -> AnalysisFacts:
    instructions = [
        InstructionFact(instruction=MockInstruction("NOP"), index=i)
        for i in range(4)
    ]
    blocks = [
        BasicBlock(id=0, instruction_indices=[0], successors=[1, 2], context="main"),
        BasicBlock(id=1, instruction_indices=[1], successors=[3], context="main"),
        BasicBlock(id=2, instruction_indices=[2], successors=[3], context="main"),
        BasicBlock(id=3, instruction_indices=[3], successors=[], context="main"),
    ]
    return AnalysisFacts(instructions=instructions, basic_blocks=blocks)


def test_scc_fixpoint_converges_simple_self_loop_without_truncation():
    facts = _facts_with_self_loop("NOP")
    analyzer = PathSensitiveDataFlowAnalyzer(
        max_loop_unroll=1,
        max_analysis_depth=10,
        max_block_visits=1,
        max_worklist_size=100,
    )

    graph = analyzer.analyze(facts)
    meta = graph.analysis_metadata

    assert graph.analysis_type == "path_sensitive"
    assert meta.get("analysis_incomplete") is False
    assert meta.get("truncated_loop_count") == 0
    assert meta.get("loop_fixpoint_edge_count", 0) >= 1
    assert meta.get("loop_fixpoint_update_count", 0) >= 1


def test_scc_fixpoint_still_fails_closed_when_iteration_budget_exhausted():
    # LDMSGADDR pushes tainted value every iteration; stack keeps growing.
    # With tiny SCC budget, solver must stop and mark loop truncation.
    facts = _facts_with_self_loop("LDMSGADDR")
    analyzer = PathSensitiveDataFlowAnalyzer(
        max_loop_unroll=1,
        max_analysis_depth=50,
        max_block_visits=100,
        max_worklist_size=1000,
    )
    analyzer.max_scc_iterations = 2

    graph = analyzer.analyze(facts)
    meta = graph.analysis_metadata
    reasons = set(meta.get("analysis_incomplete_reasons", []))

    assert meta.get("analysis_incomplete") is True
    assert meta.get("truncated_loop_count", 0) > 0
    assert "path_truncation" in reasons


def test_scc_fixpoint_loop_widening_converges_stack_growth():
    # LDMSGADDR grows stack on each loop iteration; widening should cap loop
    # summaries and allow convergence under normal SCC iteration budget.
    facts = _facts_with_self_loop("LDMSGADDR")
    analyzer = PathSensitiveDataFlowAnalyzer(
        max_loop_unroll=1,
        max_analysis_depth=50,
        max_block_visits=100,
        max_worklist_size=1000,
    )

    graph = analyzer.analyze(facts)
    meta = graph.analysis_metadata

    assert meta.get("analysis_incomplete") is False
    assert meta.get("truncated_loop_count", 0) == 0
    assert meta.get("loop_widened_states", 0) > 0


def test_adaptive_depth_budget_avoids_false_truncation_on_long_linear_cfg():
    # Depth limit should scale with CFG size so long acyclic chains do not
    # become incomplete purely due to fixed max_analysis_depth.
    facts = _facts_with_linear_chain(80)
    analyzer = PathSensitiveDataFlowAnalyzer(
        max_analysis_depth=10,
        max_block_visits=2,
        max_worklist_size=5000,
    )

    graph = analyzer.analyze(facts)
    meta = graph.analysis_metadata

    assert meta.get("analysis_incomplete") is False
    assert meta.get("truncated_count", 0) == 0
    assert meta.get("effective_depth_limit", 0) >= 80


def test_state_subsumption_respects_taint_and_guard_conservativeness():
    analyzer = PathSensitiveDataFlowAnalyzer()
    lhs = PathState(
        path_id=1,
        dataflow_state=DataFlowState(
            stack=[
                DataFlowValue(
                    source=ValueSource.UNKNOWN,
                    definition_site=1,
                    tainted=True,
                    checked=False,
                )
            ],
            registers={},
            guarded_values={},
        ),
    )
    rhs = PathState(
        path_id=2,
        dataflow_state=DataFlowState(
            stack=[
                DataFlowValue(
                    source=ValueSource.UNKNOWN,
                    definition_site=1,
                    tainted=True,
                    checked=True,
                )
            ],
            registers={},
            guarded_values={1: 1},
        ),
    )

    assert analyzer._state_subsumes(lhs, rhs) is True
    assert analyzer._state_subsumes(rhs, lhs) is False


def test_state_subsumption_requires_matching_stack_depth():
    analyzer = PathSensitiveDataFlowAnalyzer()
    lhs = PathState(
        path_id=1,
        dataflow_state=DataFlowState(
            stack=[],
            registers={},
            guarded_values={},
        ),
    )
    rhs = PathState(
        path_id=2,
        dataflow_state=DataFlowState(
            stack=[
                DataFlowValue(
                    source=ValueSource.UNKNOWN,
                    definition_site=1,
                    tainted=True,
                    checked=False,
                )
            ],
            registers={},
            guarded_values={},
        ),
    )

    assert analyzer._state_subsumes(lhs, rhs) is False


def test_enqueue_frontier_subsumption_skips_duplicate_diamond_state():
    facts = _facts_with_diamond_cfg()
    analyzer = PathSensitiveDataFlowAnalyzer(
        max_analysis_depth=20,
        max_block_visits=10,
        max_worklist_size=100,
    )

    graph = analyzer.analyze(facts)
    meta = graph.analysis_metadata

    assert meta.get("analysis_incomplete") is False
    assert meta.get("subsumed_enqueue_skips", 0) >= 1


def test_state_subsumption_handles_register_values():
    analyzer = PathSensitiveDataFlowAnalyzer()
    lhs = PathState(
        path_id=1,
        dataflow_state=DataFlowState(
            stack=[],
            registers={
                3: DataFlowValue(
                    source=ValueSource.UNKNOWN,
                    definition_site=10,
                    tainted=True,
                    checked=False,
                )
            },
            guarded_values={},
        ),
    )
    rhs = PathState(
        path_id=2,
        dataflow_state=DataFlowState(
            stack=[],
            registers={
                3: DataFlowValue(
                    source=ValueSource.UNKNOWN,
                    definition_site=10,
                    tainted=True,
                    checked=True,
                )
            },
            guarded_values={},
        ),
    )

    assert analyzer._state_subsumes(lhs, rhs) is True
    assert analyzer._state_subsumes(rhs, lhs) is False


def test_truncation_accounting_uses_unique_path_hits_without_double_counting():
    facts = _facts_with_self_loop("LDMSGADDR")
    analyzer = PathSensitiveDataFlowAnalyzer(
        max_loop_unroll=1,
        max_analysis_depth=20,
        max_block_visits=1,
        max_worklist_size=100,
    )
    analyzer.max_scc_iterations = 100

    graph = analyzer.analyze(facts)
    meta = graph.analysis_metadata

    assert meta.get("truncated_count", 0) >= 1
    assert meta.get("truncated_unique_count") == len(set(meta.get("truncated_paths", [])))
    assert meta.get("resource_truncation_count", 0) == 0
