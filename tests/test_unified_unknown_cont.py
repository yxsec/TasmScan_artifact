from tasmscan.analyzer.continuation_resolver import (
    ContinuationResolver,
    UncertainInfo,
    is_unknown_cont,
    make_unknown_cont,
    parse_unknown_cont,
)
from tasmscan.analyzer.facts import InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.ir import IRLifter
from tasmscan.ir.continuation_linker import ContinuationLinker
from tasmscan.ir.tasir_types import (
    ContinuationDescriptor,
    InstructionKind,
    RegisterLocation,
    TVMBasicBlock,
    TVMFunction,
    TVMInstruction,
    TVMModule,
)
from tasmscan.solver import DynamicTargetSolver


class DummyInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class DummyArg:
    def __init__(self, value):
        self.value = value


def _make_fact(idx: int, opcode: str, args=None) -> InstructionFact:
    return InstructionFact(
        instruction=DummyInstruction(opcode, args=args or []),
        index=idx,
        offset=0,
        length=0,
        hash="test",
        continuation_id=None,
        parent_index=None,
    )


def _make_module_with_method_contexts() -> TVMModule:
    return TVMModule(
        functions={
            100: TVMFunction(method_id=100, metadata={"context": "cont_a"}),
            200: TVMFunction(method_id=200, metadata={"context": "cont_b"}),
        },
        cross_function_continuations={
            "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
            "cont_b": ContinuationDescriptor(code_ref="cont_b"),
        },
    )


def test_unknown_cont_roundtrip_with_underscored_opcode():
    cont_id = make_unknown_cont(42, "POPCTR_c3")
    assert is_unknown_cont(cont_id)
    assert parse_unknown_cont(cont_id) == (42, "POPCTR_c3")


def test_unknown_cont_legacy_compatibility():
    assert is_unknown_cont("__unknown__")
    assert parse_unknown_cont("__unknown__") is None


def test_map_continuations_records_stack_invalidated_reason():
    resolver = ContinuationResolver()
    facts = [
        _make_fact(0, "PUSHCONT"),
        _make_fact(1, "ROLL"),
        _make_fact(2, "PUSHINT", args=[DummyArg(1)]),
        _make_fact(3, "IF"),
    ]
    branch_map, _returning, _targets, uncertain, _ = resolver.map_continuations(
        facts=facts,
        pushcont_to_cont_ids={0: ["cont_0"]},
        inline_cont_map={},
    )

    assert 3 not in branch_map
    assert 3 in uncertain
    assert any(
        info.reason == "stack_invalidated" and info.source_index == 1
        for info in uncertain[3]
    )


def test_program_analyzer_and_ir_lifter_propagate_uncertain_branches_metadata():
    analyzer = ProgramAnalyzer()
    instructions = [
        DummyInstruction("PUSHCONT"),
        DummyInstruction("ROLL"),
        DummyInstruction("PUSHINT", args=[DummyArg(1)]),
        DummyInstruction("IF"),
        DummyInstruction("NOP"),
    ]
    facts = analyzer.analyze(instructions)

    uncertain_map = facts.metadata.get("uncertain_branches", {})
    assert isinstance(uncertain_map, dict)
    assert 3 in uncertain_map
    assert any(info.reason == "stack_invalidated" for info in uncertain_map[3])

    module = IRLifter().lift(facts)
    lifted_uncertain_map = module.analysis_metadata.get("uncertain_branches", {})
    assert isinstance(lifted_uncertain_map, dict)
    assert 3 in lifted_uncertain_map


def test_solver_reason_stack_invalidated_does_not_fallback_to_pushctr_c3():
    module = _make_module_with_method_contexts()
    prev_inst = TVMInstruction(
        index=20,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    inst = TVMInstruction(index=21, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
    solver = DynamicTargetSolver()

    result = solver.solve(
        inst=inst,
        module=module,
        context_id="cont_entry",
        prev_inst=prev_inst,
        uncertain_reasons=[
            UncertainInfo("stack_invalidated", 19, "ROLL", "unmodeled_shuffle")
        ],
    )

    assert result.status == "unknown"
    assert result.strategy == "unified_reason_solver"
    assert result.reason == "no_candidates_from_reasons"
    assert result.candidate_continuations == []


def test_solver_reason_bless_returns_method_candidates():
    module = _make_module_with_method_contexts()
    inst = TVMInstruction(index=30, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
    solver = DynamicTargetSolver()

    result = solver.solve(
        inst=inst,
        module=module,
        context_id="cont_entry",
        uncertain_reasons=[UncertainInfo("bless", 29, "BLESSARGS", "")],
    )

    assert result.status == "sat"
    assert result.strategy == "unified_reason_solver"
    assert set(result.candidate_continuations) == {"cont_a", "cont_b"}


def test_solver_pattern_matches_blessargs():
    module = _make_module_with_method_contexts()
    pre_prev_inst = TVMInstruction(index=40, kind=InstructionKind.CELL_CONVERT, opcode="CTOS")
    prev_inst = TVMInstruction(index=41, kind=InstructionKind.CONT_CREATE, opcode="BLESSARGS")
    inst = TVMInstruction(index=42, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
    solver = DynamicTargetSolver()

    result = solver.solve(
        inst=inst,
        module=module,
        context_id="cont_entry",
        prev_inst=prev_inst,
        pre_prev_inst=pre_prev_inst,
    )

    assert result.status == "sat"
    assert result.strategy == "bless_execute_dynamic_cell"


def test_continuation_linker_passes_uncertain_reasons_to_solver():
    pushctr = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    execute = TVMInstruction(index=1, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")

    module = TVMModule(
        functions={
            0: TVMFunction(
                method_id=0,
                blocks={1: TVMBasicBlock(id=1, context_id="cont_entry", instructions=[pushctr, execute])},
                metadata={"context": "cont_entry"},
            ),
            100: TVMFunction(method_id=100, metadata={"context": "cont_a"}),
            200: TVMFunction(method_id=200, metadata={"context": "cont_b"}),
        },
        cross_function_continuations={
            "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
            "cont_b": ContinuationDescriptor(code_ref="cont_b"),
        },
        analysis_metadata={
            "uncertain_branches": {
                1: [UncertainInfo("stack_invalidated", 0, "ROLL", "unmodeled_shuffle")]
            }
        },
    )

    linked = ContinuationLinker().link(module)
    assert "__stack__" in execute.continuation_refs
    assert linked.analysis_metadata.get("dynamic_target_count") == 1
