import subprocess
import sys

from tasmscan.ir.continuation_linker import ContinuationLinker
from tasmscan.ir.dataflow import DataFlowAnalyzer
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


def _make_module_with_method_contexts() -> TVMModule:
    conts = {
        "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
        "cont_a": ContinuationDescriptor(code_ref="cont_a"),
        "cont_b": ContinuationDescriptor(code_ref="cont_b"),
    }
    return TVMModule(
        functions={
            100: TVMFunction(method_id=100, metadata={"context": "cont_a"}),
            200: TVMFunction(method_id=200, metadata={"context": "cont_b"}),
        },
        cross_function_continuations=conts,
    )


def _make_large_method_context_module(count: int = 70) -> TVMModule:
    functions = {}
    continuations = {}
    for i in range(count):
        cont_id = f"cont_{i}"
        functions[1000 + i] = TVMFunction(method_id=1000 + i, metadata={"context": cont_id})
        continuations[cont_id] = ContinuationDescriptor(code_ref=cont_id)
    return TVMModule(functions=functions, cross_function_continuations=continuations)


def test_dynamic_target_solver_resolves_dict_immediate_to_method_context():
    module = _make_module_with_method_contexts()
    inst = TVMInstruction(
        index=10,
        kind=InstructionKind.CONT_CALL,
        opcode="CALLDICT",
        immediates=[100],
    )
    solver = DynamicTargetSolver()
    result = solver.solve(inst=inst, module=module, context_id="cont_entry", prev_inst=None)

    assert result.status == "sat"
    assert result.candidate_continuations == ["cont_a"]
    assert result.strategy == "dict_dispatch_immediate"


def test_dynamic_target_solver_resolves_jmpdict_long_immediate_to_method_context():
    module = _make_module_with_method_contexts()
    inst = TVMInstruction(
        index=11,
        kind=InstructionKind.CONT_JUMP,
        opcode="JMPDICT_LONG",
        immediates=[200],
    )
    solver = DynamicTargetSolver()
    result = solver.solve(inst=inst, module=module, context_id="cont_entry", prev_inst=None)

    assert result.status == "sat"
    assert result.candidate_continuations == ["cont_b"]
    assert result.strategy == "dict_dispatch_immediate"


def test_dynamic_target_solver_resolves_pushctr_c3_jmpx_to_method_candidates():
    module = _make_module_with_method_contexts()
    prev_inst = TVMInstruction(
        index=20,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    inst = TVMInstruction(index=21, kind=InstructionKind.CONT_JUMP, opcode="JMPX")
    solver = DynamicTargetSolver()

    result = solver.solve(inst=inst, module=module, context_id="cont_entry", prev_inst=prev_inst)

    assert result.status == "sat"
    assert set(result.candidate_continuations) == {"cont_a", "cont_b"}
    assert result.strategy == "pushctr_c3_method_dict"


def test_dynamic_target_solver_uses_cache_for_repeated_pattern():
    module = _make_module_with_method_contexts()
    prev_inst = TVMInstruction(
        index=30,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    inst = TVMInstruction(index=31, kind=InstructionKind.CONT_JUMP, opcode="JMPX")
    solver = DynamicTargetSolver()

    first = solver.solve(inst=inst, module=module, context_id="cont_entry", prev_inst=prev_inst)
    second = solver.solve(inst=inst, module=module, context_id="cont_entry", prev_inst=prev_inst)

    assert first.from_cache is False
    assert second.from_cache is True
    stats = solver.stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


def test_dynamic_target_solver_cache_key_includes_instruction_continuation_refs():
    module = _make_module_with_method_contexts()
    solver = DynamicTargetSolver()

    inst_a = TVMInstruction(
        index=32,
        kind=InstructionKind.CONT_CALL,
        opcode="CALLREF",
        continuation_refs=["cont_a"],
    )
    inst_b = TVMInstruction(
        index=33,
        kind=InstructionKind.CONT_CALL,
        opcode="CALLREF",
        continuation_refs=["cont_b"],
    )

    res_a = solver.solve(inst=inst_a, module=module, context_id="cont_entry")
    res_b = solver.solve(inst=inst_b, module=module, context_id="cont_entry")

    assert res_a.status == "sat"
    assert res_a.candidate_continuations == ["cont_a"]
    assert res_b.status == "sat"
    assert res_b.candidate_continuations == ["cont_b"]
    assert res_b.from_cache is False


def test_solver_package_import_works_without_prior_ir_import():
    proc = subprocess.run(
        [sys.executable, "-c", "from tasmscan.solver import DynamicTargetSolver; print(DynamicTargetSolver.__name__)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "DynamicTargetSolver"


def test_dynamic_target_solver_limits_large_pushctr_c3_candidate_sets():
    module = _make_large_method_context_module()
    prev_inst = TVMInstruction(
        index=40,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    inst = TVMInstruction(index=41, kind=InstructionKind.CONT_JUMP, opcode="JMPX")
    solver = DynamicTargetSolver()

    result = solver.solve(inst=inst, module=module, context_id="cont_0", prev_inst=prev_inst)

    assert result.status == "unknown"
    assert result.reason == "candidate_set_too_large"
    assert result.candidate_continuations == []


def test_dynamic_target_solver_marks_execute_truncation_as_unknown():
    module = _make_large_method_context_module()
    prev_inst = TVMInstruction(
        index=50,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    inst = TVMInstruction(index=51, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
    solver = DynamicTargetSolver()

    result = solver.solve(inst=inst, module=module, context_id="cont_0", prev_inst=prev_inst)

    assert result.status == "unknown"
    assert result.strategy == "pushctr_c3_method_dict_truncated"
    assert result.reason == "candidate_set_truncated"
    assert len(result.candidate_continuations) == solver.MAX_C3_DISPATCH_CANDIDATES


def test_dynamic_target_solver_uses_pushint_method_hint_for_execute():
    module = _make_large_method_context_module()
    pre_prev_inst = TVMInstruction(
        index=49,
        kind=InstructionKind.STACK_PUSH,
        opcode="PUSHINT_LONG",
        immediates=[1007],
    )
    prev_inst = TVMInstruction(
        index=50,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    inst = TVMInstruction(index=51, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
    solver = DynamicTargetSolver()

    result = solver.solve(
        inst=inst,
        module=module,
        context_id="cont_0",
        prev_inst=prev_inst,
        pre_prev_inst=pre_prev_inst,
    )

    assert result.status == "sat"
    assert result.strategy == "pushctr_c3_method_id_hint"
    assert result.candidate_continuations == ["cont_7"]


def test_dynamic_target_solver_resolves_bless_execute_to_method_candidates():
    module = _make_module_with_method_contexts()
    pre_prev_inst = TVMInstruction(
        index=59,
        kind=InstructionKind.CELL_CONVERT,
        opcode="CTOS",
    )
    prev_inst = TVMInstruction(
        index=60,
        kind=InstructionKind.CONT_CREATE,
        opcode="BLESS",
    )
    inst = TVMInstruction(index=61, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
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
    assert set(result.candidate_continuations) == {"cont_a", "cont_b"}


def test_dynamic_target_solver_resolves_execute_from_prev_pushcont():
    module = _make_module_with_method_contexts()
    prev_inst = TVMInstruction(
        index=70,
        kind=InstructionKind.CONT_CREATE,
        opcode="PUSHCONT",
        continuation_refs=["cont_a"],
    )
    inst = TVMInstruction(index=71, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")
    solver = DynamicTargetSolver()

    result = solver.solve(
        inst=inst,
        module=module,
        context_id="cont_entry",
        prev_inst=prev_inst,
        pre_prev_inst=None,
    )

    assert result.status == "sat"
    assert result.strategy == "stack_dispatch_prev_cont_create"
    assert result.candidate_continuations == ["cont_a"]


def test_dynamic_target_solver_resolves_callref_from_prev_pushrefcont():
    module = _make_module_with_method_contexts()
    prev_inst = TVMInstruction(
        index=80,
        kind=InstructionKind.CONT_CREATE,
        opcode="PUSHREFCONT",
        continuation_refs=["cont_b"],
    )
    inst = TVMInstruction(index=81, kind=InstructionKind.CONT_CALL, opcode="CALLREF")
    solver = DynamicTargetSolver()

    result = solver.solve(
        inst=inst,
        module=module,
        context_id="cont_entry",
        prev_inst=prev_inst,
        pre_prev_inst=None,
    )

    assert result.status == "sat"
    assert result.strategy == "ref_dispatch_neighbor_pattern"
    assert result.candidate_continuations == ["cont_b"]


def test_continuation_linker_solves_pushctr_c3_jmpx_without_dynamic_marker():
    pushctr = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    jmpx = TVMInstruction(index=1, kind=InstructionKind.CONT_JUMP, opcode="JMPX")

    main_block = TVMBasicBlock(id=1, context_id="cont_entry", instructions=[pushctr, jmpx])
    method_a_block = TVMBasicBlock(id=2, context_id="cont_a", instructions=[])
    method_b_block = TVMBasicBlock(id=3, context_id="cont_b", instructions=[])

    module = TVMModule(
        functions={
            0: TVMFunction(
                method_id=0,
                blocks={1: main_block},
                metadata={"context": "cont_entry"},
            ),
            100: TVMFunction(
                method_id=100,
                blocks={2: method_a_block},
                metadata={"context": "cont_a"},
            ),
            200: TVMFunction(
                method_id=200,
                blocks={3: method_b_block},
                metadata={"context": "cont_b"},
            ),
        },
        cross_function_continuations={
            "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
            "cont_b": ContinuationDescriptor(code_ref="cont_b"),
        },
    )

    linked = ContinuationLinker().link(module)

    assert "__stack__" not in jmpx.continuation_refs
    assert set(jmpx.continuation_refs) == {"cont_a", "cont_b"}
    assert linked.analysis_metadata.get("dynamic_target_count") == 0
    assert linked.analysis_metadata.get("analysis_incomplete") is not True

    solver_stats = linked.analysis_metadata.get("dynamic_target_solver_stats", {})
    assert solver_stats.get("solved_count", 0) >= 1
    results = linked.analysis_metadata.get("dynamic_target_solver_results", [])
    assert any(r.get("instruction_index") == 1 and r.get("status") == "sat" for r in results)


def test_continuation_linker_resolves_execute_with_pushint_hint_on_large_module():
    pushint = TVMInstruction(
        index=0,
        kind=InstructionKind.STACK_PUSH,
        opcode="PUSHINT_LONG",
        immediates=[1200],
    )
    pushctr = TVMInstruction(
        index=1,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    execute = TVMInstruction(index=2, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")

    entry_block = TVMBasicBlock(id=1, context_id="cont_entry", instructions=[pushint, pushctr, execute])
    large_module = _make_large_method_context_module(220)
    large_module.functions[0] = TVMFunction(
        method_id=0,
        blocks={1: entry_block},
        metadata={"context": "cont_entry"},
    )
    large_module.cross_function_continuations["cont_entry"] = ContinuationDescriptor(
        code_ref="cont_entry"
    )

    linked = ContinuationLinker().link(large_module)

    assert "__stack__" not in execute.continuation_refs
    assert execute.continuation_refs == ["cont_200"]
    assert linked.analysis_metadata.get("dynamic_target_count") == 0
    obligations = linked.analysis_metadata.get("dynamic_target_solver_obligations", [])
    assert obligations == []
    results = linked.analysis_metadata.get("dynamic_target_solver_results", [])
    assert any(
        r.get("instruction_index") == 2 and r.get("strategy") == "pushctr_c3_method_id_hint"
        for r in results
    )


def test_continuation_linker_treats_nonlocal_calldict_miss_as_not_dynamic():
    calldict = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_CALL,
        opcode="CALLDICT",
        immediates=[9999],
    )

    entry_block = TVMBasicBlock(id=1, context_id="cont_entry", instructions=[calldict])
    module = TVMModule(
        functions={
            0: TVMFunction(method_id=0, blocks={1: entry_block}, metadata={"context": "cont_entry"}),
            100: TVMFunction(method_id=100, metadata={"context": "cont_a"}),
        },
        cross_function_continuations={
            "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
        },
    )

    linked = ContinuationLinker().link(module)

    assert linked.analysis_metadata.get("dynamic_target_count") == 0
    assert linked.analysis_metadata.get("analysis_incomplete") is not True
    assert calldict.continuation_refs == []
    obligations = linked.analysis_metadata.get("dynamic_target_solver_obligations", [])
    assert obligations == []
    results = linked.analysis_metadata.get("dynamic_target_solver_results", [])
    assert any(
        r.get("instruction_index") == 0
        and r.get("opcode") == "CALLDICT"
        and r.get("status") == "unsat"
        and r.get("reason") == "method_id_not_in_module"
        for r in results
    )


def test_continuation_linker_treats_nonlocal_jmpdict_long_miss_as_not_dynamic():
    jmpdict_long = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_JUMP,
        opcode="JMPDICT_LONG",
        immediates=[9999],
    )

    entry_block = TVMBasicBlock(id=1, context_id="cont_entry", instructions=[jmpdict_long])
    module = TVMModule(
        functions={
            0: TVMFunction(method_id=0, blocks={1: entry_block}, metadata={"context": "cont_entry"}),
            100: TVMFunction(method_id=100, metadata={"context": "cont_a"}),
        },
        cross_function_continuations={
            "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
        },
    )

    linked = ContinuationLinker().link(module)

    assert linked.analysis_metadata.get("dynamic_target_count") == 0
    assert linked.analysis_metadata.get("analysis_incomplete") is not True
    assert jmpdict_long.continuation_refs == []
    obligations = linked.analysis_metadata.get("dynamic_target_solver_obligations", [])
    assert obligations == []
    results = linked.analysis_metadata.get("dynamic_target_solver_results", [])
    assert any(
        r.get("instruction_index") == 0
        and r.get("opcode") == "JMPDICT_LONG"
        and r.get("status") == "unsat"
        and r.get("reason") == "method_id_not_in_module"
        for r in results
    )


def test_dataflow_analyze_tasir_preserves_dynamic_solver_metadata():
    module = TVMModule()
    module.analysis_metadata["dynamic_target_solver_stats"] = {"solve_count": 3, "cache_hits": 1}
    module.analysis_metadata["dynamic_target_solver_results"] = [{"instruction_index": 7, "status": "sat"}]
    module.analysis_metadata["dynamic_target_solver_obligations"] = [{"instruction_index": 9, "reason": "unknown"}]

    graph = DataFlowAnalyzer().analyze_tasir(module, path_sensitive=False)

    assert graph.analysis_metadata.get("dynamic_target_solver_stats", {}).get("solve_count") == 3
    assert graph.analysis_metadata.get("dynamic_target_solver_results", [])[0]["instruction_index"] == 7
    assert graph.analysis_metadata.get("dynamic_target_solver_obligations", [])[0]["instruction_index"] == 9


def test_continuation_linker_treats_truncated_execute_as_dynamic_obligation():
    pushctr = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_LOAD,
        opcode="PUSHCTR",
        inputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    execute = TVMInstruction(index=1, kind=InstructionKind.CONT_CALL, opcode="EXECUTE")

    entry_block = TVMBasicBlock(id=1, context_id="cont_entry", instructions=[pushctr, execute])
    module = _make_large_method_context_module(120)
    module.functions[0] = TVMFunction(
        method_id=0,
        blocks={1: entry_block},
        metadata={"context": "cont_entry"},
    )
    module.cross_function_continuations["cont_entry"] = ContinuationDescriptor(
        code_ref="cont_entry"
    )

    linked = ContinuationLinker().link(module)

    assert "__stack__" in execute.continuation_refs
    assert linked.analysis_metadata.get("dynamic_target_count", 0) >= 1
    obligations = linked.analysis_metadata.get("dynamic_target_solver_obligations", [])
    assert any(
        o.get("instruction_index") == 1
        and o.get("reason") == "candidate_set_truncated"
        for o in obligations
    )


def test_continuation_linker_solves_callref_from_neighbor_cont_create():
    pushref = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_CREATE,
        opcode="PUSHREFCONT",
        continuation_refs=["cont_a"],
    )
    callref = TVMInstruction(index=1, kind=InstructionKind.CONT_CALL, opcode="CALLREF")
    block = TVMBasicBlock(id=1, context_id="cont_entry", instructions=[pushref, callref])
    module = TVMModule(
        functions={
            0: TVMFunction(method_id=0, blocks={1: block}, metadata={"context": "cont_entry"}),
            100: TVMFunction(method_id=100, metadata={"context": "cont_a"}),
        },
        cross_function_continuations={
            "cont_entry": ContinuationDescriptor(code_ref="cont_entry"),
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
        },
    )

    linked = ContinuationLinker().link(module)
    assert "__cellref__" not in callref.continuation_refs
    assert callref.continuation_refs == ["cont_a"]
    assert linked.analysis_metadata.get("dynamic_target_count", 0) == 0
