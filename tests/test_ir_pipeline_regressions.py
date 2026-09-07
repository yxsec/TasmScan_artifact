import json

from tasmscan.analyzer.facts import (
    AnalysisFacts,
    BasicBlock,
    Continuation,
    ControlFlowEdge,
    InstructionFact,
    StackState,
)
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.config import INMSGPARAM_SENDER_INDEX
from tasmscan.ir.dataflow import DataFlowAnalyzer
from tasmscan.ir.dataflow.program_view import ensure_program_view
from tasmscan.ir.dataflow.types import DataFlowValue, ValueSource
from tasmscan.ir.continuation_linker import ContinuationLinker
from tasmscan.ir.ir_builder import IRBuilder
from tasmscan.ir.ir_lifter import IRLifter
from tasmscan.ir.opcode_kind_mapper import get_instruction_kind_from_opcode
from tasmscan.ir.path_sensitive_dataflow import PathSensitiveDataFlowAnalyzer
from tasmscan.ir.tasir_types import (
    BlockEdge,
    ContinuationDescriptor,
    InstructionKind,
    RegisterLocation,
    TVMBasicBlock,
    TVMFunction,
    TVMInstruction,
    TVMModule,
)
from tasmscan.scanner import SecurityScanner


class MockArg:
    def __init__(self, arg_type, value):
        self.type = arg_type
        self.value = value


class MockValueArg:
    def __init__(self, value):
        self.value = value


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


class MockCode:
    def __init__(self, instructions):
        self.instructions = instructions


class MockContinuation:
    def __init__(self, instructions):
        self.instructions = instructions


def test_program_call_graph_uses_method_labels_and_lifter_parses_them():
    facts = ProgramAnalyzer().analyze(
        [MockInstruction("CALLDICT", [MockArg("uint", 5)])],
        cell=None,
    )

    assert len(facts.call_graph_edges) == 1
    assert facts.call_graph_edges[0].caller == "method:0"
    assert facts.call_graph_edges[0].callee == "method:5"

    module = IRBuilder().build_tasir(facts)
    assert (0, 5) in module.call_graph


def test_ir_lifter_maps_cfg_edges_by_instruction_index_not_block_id():
    facts = AnalysisFacts(
        instructions=[
            InstructionFact(instruction=MockInstruction("NOP"), index=10),
            InstructionFact(instruction=MockInstruction("NOP"), index=20),
        ],
        basic_blocks=[
            BasicBlock(id=0, instruction_indices=[10], successors=[1], context="main"),
            BasicBlock(id=1, instruction_indices=[20], successors=[], context="main"),
        ],
        cfg_edges=[ControlFlowEdge(source=10, target=20, kind="jump")],
    )

    module = IRLifter().lift(facts)
    block0 = module.functions[0].blocks[0]
    block1 = module.functions[0].blocks[1]

    assert any(
        edge.target_block == 1 and edge.edge_kind == "jump"
        for edge in block0.successors
    )
    assert any(
        edge.source_block == 0 and edge.edge_kind == "jump"
        for edge in block1.predecessors
    )


def test_continuations_are_preserved_in_facts_metadata_for_ir_lifting():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("IF", [MockArg("code", MockCode([MockInstruction("ACCEPT")]))]),
            MockInstruction("SENDRAWMSG"),
        ],
        cell=None,
    )

    assert facts.metadata.get("continuation_count") == 1
    assert "continuations" in facts.metadata
    assert len(facts.metadata["continuations"]) == 1

    module = IRBuilder().build_tasir(facts)
    assert len(module.cross_function_continuations) == 1


def test_ir_lifter_preserves_callxargs_continuation_refs_from_metadata():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction(
                "PUSHCONT",
                [MockArg("code", MockContinuation([MockInstruction("RET")]))],
            ),
            MockInstruction("CALLXARGS", [MockArg("uint", 0), MockArg("uint", 0)]),
            MockInstruction("NOP"),
        ],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    call_inst = next(inst for inst in module.all_instructions() if inst.opcode == "CALLXARGS")
    assert call_inst.continuation_refs == ["cont_0"]


def test_ir_lifter_preserves_ifelse_inline_continuation_refs_from_metadata():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("PUSHINT", [MockArg("uint", 1)]),
            MockInstruction(
                "IFELSE",
                [
                    MockArg("code", MockContinuation([MockInstruction("ACCEPT")])),
                    MockArg("code", MockContinuation([MockInstruction("SENDRAWMSG")])),
                ],
            ),
            MockInstruction("NOP"),
        ],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    ifelse_inst = next(inst for inst in module.all_instructions() if inst.opcode == "IFELSE")
    assert ifelse_inst.continuation_refs == ["cont_0", "cont_1"]


def test_continuation_linker_marks_unresolved_callxargs_as_dynamic_and_incomplete():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("CALLXARGS", [MockArg("uint", 0), MockArg("uint", 0)]),
            MockInstruction("NOP"),
        ],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    call_inst = next(inst for inst in module.all_instructions() if inst.opcode == "CALLXARGS")

    assert "__stack__" in call_inst.continuation_refs
    assert module.analysis_metadata.get("dynamic_target_count", 0) >= 1
    assert module.analysis_metadata.get("analysis_incomplete") is True
    assert "dynamic_continuation_targets" in module.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )

    graph = DataFlowAnalyzer().analyze_tasir(module, path_sensitive=False)
    assert graph.analysis_metadata.get("analysis_incomplete") is True
    assert graph.analysis_metadata.get("dynamic_target_count", 0) >= 1


def test_booleval_is_cont_call_and_unresolved_target_marks_dynamic_and_incomplete():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("BOOLEVAL"),
            MockInstruction("NOP"),
        ],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    booleval = next(inst for inst in module.all_instructions() if inst.opcode == "BOOLEVAL")

    assert booleval.kind == InstructionKind.CONT_CALL
    assert "__stack__" in booleval.continuation_refs
    assert module.analysis_metadata.get("dynamic_target_count", 0) >= 1
    assert module.analysis_metadata.get("analysis_incomplete") is True
    assert "dynamic_continuation_targets" in module.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )


def test_jmpx_is_modeled_as_cont_jump_not_cont_call():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("JMPX"),
            MockInstruction("NOP"),
        ],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    jmpx = next(inst for inst in module.all_instructions() if inst.opcode == "JMPX")
    assert jmpx.kind == InstructionKind.CONT_JUMP


def test_program_analyzer_exposes_cfg_build_metadata():
    facts = ProgramAnalyzer().analyze(
        [MockInstruction("PUSHINT", [MockArg("uint", 1)]), MockInstruction("NOP")],
        cell=None,
    )
    assert "cfg_build" in facts.metadata
    assert facts.metadata["cfg_build"].get("entry_shape_converged") is True


def test_dataflow_preserves_analysis_incomplete_reasons_from_facts():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
        metadata={
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["cfg_entry_shape_non_converged"],
        },
    )

    graph = DataFlowAnalyzer().analyze(facts, path_sensitive=False)
    assert graph.analysis_metadata.get("analysis_incomplete") is True
    assert "cfg_entry_shape_non_converged" in graph.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )


def test_analyze_tasir_preserves_facts_metadata_without_roundtrip_loss():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
        basic_blocks=[BasicBlock(id=0, instruction_indices=[0], successors=[], context="main")],
        metadata={
            "continuation_extraction_failed": True,
            "continuation_extraction_error": "boom",
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["continuation_extraction_failed"],
        },
    )

    module = IRBuilder().build_tasir(facts)
    module._facts = facts

    graph = DataFlowAnalyzer().analyze_tasir(module, path_sensitive=False)
    assert graph.analysis_metadata.get("continuation_extraction_failed") is True
    assert graph.analysis_metadata.get("continuation_extraction_error") == "boom"
    assert graph.analysis_metadata.get("analysis_incomplete") is True
    assert "continuation_extraction_failed" in graph.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )


def test_analyze_tasir_preserves_cfg_widened_reason_without_private_facts_attachment():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
        basic_blocks=[BasicBlock(id=0, instruction_indices=[0], successors=[], context="main")],
        metadata={
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["cfg_entry_shape_widened"],
        },
    )

    module = IRBuilder().build_tasir(facts)
    assert "cfg_entry_shape_widened" in module.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )

    graph = DataFlowAnalyzer().analyze_tasir(module, path_sensitive=False)
    assert graph.analysis_metadata.get("analysis_incomplete") is True
    assert "cfg_entry_shape_widened" in graph.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )


def test_ir_lifter_maps_inmsg_src_as_message_load_taint_source():
    facts = ProgramAnalyzer().analyze(
        [MockInstruction("INMSG_SRC"), MockInstruction("NOP")],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    inst = next(i for i in module.all_instructions() if i.opcode == "INMSG_SRC")

    assert inst.kind.name == "CELL_LOAD"
    assert inst.is_taint_source is True


def test_ir_lifter_maps_pushnull_as_tuple_null_constant():
    facts = ProgramAnalyzer().analyze(
        [MockInstruction("PUSHNULL"), MockInstruction("NOP")],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    inst = next(i for i in module.all_instructions() if i.opcode == "PUSHNULL")

    assert inst.kind == InstructionKind.TUPLE_CREATE


def test_ir_lifter_maps_pushint_as_stack_push_constant():
    facts = ProgramAnalyzer().analyze(
        [MockInstruction("PUSHINT", [MockArg("uint", 7)]), MockInstruction("NOP")],
        cell=None,
    )

    module = IRBuilder().build_tasir(facts)
    inst = next(i for i in module.all_instructions() if i.opcode == "PUSHINT")
    assert inst.kind == InstructionKind.STACK_PUSH


def test_instruction_table_has_no_unknown_instruction_kind_mappings():
    with open("tasmscan/disassembler/instruction_table.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    opcodes = sorted(
        {
            inst["name"].upper()
            for inst in data.get("instructions", [])
            if isinstance(inst, dict) and "name" in inst
        }
    )

    unknown = [
        opcode
        for opcode in opcodes
        if get_instruction_kind_from_opcode(opcode) == InstructionKind.UNKNOWN
    ]
    assert unknown == []


def test_stack_effect_database_has_no_unknown_instruction_kind_mappings():
    from tasmscan.ir.stack_effects import StackEffectDatabase

    db = StackEffectDatabase()
    unknown = [
        opcode
        for opcode in sorted(db.effects.keys())
        if get_instruction_kind_from_opcode(opcode) == InstructionKind.UNKNOWN
    ]
    assert unknown == []


def test_dataflow_taint_registry_opcodes_have_resolvable_stack_effects():
    from tasmscan.ir.dataflow.taint_registry import (
        CONDITIONAL_TAINT_SOURCES,
        DYNAMIC_CALL_OPCODES,
        MESSAGE_SLICE_LOADERS,
        TAINT_SOURCES,
    )
    from tasmscan.ir.stack_effects import get_stack_effect

    tracked_opcodes = sorted(
        set(DYNAMIC_CALL_OPCODES)
        | set(MESSAGE_SLICE_LOADERS)
        | set(CONDITIONAL_TAINT_SOURCES.keys())
        | set(TAINT_SOURCES.keys())
    )
    missing = [opcode for opcode in tracked_opcodes if get_stack_effect(opcode) is None]
    assert missing == []


def test_analyze_tasir_merges_incomplete_reasons_from_facts_and_module():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
        basic_blocks=[BasicBlock(id=0, instruction_indices=[0], successors=[], context="main")],
        metadata={
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["cfg_entry_shape_non_converged"],
        },
    )

    module = IRBuilder().build_tasir(facts)
    module._facts = facts
    module.analysis_metadata["analysis_incomplete"] = True
    module.analysis_metadata["analysis_incomplete_reasons"] = ["dynamic_continuation_targets"]

    graph = DataFlowAnalyzer().analyze_tasir(module, path_sensitive=False)
    reasons = set(graph.analysis_metadata.get("analysis_incomplete_reasons", []))
    assert "cfg_entry_shape_non_converged" in reasons
    assert "dynamic_continuation_targets" in reasons


def test_dataflow_marks_unknown_successor_as_incomplete():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
        basic_blocks=[
            BasicBlock(
                id=0,
                instruction_indices=[0],
                successors=[],
                context="main",
                has_unknown_successor=True,
            )
        ],
    )

    graph = DataFlowAnalyzer().analyze(facts, path_sensitive=False)
    assert graph.analysis_metadata.get("analysis_incomplete") is True
    assert graph.analysis_metadata.get("unknown_successor_blocks") == [0]
    assert "cfg_unknown_successor" in graph.analysis_metadata.get(
        "analysis_incomplete_reasons", []
    )


def test_dataflow_extracts_constant_from_raw_int_args():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("PUSHINT", [7]), index=0)],
    )

    graph = DataFlowAnalyzer().analyze(facts, path_sensitive=False)
    values = graph.values.get(0, [])
    assert values
    assert values[0].metadata.get("const") == 7


def test_dataflow_processes_facts_in_instruction_index_order():
    facts = AnalysisFacts(
        instructions=[
            InstructionFact(instruction=MockInstruction("SENDRAWMSG"), index=2),
            InstructionFact(
                instruction=MockInstruction("INMSGPARAM", [MockArg("uint", INMSGPARAM_SENDER_INDEX)]),
                index=0,
            ),
            InstructionFact(instruction=MockInstruction("NOP"), index=1),
        ],
    )

    graph = DataFlowAnalyzer().analyze(facts, path_sensitive=False)
    assert (0, 2) in graph.tainted_propagation


def test_sensitive_taint_pairs_use_global_dedup():
    analyzer = DataFlowAnalyzer()
    analyzer.current_state.stack = [
        DataFlowValue(
            source=ValueSource.MESSAGE_BODY,
            definition_site=7,
            tainted=True,
            metadata={"taint_origins": [7]},
        )
    ]

    analyzer._handle_sensitive_operation("SENDRAWMSG", 20)
    analyzer._handle_sensitive_operation("SENDRAWMSG", 20)

    assert analyzer.graph.tainted_propagation.count((7, 20)) == 1


def test_guard_semantics_align_between_event_map_and_interprocedural_summary():
    facts = ProgramAnalyzer().analyze(
        [MockInstruction("IF"), MockInstruction("SENDRAWMSG")],
        cell=None,
    )

    assert len(facts.events_of("guard")) == 1

    module = IRBuilder().build_tasir(facts)
    view = module.build_inter_procedural_view()

    assert view.function_summaries[0]["has_guard"] is True
    if_labels = [
        label.value
        for inst in module.all_instructions()
        if inst.opcode == "IF"
        for label in inst.semantic_labels
    ]
    assert "guard" in if_labels


def test_scanner_surfaces_detector_failures_in_findings(monkeypatch):
    class FailingDetector:
        name = "failing_detector"

        def detect(self, _facts):
            raise RuntimeError("boom")

    class FakeAnalyzer:
        def analyze(self, _instructions, _cell):
            return AnalysisFacts(
                instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
            )

    scanner = SecurityScanner(enable_all_detectors=False, use_tasir=False)
    scanner.detectors = [FailingDetector()]
    scanner.analyzer = FakeAnalyzer()

    monkeypatch.setattr(
        "tasmscan.scanner.decompile_cell",
        lambda _cell: [MockInstruction("NOP")],
    )

    findings = scanner.scan_cell(object())
    failure_findings = [f for f in findings if f.detector == "analysis_incomplete"]
    assert len(failure_findings) == 1
    failed = failure_findings[0].extra["failed_detectors"]
    assert failed[0]["name"] == "failing_detector"


def test_ir_lifter_continuation_context_matching_uses_parent_chain_not_substring():
    facts = AnalysisFacts(
        instructions=[
            InstructionFact(instruction=MockInstruction("NOP"), index=0, continuation_id="cont_1"),
            InstructionFact(instruction=MockInstruction("NOP"), index=1, continuation_id="cont_10"),
            InstructionFact(instruction=MockInstruction("NOP"), index=2, continuation_id="cont_nested"),
        ],
        basic_blocks=[
            BasicBlock(id=0, instruction_indices=[0], successors=[], context="cont_1"),
            BasicBlock(id=1, instruction_indices=[1], successors=[], context="cont_10"),
            BasicBlock(id=2, instruction_indices=[2], successors=[], context="cont_nested"),
        ],
        entry_points=[
            {"kind": "method", "context": "cont_1", "block_id": 0, "method_id": 1},
            {"kind": "method", "context": "cont_10", "block_id": 1, "method_id": 10},
        ],
        metadata={
            "continuations": {
                "cont_1": Continuation(
                    id="cont_1",
                    instructions=[],
                    entry_index=0,
                    parent_context="main",
                    parent_local_index=0,
                    kind="method",
                    method_id=1,
                ),
                "cont_10": Continuation(
                    id="cont_10",
                    instructions=[],
                    entry_index=0,
                    parent_context="main",
                    parent_local_index=1,
                    kind="method",
                    method_id=10,
                ),
                "cont_nested": Continuation(
                    id="cont_nested",
                    instructions=[],
                    entry_index=0,
                    parent_context="cont_10",
                    parent_local_index=0,
                    kind="continuation",
                ),
            }
        },
    )

    module = IRLifter().lift(facts)

    assert set(module.functions[1].continuations.keys()) == {"cont_1"}
    assert set(module.functions[10].continuations.keys()) == {"cont_10", "cont_nested"}


def test_program_analyzer_callsite_collection_tolerates_args_without_type():
    facts = ProgramAnalyzer().analyze(
        [
            MockInstruction("CALLDICT", [MockValueArg(42)]),
            MockInstruction("RET"),
        ],
        cell=None,
    )

    assert len(facts.call_sites) == 1
    assert facts.call_sites[0].target_type == "method"
    assert facts.call_sites[0].target == 42


def test_ir_lifter_derives_block_edge_continuation_ref_from_target_context():
    facts = AnalysisFacts(
        instructions=[
            InstructionFact(instruction=MockInstruction("CALLXARGS"), index=0),
            InstructionFact(instruction=MockInstruction("NOP"), index=1),
            InstructionFact(instruction=MockInstruction("RET"), index=10, continuation_id="cont_0"),
        ],
        basic_blocks=[
            BasicBlock(id=0, instruction_indices=[0], successors=[1, 2], context="main"),
            BasicBlock(id=1, instruction_indices=[1], successors=[], context="main"),
            BasicBlock(id=2, instruction_indices=[10], successors=[1], context="cont_0"),
        ],
        cfg_edges=[
            ControlFlowEdge(source=0, target=10, kind="call_cont"),
            ControlFlowEdge(source=10, target=1, kind="return_cont", metadata={"cont_id": "cont_0"}),
        ],
        metadata={"continuations": {"cont_0": {"id": "cont_0"}}},
    )

    module = IRLifter().lift(facts)
    main_block = module.functions[0].blocks[0]
    call_edge = next(edge for edge in main_block.successors if edge.edge_kind == "call_cont")
    assert call_edge.continuation_ref == "cont_0"

    cont_block = next(block for block in module.all_blocks() if block.context_id == "cont_0")
    ret_edge = next(edge for edge in cont_block.successors if edge.edge_kind == "return_cont")
    assert ret_edge.continuation_ref == "cont_0"


def test_ir_lifter_block_metadata_preserves_stack_range_profile():
    facts = AnalysisFacts(
        instructions=[InstructionFact(instruction=MockInstruction("NOP"), index=0)],
        basic_blocks=[BasicBlock(id=0, instruction_indices=[0], successors=[], context="main")],
        stack_states=[
            StackState(
                instruction_index=0,
                height_before=5,
                height_after=None,
                delta=None,
                unknown=True,
                height_min=4,
                height_max=8,
            )
        ],
    )

    module = IRLifter().lift(facts)
    block = module.functions[0].blocks[0]
    assert block.entry_stack_height == 5
    assert block.exit_stack_height is None
    profile = block.metadata.get("stack_profile", {})
    assert profile.get("entry_min") == 4
    assert profile.get("entry_max") == 8
    assert profile.get("entry_unknown") is True
    assert profile.get("exit_min") == 4
    assert profile.get("exit_max") == 8
    assert profile.get("exit_unknown") is True


def test_program_view_preserves_successor_edge_kinds_for_facts_and_module():
    facts = AnalysisFacts(
        instructions=[
            InstructionFact(instruction=MockInstruction("NOP"), index=0),
            InstructionFact(instruction=MockInstruction("NOP"), index=1),
        ],
        basic_blocks=[
            BasicBlock(id=0, instruction_indices=[0], successors=[1], context="main"),
            BasicBlock(id=1, instruction_indices=[1], successors=[], context="main"),
        ],
        cfg_edges=[
            ControlFlowEdge(source=0, target=1, kind="guard_throw"),
            ControlFlowEdge(source=0, target=1, kind="jump"),
        ],
    )
    facts_view = ensure_program_view(facts)
    facts_block = next(block for block in facts_view.basic_blocks if block.id == 0)
    assert facts_block.successor_edge_kinds[1] == ["guard_throw", "jump"]

    inst0 = TVMInstruction(index=0, kind=InstructionKind.NOP, opcode="NOP")
    inst1 = TVMInstruction(index=1, kind=InstructionKind.NOP, opcode="NOP")
    block0 = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[inst0],
        successors=[
            BlockEdge(source_block=0, target_block=1, edge_kind="guard_throw"),
            BlockEdge(source_block=0, target_block=1, edge_kind="fallthrough"),
        ],
    )
    block1 = TVMBasicBlock(id=1, context_id="main", instructions=[inst1], successors=[])
    module = TVMModule(functions={0: TVMFunction(method_id=0, blocks={0: block0, 1: block1})})
    module_view = ensure_program_view(module)
    module_block = next(block for block in module_view.basic_blocks if block.id == 0)
    assert module_block.successors == [1]
    assert module_block.successor_edge_kinds[1] == ["fallthrough", "guard_throw"]


def test_continuation_linker_uses_cfg_fixpoint_for_back_edge_register_state():
    save_inst = TVMInstruction(
        index=10,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    nop_inst = TVMInstruction(index=11, kind=InstructionKind.NOP, opcode="NOP")
    popctr_inst = TVMInstruction(
        index=20,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    block_a = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[save_inst, nop_inst],
        successors=[BlockEdge(source_block=0, target_block=1, edge_kind="jump")],
    )
    block_b = TVMBasicBlock(
        id=1,
        context_id="main",
        instructions=[popctr_inst],
        successors=[BlockEdge(source_block=1, target_block=0, edge_kind="jump")],
    )
    module = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block_a, 1: block_b})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )

    linked = ContinuationLinker().link(module)
    saved_value = linked.cross_function_continuations["cont_target"].savelist.registers.get(0)
    assert saved_value is not None
    assert saved_value.definition_site == 20


def test_continuation_linker_propagates_ambiguous_save_targets_conservatively():
    popctr_inst = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    ambiguous_save = TVMInstruction(
        index=1,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_a", "cont_b"],
    )
    block = TVMBasicBlock(id=0, context_id="main", instructions=[popctr_inst, ambiguous_save])
    module = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_a": ContinuationDescriptor(code_ref="cont_a"),
            "cont_b": ContinuationDescriptor(code_ref="cont_b"),
        },
    )

    linked = ContinuationLinker().link(module)
    saved_a = linked.cross_function_continuations["cont_a"].savelist.registers.get(0)
    saved_b = linked.cross_function_continuations["cont_b"].savelist.registers.get(0)
    assert saved_a is not None
    assert saved_b is not None
    resolution = linked.analysis_metadata.get("cont_ref_resolution", {})
    assert resolution.get("ambiguous_overapprox_hits", 0) >= 1


def test_continuation_linker_marks_unresolved_save_targets_incomplete():
    save_inst = TVMInstruction(
        index=1,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=[],
    )
    block = TVMBasicBlock(id=0, context_id="main", instructions=[save_inst])
    module = TVMModule(functions={0: TVMFunction(method_id=0, blocks={0: block})})

    linked = ContinuationLinker().link(module)
    reasons = linked.analysis_metadata.get("analysis_incomplete_reasons", [])
    assert "savelist_target_unresolved" in reasons
    resolution = linked.analysis_metadata.get("cont_ref_resolution", {})
    assert resolution.get("unresolved", 0) >= 1


def test_continuation_linker_drop_conflict_strategy_skips_conflicting_savelist_entry():
    popctr_a = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    popctr_b = TVMInstruction(
        index=10,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    save_inst = TVMInstruction(
        index=20,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    block_a = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[popctr_a],
        successors=[BlockEdge(source_block=0, target_block=2, edge_kind="jump")],
    )
    block_b = TVMBasicBlock(
        id=1,
        context_id="main",
        instructions=[popctr_b],
        successors=[BlockEdge(source_block=1, target_block=2, edge_kind="jump")],
    )
    block_a.successors.append(BlockEdge(source_block=0, target_block=1, edge_kind="fallthrough"))
    block_join = TVMBasicBlock(id=2, context_id="main", instructions=[save_inst], successors=[])

    module_merge = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block_a, 1: block_b, 2: block_join})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )
    linked_merge = ContinuationLinker(register_merge_strategy="merge").link(module_merge)
    merged_saved = linked_merge.cross_function_continuations["cont_target"].savelist.registers.get(0)
    assert merged_saved is not None
    assert merged_saved.metadata.get("candidate_definition_sites") == [0, 10]
    assert linked_merge.analysis_metadata.get("register_merge_conflicts", 0) >= 1

    # Rebuild module (link() mutates in place)
    module_drop = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block_a, 1: block_b, 2: block_join})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )
    linked_drop = ContinuationLinker(register_merge_strategy="drop_conflict").link(module_drop)
    assert linked_drop.cross_function_continuations["cont_target"].savelist.registers == {}
    assert linked_drop.analysis_metadata.get("register_merge_conflicts", 0) >= 1


def test_continuation_linker_ignores_dead_save_points():
    pop_live = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    ret_live = TVMInstruction(index=1, kind=InstructionKind.CONT_RETURN, opcode="RET")
    live_block = TVMBasicBlock(id=0, context_id="main", instructions=[pop_live, ret_live], successors=[])

    pop_dead = TVMInstruction(
        index=10,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    save_dead = TVMInstruction(
        index=11,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    dead_block = TVMBasicBlock(id=1, context_id="main", instructions=[pop_dead, save_dead], successors=[])

    module = TVMModule(
        functions={0: TVMFunction(method_id=0, entry_block_id=0, blocks={0: live_block, 1: dead_block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )

    linked = ContinuationLinker().link(module)
    assert linked.analysis_metadata.get("save_point_count") == 0
    assert linked.cross_function_continuations["cont_target"].savelist.registers == {}


def test_continuation_linker_savelist_candidate_cap_widens_and_reports_metadata():
    popctr_a = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    save_a = TVMInstruction(
        index=1,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    popctr_b = TVMInstruction(
        index=2,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    save_b = TVMInstruction(
        index=3,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    block = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[popctr_a, save_a, popctr_b, save_b],
        successors=[],
    )
    module = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )

    linked = ContinuationLinker(max_savelist_candidates_per_register=1).link(module)
    saved_values = linked.cross_function_continuations["cont_target"].savelist.restore_all(0)
    assert len(saved_values) == 1
    assert saved_values[0].metadata.get("savelist_widened") is True
    assert linked.analysis_metadata.get("savelist_widened_registers", 0) >= 1


def test_continuation_linker_allows_unbounded_savelist_candidates_when_cap_zero():
    popctr_a = TVMInstruction(
        index=0,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    save_a = TVMInstruction(
        index=1,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    popctr_b = TVMInstruction(
        index=2,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=0)],
        immediates=[0],
    )
    save_b = TVMInstruction(
        index=3,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
        continuation_refs=["cont_target"],
    )
    block = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[popctr_a, save_a, popctr_b, save_b],
        successors=[],
    )
    module = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )

    linked = ContinuationLinker(max_savelist_candidates_per_register=0).link(module)
    saved_values = linked.cross_function_continuations["cont_target"].savelist.restore_all(0)
    assert len(saved_values) == 2
    assert all(v.metadata.get("savelist_widened") is not True for v in saved_values)
    assert linked.analysis_metadata.get("savelist_widened_registers", 0) == 0


def test_continuation_linker_resolves_save_target_via_cfg_stack_def_use():
    pushcont = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_CREATE,
        opcode="PUSHCONT",
        continuation_refs=["cont_target"],
    )
    nop_inst = TVMInstruction(index=1, kind=InstructionKind.NOP, opcode="NOP")
    save_inst = TVMInstruction(
        index=2,
        kind=InstructionKind.CONT_SAVE,
        opcode="SAVE",
    )
    block = TVMBasicBlock(id=0, context_id="main", instructions=[pushcont, nop_inst, save_inst])

    module_default = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )
    linked_default = ContinuationLinker(cont_ref_search_window=2).link(module_default)
    assert 0 in linked_default.cross_function_continuations["cont_target"].savelist.registers

    module_narrow = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )
    linked_narrow = ContinuationLinker(cont_ref_search_window=1).link(module_narrow)
    assert 0 in linked_narrow.cross_function_continuations["cont_target"].savelist.registers
    resolution = linked_narrow.analysis_metadata.get("cont_ref_resolution", {})
    assert resolution.get("stack_def_use_hits", 0) >= 1


def test_continuation_linker_cont_ref_search_window_still_used_as_fallback():
    pushcont = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_CREATE,
        opcode="PUSHCONT",
        continuation_refs=["cont_target"],
    )
    # Long unknown span clears stack-def/use precision; backward CFG search is
    # depth-capped, so wide window remains a necessary fallback.
    unknown_insts = [
        TVMInstruction(index=i, kind=InstructionKind.UNKNOWN, opcode=f"UNKNOWN_OP_{i}")
        for i in range(1, 20)
    ]
    save_inst = TVMInstruction(index=20, kind=InstructionKind.CONT_SAVE, opcode="SAVE")
    block = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[pushcont, *unknown_insts, save_inst],
    )

    module_wide = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )
    linked_wide = ContinuationLinker(cont_ref_search_window=20).link(module_wide)
    assert 0 in linked_wide.cross_function_continuations["cont_target"].savelist.registers
    assert linked_wide.analysis_metadata.get("cont_ref_resolution", {}).get("window_hits", 0) >= 1

    module_narrow = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )
    linked_narrow = ContinuationLinker(cont_ref_search_window=1).link(module_narrow)
    assert linked_narrow.cross_function_continuations["cont_target"].savelist.registers == {}


def test_continuation_linker_cfg_backward_search_recovers_long_distance_save_target():
    pushcont = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_CREATE,
        opcode="PUSHCONT",
        continuation_refs=["cont_target"],
    )
    unknown_a = TVMInstruction(index=1, kind=InstructionKind.UNKNOWN, opcode="UNKNOWN_A")
    unknown_b = TVMInstruction(index=2, kind=InstructionKind.UNKNOWN, opcode="UNKNOWN_B")
    save_inst = TVMInstruction(index=3, kind=InstructionKind.CONT_SAVE, opcode="SAVE")
    block = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[pushcont, unknown_a, unknown_b, save_inst],
        successors=[],
    )
    module = TVMModule(
        functions={0: TVMFunction(method_id=0, blocks={0: block})},
        cross_function_continuations={
            "cont_target": ContinuationDescriptor(code_ref="cont_target")
        },
    )

    linked = ContinuationLinker(cont_ref_search_window=1).link(module)
    assert 0 in linked.cross_function_continuations["cont_target"].savelist.registers
    resolution = linked.analysis_metadata.get("cont_ref_resolution", {})
    assert resolution.get("cfg_backward_hits", 0) >= 1


def test_path_sensitive_metadata_uses_successor_edge_kinds():
    inst0 = TVMInstruction(index=0, kind=InstructionKind.NOP, opcode="NOP")
    inst1 = TVMInstruction(index=1, kind=InstructionKind.NOP, opcode="NOP")
    block0 = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[inst0],
        successors=[
            BlockEdge(source_block=0, target_block=1, edge_kind="guard_throw"),
            BlockEdge(source_block=0, target_block=1, edge_kind="fallthrough"),
        ],
    )
    block1 = TVMBasicBlock(id=1, context_id="main", instructions=[inst1], successors=[])
    module = TVMModule(functions={0: TVMFunction(method_id=0, blocks={0: block0, 1: block1})})

    graph = DataFlowAnalyzer().analyze_tasir(module, path_sensitive=True)
    metadata = graph.analysis_metadata
    assert metadata.get("semantic_branch_split_count", 0) >= 1
    hist = metadata.get("successor_edge_kind_histogram", {})
    assert hist.get("guard_throw", 0) >= 1
    assert metadata.get("exception_edge_count", 0) >= 1


def test_path_sensitive_filters_procedural_edges_from_loop_fixpoint():
    analyzer = PathSensitiveDataFlowAnalyzer()
    assert analyzer._edge_kind_allows_loop_fixpoint(["call_cont"]) is False
    assert analyzer._edge_kind_allows_loop_fixpoint(["return_cont"]) is False
    assert analyzer._edge_kind_allows_loop_fixpoint(["jump"]) is True
