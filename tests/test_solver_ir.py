from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.ir.ir_builder import IRBuilder
from tasmscan.ir.solver_ir import SMTLibExporter, SolverIRLowering
from tasmscan.ir.tasir_types import (
    BlockEdge,
    InstructionKind,
    RegisterLocation,
    TVMBasicBlock,
    TVMFunction,
    TVMInstruction,
    TVMModule,
)


class MockArg:
    def __init__(self, arg_type, value):
        self.type = arg_type
        self.value = value


class MockInstruction:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []


def _build_solver_module(instructions):
    facts = ProgramAnalyzer().analyze(instructions, cell=None)
    builder = IRBuilder()
    return builder.build_solver_ir(facts)


def test_solver_ir_lowers_pushctr_and_dict_dispatch():
    solver_module = _build_solver_module(
        [
            MockInstruction("PUSHCTR", [MockArg("uint", 3)]),
            MockInstruction("CALLDICT", [MockArg("uint", 0)]),
            MockInstruction("JMPDICT", [MockArg("uint", 0)]),
        ]
    )

    calldict = next(inst for inst in solver_module.instructions.values() if inst.opcode == "CALLDICT")
    jmpdict = next(inst for inst in solver_module.instructions.values() if inst.opcode == "JMPDICT")

    assert calldict.metadata.get("dispatch_kind") == "dict_lookup"
    assert calldict.metadata.get("method_id") == 0
    assert calldict.metadata.get("candidate_method_ids") == [0]
    assert calldict.reads
    assert jmpdict.metadata.get("method_id") == 0
    assert any("dict_lookup" in c.expression and " 0" in c.expression for c in solver_module.constraints)
    assert len(solver_module.obligations) == 0


def test_solver_ir_records_obligation_when_dict_method_id_missing():
    solver_module = _build_solver_module(
        [
            MockInstruction("CALLDICT"),
            MockInstruction("NOP"),
        ]
    )

    obligations = [o for o in solver_module.obligations if o.opcode == "CALLDICT"]
    assert obligations
    assert obligations[0].reason == "missing_method_id_immediate"
    assert obligations[0].method_id is None


def test_solver_ir_records_obligation_when_dict_method_unknown():
    solver_module = _build_solver_module(
        [
            MockInstruction("CALLDICT", [MockArg("uint", 999)]),
            MockInstruction("NOP"),
        ]
    )

    obligations = [o for o in solver_module.obligations if o.opcode == "CALLDICT"]
    assert obligations
    assert obligations[0].reason == "method_id_not_in_module"
    assert obligations[0].method_id == 999
    calldict = next(inst for inst in solver_module.instructions.values() if inst.opcode == "CALLDICT")
    assert calldict.metadata.get("candidate_method_ids") == []


def test_solver_ir_smt_export_contains_symbols_constraints_and_check_sat():
    solver_module = _build_solver_module(
        [
            MockInstruction("PUSHCTR", [MockArg("uint", 3)]),
            MockInstruction("CALLDICT", [MockArg("uint", 0)]),
        ]
    )
    smt = SMTLibExporter().export(solver_module)

    assert "(declare-fun" in smt
    assert "(assert" in smt
    assert "dict_lookup" in smt
    assert "(check-sat)" in smt


def test_solver_ir_binds_fallback_exit_symbol_to_actual_predecessor_exit():
    # block 0 has predecessor block 1 (higher ID). During sorted lowering, block 0 is
    # visited first and therefore needs a deferred fallback symbol for block 1 exit.
    call_inst = TVMInstruction(
        index=0,
        kind=InstructionKind.CONT_CALL,
        opcode="CALLDICT",
        immediates=[0],
    )
    block0 = TVMBasicBlock(
        id=0,
        context_id="main",
        instructions=[call_inst],
        predecessors=[BlockEdge(source_block=1, target_block=0, edge_kind="jump")],
    )

    popctr_inst = TVMInstruction(
        index=1,
        kind=InstructionKind.REGISTER_STORE,
        opcode="POPCTR",
        outputs=[RegisterLocation(index=3)],
        immediates=[3],
    )
    block1 = TVMBasicBlock(
        id=1,
        context_id="main",
        instructions=[popctr_inst],
        successors=[BlockEdge(source_block=1, target_block=0, edge_kind="jump")],
    )

    module = TVMModule(functions={0: TVMFunction(method_id=0, blocks={0: block0, 1: block1})})
    solver_module = SolverIRLowering().lower(module)

    late_bind_constraints = [
        c for c in solver_module.constraints
        if c.metadata.get("kind") == "late_block_exit_bind" and c.metadata.get("block_id") == 1
    ]
    assert late_bind_constraints
    late_bind = late_bind_constraints[0]

    fallback_symbol = late_bind.metadata.get("fallback_symbol")
    actual_exit_symbol = solver_module.blocks[1].exit_state.ctrl_regs.get(3)
    assert fallback_symbol
    assert actual_exit_symbol
    assert late_bind.expression == f"(= {fallback_symbol} {actual_exit_symbol})"
