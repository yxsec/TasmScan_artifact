from typing import Dict, List

from tasmscan.analyzer.constants import MAIN_CONTEXT
from tasmscan.analyzer.facts import Continuation, InstructionFact
from tasmscan.analyzer.program_analyzer import ProgramAnalyzer


class DummyInstruction:
    def __init__(self, name: str, args=None):
        self.name = name
        self.args = args or []


class DummyArg:
    def __init__(self, value: int):
        self.value = value


class DummyCode:
    def __init__(self, instructions):
        self.instructions = instructions


class DummyCodeValueArg:
    def __init__(self, instructions):
        self.value = DummyCode(instructions)
        self.type = "code"


def make_fact(idx: int, opcode: str, continuation_id=None, parent_index=None, args=None) -> InstructionFact:
    return InstructionFact(
        instruction=DummyInstruction(opcode, args=args or []),
        index=idx,
        offset=0,
        length=0,
        hash="test",
        continuation_id=continuation_id,
        parent_index=parent_index,
    )


def test_extract_continuations_keeps_empty_pushcont_code_args():
    analyzer = ProgramAnalyzer()
    instructions = [
        DummyInstruction("PUSHCONT_SHORT", args=[DummyCodeValueArg([])]),
    ]

    push_map, inline_map, continuations = analyzer._continuation_resolver.extract_continuations(instructions)

    assert (MAIN_CONTEXT, 0) in push_map
    cont_ids = push_map[(MAIN_CONTEXT, 0)]
    assert len(cont_ids) == 1
    cont_id = cont_ids[0]
    assert cont_id in continuations
    assert continuations[cont_id].instructions == []
    assert inline_map == {}


def test_map_continuations_records_return_target():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT"),
        make_fact(2, "IF"),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 in returning_branches
    assert cont_targets["cont_0"] == [3]
    assert not uncertain


def test_map_continuations_tail_jump_no_return():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "JMPX"),
        make_fact(2, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[1] == ["cont_0"]
    assert not returning_branches
    assert "cont_0" not in cont_targets
    assert not uncertain


def test_map_continuations_ifret_no_return_target():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "IFRET"),
        make_fact(2, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 1 not in branch_map
    assert 1 not in returning_branches
    assert "cont_0" not in cont_targets
    assert not uncertain


def test_map_continuations_with_dup_preserves_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "DUP"),
        make_fact(2, "PUSHINT"),
        make_fact(3, "IF"),
        make_fact(4, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[3] == ["cont_0"]
    assert 3 in returning_branches
    assert cont_targets["cont_0"] == [4]
    assert not uncertain


def test_map_continuations_rot_rotates_continuations():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHCONT"),
        make_fact(3, "ROT"),
        make_fact(4, "PUSHINT"),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_0"], 1: ["cont_1"], 2: ["cont_2"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # ROT: a b c -> b c a (top becomes original bottom)
    assert branch_map[5] == ["cont_1"]
    assert 5 in returning_branches
    assert cont_targets["cont_1"] == [6]
    assert not uncertain


def test_map_continuations_rotrev_rotates_continuations():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHCONT"),
        make_fact(3, "ROTREV"),
        make_fact(4, "PUSHINT"),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_0"], 1: ["cont_1"], 2: ["cont_2"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # ROTREV: a b c -> c a b (top becomes original middle)
    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert cont_targets["cont_0"] == [6]
    assert not uncertain


def test_map_continuations_xchgx_with_constant_index_preserves_continuation_resolution():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(1)]),  # condition value
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHINT", args=[DummyArg(1)]),  # XCHGX index
        make_fact(3, "XCHGX"),
        make_fact(4, "IF"),
        make_fact(5, "NOP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_0"]
    assert 4 in returning_branches
    assert cont_targets["cont_0"] == [5]
    assert not uncertain


def test_map_continuations_roll_with_constant_index_preserves_continuation_resolution():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(1)]),  # condition value
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHINT", args=[DummyArg(1)]),  # ROLL index
        make_fact(3, "ROLL"),
        make_fact(4, "IF"),
        make_fact(5, "NOP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_0"]
    assert 4 in returning_branches
    assert cont_targets["cont_0"] == [5]
    assert not uncertain


def test_map_continuations_dropx_with_constant_count_preserves_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(1)]),  # dropped by DROPX
        make_fact(2, "PUSHINT", args=[DummyArg(1)]),  # DROPX count
        make_fact(3, "DROPX"),
        make_fact(4, "PUSHINT", args=[DummyArg(1)]),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert cont_targets["cont_0"] == [6]
    assert not uncertain


def test_map_continuations_tuplevar_with_known_arity_keeps_stack_trackable():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(0)]),  # n=0
        make_fact(2, "TUPLEVAR"),
        make_fact(3, "DROP"),  # remove tuple value
        make_fact(4, "PUSHINT", args=[DummyArg(1)]),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert cont_targets["cont_0"] == [6]
    assert not uncertain


def test_map_continuations_untuplevar_with_known_arity_keeps_stack_trackable():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(99)]),  # tuple placeholder
        make_fact(2, "PUSHINT", args=[DummyArg(1)]),   # n=1
        make_fact(3, "UNTUPLEVAR"),
        make_fact(4, "DROP"),  # remove unpacked value
        make_fact(5, "PUSHINT", args=[DummyArg(1)]),
        make_fact(6, "IF"),
        make_fact(7, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[6] == ["cont_0"]
    assert 6 in returning_branches
    assert cont_targets["cont_0"] == [7]
    assert not uncertain


def test_map_continuations_push3_keeps_branch_trackable_after_cleanup():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(1)]),
        make_fact(2, "PUSH3", args=[DummyArg(1), DummyArg(1), DummyArg(1)]),
        make_fact(3, "DROP"),
        make_fact(4, "DROP"),
        make_fact(5, "DROP"),
        make_fact(6, "IF"),
        make_fact(7, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[6] == ["cont_0"]
    assert 6 in returning_branches
    assert cont_targets["cont_0"] == [7]
    assert not uncertain


def test_map_continuations_istuple_normalizes_condition_slot():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "DUP"),
        make_fact(2, "ISTUPLE"),
        make_fact(3, "IF"),
        make_fact(4, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[3] == ["cont_0"]
    assert 3 in returning_branches
    assert cont_targets["cont_0"] == [4]
    assert not uncertain


def test_map_continuations_pop_long_zero_index_keeps_known_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(123)]),
        make_fact(2, "POP_LONG", args=[DummyArg(0)]),
        make_fact(3, "PUSHINT", args=[DummyArg(1)]),
        make_fact(4, "IF"),
        make_fact(5, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_0"]
    assert 4 in returning_branches
    assert cont_targets["cont_0"] == [5]
    assert not uncertain


def test_map_continuations_literal_push_opcode_keeps_condition_layout():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHNULL"),
        make_fact(2, "IF"),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 in returning_branches
    assert cont_targets["cont_0"] == [3]
    assert not uncertain


def test_map_continuations_swap2_reorders_pairs_and_preserves_resolution():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(1)]),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHINT", args=[DummyArg(11)]),
        make_fact(3, "PUSHINT", args=[DummyArg(22)]),
        make_fact(4, "SWAP2"),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert cont_targets["cont_0"] == [6]
    assert not uncertain


def test_map_continuations_pick_with_constant_index_preserves_resolution():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(1)]),  # condition
        make_fact(2, "PUSHINT", args=[DummyArg(1)]),  # PICK index
        make_fact(3, "PICK"),
        make_fact(4, "IF"),
        make_fact(5, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_0"]
    assert 4 in returning_branches
    assert cont_targets["cont_0"] == [5]
    assert not uncertain


def test_map_continuations_onlytopx_with_constant_count_preserves_resolution():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(99)]),  # dropped by ONLYTOPX
        make_fact(1, "PUSHINT", args=[DummyArg(1)]),   # condition
        make_fact(2, "PUSHCONT"),
        make_fact(3, "PUSHINT", args=[DummyArg(2)]),   # keep top 2
        make_fact(4, "ONLYTOPX"),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {2: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert cont_targets["cont_0"] == [6]
    assert not uncertain


def test_map_continuations_onlyx_with_constant_count_preserves_resolution():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT", args=[DummyArg(1)]),   # condition
        make_fact(2, "PUSHINT", args=[DummyArg(99)]),  # dropped by ONLYX
        make_fact(3, "PUSHINT", args=[DummyArg(2)]),   # keep bottom 2
        make_fact(4, "ONLYX"),
        make_fact(5, "IF"),
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert cont_targets["cont_0"] == [6]
    assert not uncertain


def test_map_continuations_nip_removes_second_element():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT"),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "NIP"),
        make_fact(3, "PUSHINT"),
        make_fact(4, "IF"),
        make_fact(5, "NOP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_0"]
    assert 4 in returning_branches
    assert cont_targets["cont_0"] == [5]
    assert not uncertain


def test_map_continuations_stack_uncertain_marks_branch():
    """ROLL with non-integer index still marks continuation resolution uncertain."""
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "ROLL"),  # Index is not an integer value token here
        make_fact(2, "PUSHINT"),
        make_fact(3, "IF"),    # Branch cannot resolve continuation due to stack uncertainty
        make_fact(4, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # Branch at index 2 should be marked as uncertain (continuation not resolved)
    assert 3 not in branch_map  # No continuation resolved
    assert 3 in uncertain  # Branch marked as uncertain
    assert not returning_branches
    assert "cont_0" not in cont_targets


def test_map_continuations_returning_branch_preserves_stack():
    """Returning branches preserve remaining stack, allowing later resolution.

    TVM semantics: a returning IF pops condition + 1 continuation, executes it,
    then returns. Elements below the consumed items remain on the stack.
    After IF consumes cont_1, cont_0 is still available for the second IF.
    """
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHINT"),
        make_fact(3, "IF"),   # Returning conditional - consumes cont_1
        make_fact(4, "PUSHINT"),
        make_fact(5, "IF"),   # Stack still has cont_0 - should resolve
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_0"], 1: ["cont_1"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # First IF resolves cont_1 normally
    assert branch_map[3] == ["cont_1"]
    assert 3 in returning_branches
    assert cont_targets["cont_1"] == [4]

    # Second IF now correctly resolves cont_0 (preserved after first IF returned)
    assert branch_map[5] == ["cont_0"]
    assert 5 in returning_branches
    assert 5 not in uncertain


def test_map_continuations_callx_records_return_target():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "EXECUTE"),
        make_fact(2, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[1] == ["cont_0"]
    assert 1 in returning_branches
    assert cont_targets["cont_0"] == [2]
    assert not uncertain


def test_map_continuations_callxargs_with_stack_params_resolves_tos_cont():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(42)]),  # arg0
        make_fact(1, "PUSHCONT"),                      # cont
        make_fact(2, "CALLXARGS", args=[DummyArg(1), DummyArg(0)]),
        make_fact(3, "NOP"),
    ]
    mapping = {1: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_call"]
    assert 2 in returning_branches
    assert cont_targets["cont_call"] == [3]
    assert 2 not in uncertain


def test_map_continuations_callccargs_with_stack_params_resolves_tos_cont():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(42)]),  # arg0
        make_fact(1, "PUSHCONT"),                      # cont
        make_fact(2, "CALLCCARGS", args=[DummyArg(1), DummyArg(0)]),
        make_fact(3, "NOP"),
    ]
    mapping = {1: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_call"]
    assert 2 in returning_branches
    assert cont_targets["cont_call"] == [3]
    assert 2 not in uncertain


def test_map_continuations_callxargs_suffix_resolves_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(9)]),          # arg0
        make_fact(1, "PUSHCONT"),                             # cont
        make_fact(2, "CALLXARGS_1", args=[DummyArg(1), DummyArg(0)]),
        make_fact(3, "NOP"),
    ]
    mapping = {1: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_call"]
    assert 2 in returning_branches
    assert cont_targets["cont_call"] == [3]
    assert 2 not in uncertain


def test_map_continuations_callccargs_preserves_stack_for_following_if():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # cont_keep
        make_fact(1, "PUSHCONT"),  # cont_call
        make_fact(2, "CALLCCARGS", args=[DummyArg(0), DummyArg(0)]),  # p=0, r=0
        make_fact(3, "PUSHINT", args=[DummyArg(1)]),
        make_fact(4, "IF"),
        make_fact(5, "NOP"),
    ]
    mapping = {0: ["cont_keep"], 1: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_call"]
    assert 2 in returning_branches
    assert branch_map[4] == ["cont_keep"]
    assert 4 in returning_branches
    assert 4 not in uncertain


def test_map_continuations_callccvarargs_preserves_underlying_stack():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),                         # cont_keep
        make_fact(1, "PUSHINT", args=[DummyArg(42)]),    # arg0
        make_fact(2, "PUSHCONT"),                         # cont_call
        make_fact(3, "PUSHINT", args=[DummyArg(1)]),     # p
        make_fact(4, "PUSHINT", args=[DummyArg(0)]),     # r
        make_fact(5, "CALLCCVARARGS"),                    # consumes r, p, cont_call, arg0
        make_fact(6, "PUSHINT", args=[DummyArg(1)]),
        make_fact(7, "IF"),                               # should still resolve cont_keep
        make_fact(8, "NOP"),
    ]
    mapping = {0: ["cont_keep"], 2: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[5] == ["cont_call"]
    assert 5 in returning_branches
    assert branch_map[7] == ["cont_keep"]
    assert 7 in returning_branches
    assert 7 not in uncertain


def test_map_continuations_calldict_uses_c3():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "POPCTR", args=[DummyArg(3)]),  # c3 = cont_0
        make_fact(2, "CALLDICT", args=[DummyArg(1)]),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 in returning_branches
    assert cont_targets["cont_0"] == [3]
    assert not uncertain


def test_map_continuations_calldict_unknown_c3_is_not_marked_uncertain():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "CALLDICT", args=[DummyArg(1)]),
        make_fact(1, "NOP"),
    ]

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, pushcont_to_cont_ids={}, inline_cont_map={}
    )

    assert 0 not in branch_map
    assert 0 not in returning_branches
    assert not cont_targets
    assert 0 not in uncertain


def test_map_continuations_setretctr_does_not_alias_stack_cont_to_c0():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "SETRETCTR", args=[DummyArg(3)]),
        make_fact(2, "PUSHCTR", args=[DummyArg(0)]),
        make_fact(3, "EXECUTE"),
        make_fact(4, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # SETRETCTR mutates c0 internals; it must not resolve to the popped stack continuation.
    assert 3 not in branch_map
    assert 3 not in returning_branches
    assert "cont_0" not in cont_targets
    assert 3 not in uncertain


def test_map_continuations_popsave_c3_propagates_to_calldict():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "POPSAVE", args=[DummyArg(3)]),
        make_fact(2, "CALLDICT", args=[DummyArg(1)]),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 in returning_branches
    assert cont_targets["cont_0"] == [3]
    assert not uncertain


def test_map_continuations_preparedict_pushes_c3_cont_for_execute():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PREPAREDICT", args=[DummyArg(7)]),
        make_fact(1, "EXECUTE"),
        make_fact(2, "NOP"),
    ]

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        initial_ctrl_regs={3: "cont_method"},
    )

    assert branch_map[1] == ["cont_method"]
    assert 1 in returning_branches
    assert cont_targets["cont_method"] == [2]
    assert 1 not in uncertain


def test_map_continuations_pushctrx_with_constant_index_uses_ctrl_reg():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "POPCTR", args=[DummyArg(3)]),       # c3 = cont_method
        make_fact(2, "PUSHINT", args=[DummyArg(3)]),      # index for PUSHCTRX
        make_fact(3, "PUSHCTRX"),
        make_fact(4, "EXECUTE"),
        make_fact(5, "NOP"),
    ]
    mapping = {0: ["cont_method"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_method"]
    assert 4 in returning_branches
    assert cont_targets["cont_method"] == [5]
    assert 4 not in uncertain


def test_map_continuations_popctrx_with_constant_index_propagates_ctrl_reg():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),                          # x=cont_method
        make_fact(1, "PUSHINT", args=[DummyArg(3)]),      # i=3
        make_fact(2, "POPCTRX"),                          # c3 = cont_method
        make_fact(3, "CALLDICT", args=[DummyArg(7)]),
        make_fact(4, "NOP"),
    ]
    mapping = {0: ["cont_method"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[3] == ["cont_method"]
    assert 3 in returning_branches
    assert cont_targets["cont_method"] == [4]
    assert 3 not in uncertain


def test_map_continuations_popctrx_invalid_index_does_not_propagate_ctrl_reg():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),                          # x=cont_method
        make_fact(1, "PUSHINT", args=[DummyArg(6)]),      # i=6 (invalid control-reg index)
        make_fact(2, "POPCTRX"),
        make_fact(3, "PUSHINT", args=[DummyArg(6)]),
        make_fact(4, "PUSHCTRX"),
        make_fact(5, "EXECUTE"),
        make_fact(6, "NOP"),
    ]
    mapping = {0: ["cont_method"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 5 not in branch_map
    assert 5 not in returning_branches
    assert "cont_method" not in cont_targets
    assert 5 not in uncertain


def test_map_continuations_setcontctrx_pops_index_and_does_not_leak_extra_value():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),                          # cont_x (must be fully consumed)
        make_fact(1, "PUSHCONT"),                          # cont_c
        make_fact(2, "PUSHINT", args=[DummyArg(3)]),      # i
        make_fact(3, "SETCONTCTRX"),
        make_fact(4, "PUSHINT", args=[DummyArg(1)]),
        make_fact(5, "IF"),                                # consumes synthetic c'
        make_fact(6, "PUSHINT", args=[DummyArg(1)]),
        make_fact(7, "IF"),                                # must not resolve leaked cont_x
    ]
    mapping = {0: ["cont_x"], 1: ["cont_c"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 5 in branch_map
    assert 5 in returning_branches
    assert 7 not in branch_map
    assert 7 in uncertain
    assert "cont_x" not in cont_targets


def test_map_continuations_samealtsave_invalidates_precise_c0_identity():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "POPCTR", args=[DummyArg(0)]),       # c0 = cont_0
        make_fact(2, "PUSHCONT"),
        make_fact(3, "POPCTR", args=[DummyArg(1)]),       # c1 = cont_1
        make_fact(4, "SAMEALTSAVE"),
        make_fact(5, "PUSHCTR", args=[DummyArg(0)]),      # c0 should be unknown after SAMEALTSAVE
        make_fact(6, "EXECUTE"),
    ]
    mapping = {0: ["cont_0"], 2: ["cont_1"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 6 not in branch_map
    assert 6 not in returning_branches
    assert 6 not in uncertain
    assert not cont_targets


def test_map_continuations_try_consumes_handler_but_targets_body_only():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # body
        make_fact(1, "PUSHCONT"),  # handler
        make_fact(2, "TRY"),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_body"], 1: ["cont_handler"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_body"]
    assert 2 in returning_branches
    assert cont_targets["cont_body"] == [3]
    assert "cont_handler" not in cont_targets
    assert 2 not in uncertain


def test_map_continuations_try_does_not_misresolve_top_handler_as_body():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(0)]),
        make_fact(1, "PUSHINT", args=[DummyArg(0)]),
        make_fact(2, "SETCONTCTR", args=[DummyArg(1)]),  # unknown continuation placeholder
        make_fact(3, "PUSHCONT"),                        # known continuation is handler (TOS)
        make_fact(4, "TRY"),
        make_fact(5, "NOP"),
    ]
    mapping = {3: ["cont_handler"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 4 in branch_map
    assert all(cont_id != "cont_handler" for cont_id in branch_map[4])
    assert 4 in returning_branches
    assert "cont_handler" not in cont_targets
    assert 4 not in uncertain


def test_map_continuations_booleval_is_returning_call():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "BOOLEVAL"),
        make_fact(2, "NOP"),
    ]
    mapping = {0: ["cont_bool"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[1] == ["cont_bool"]
    assert 1 in returning_branches
    assert cont_targets["cont_bool"] == [2]
    assert 1 not in uncertain


def test_map_continuations_composboth_does_not_reuse_input_continuations():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "COMPOSBOTH"),
        make_fact(3, "EXECUTE"),
    ]
    mapping = {0: ["cont_a"], 1: ["cont_b"]}

    branch_map, _, _, _, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    if 3 in branch_map:
        assert "cont_a" not in branch_map[3]
        assert "cont_b" not in branch_map[3]


def test_map_continuations_setcontctrmany_does_not_reuse_input_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "SETCONTCTRMANY", args=[DummyArg(1)]),
        make_fact(2, "EXECUTE"),
    ]
    mapping = {0: ["cont_a"]}

    branch_map, _, _, _, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    if 2 in branch_map:
        assert "cont_a" not in branch_map[2]


def test_map_continuations_initial_ctrl_regs_resolves_pushctr_c3():
    """PUSHCTR c3 followed by JMPX should resolve when ctrl_regs propagated."""
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCTR", args=[DummyArg(3)]),
        make_fact(1, "JMPX"),
    ]

    branch_map, _, _, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts,
        pushcont_to_cont_ids={},
        inline_cont_map={},
        initial_ctrl_regs={3: "cont_method_dict"},
    )

    assert 1 in branch_map
    assert branch_map[1] == ["cont_method_dict"]
    assert 1 not in uncertain


def test_map_continuations_callxvarargs_resolves_constant_p():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(42)]),  # arg0
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHINT", args=[DummyArg(1)]),   # p
        make_fact(3, "PUSHINT", args=[DummyArg(0)]),   # r
        make_fact(4, "CALLXVARARGS"),
        make_fact(5, "NOP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_0"]
    assert 4 in returning_branches
    assert cont_targets["cont_0"] == [5]
    assert not uncertain


def test_map_continuations_callxvarargs_p_minus_one_does_not_leak_stale_caller_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),                        # stale continuation below call frame
        make_fact(1, "PUSHCONT"),                        # call target
        make_fact(2, "PUSHINT", args=[DummyArg(-1)]),   # p = -1 (pass whole stack)
        make_fact(3, "PUSHINT", args=[DummyArg(0)]),    # r = 0
        make_fact(4, "CALLXVARARGS"),
        make_fact(5, "PUSHINT", args=[DummyArg(1)]),
        make_fact(6, "IF"),
        make_fact(7, "NOP"),
    ]
    mapping = {0: ["cont_stale"], 1: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_call"]
    assert 4 in returning_branches
    assert cont_targets["cont_call"] == [5]
    assert 6 not in branch_map
    assert 6 in uncertain
    assert "cont_stale" not in cont_targets


def test_map_continuations_callccvarargs_p_minus_one_does_not_leak_stale_caller_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),                        # stale continuation below call frame
        make_fact(1, "PUSHCONT"),                        # call target
        make_fact(2, "PUSHINT", args=[DummyArg(-1)]),   # p = -1 (pass whole stack)
        make_fact(3, "PUSHINT", args=[DummyArg(0)]),    # r = 0
        make_fact(4, "CALLCCVARARGS"),
        make_fact(5, "PUSHINT", args=[DummyArg(1)]),
        make_fact(6, "IF"),
        make_fact(7, "NOP"),
    ]
    mapping = {0: ["cont_stale"], 1: ["cont_call"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[4] == ["cont_call"]
    assert 4 in returning_branches
    assert cont_targets["cont_call"] == [5]
    assert 6 not in branch_map
    assert 6 in uncertain
    assert "cont_stale" not in cont_targets


def test_map_continuations_callxvarargs_unknown_tail_single_candidate_fallback():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT", continuation_id="cont_ctx"),
        make_fact(1, "PUSHINT", continuation_id="cont_ctx", args=[DummyArg(8)]),
        make_fact(2, "PUSHINT", continuation_id="cont_ctx", args=[DummyArg(4)]),
        make_fact(3, "CALLXVARARGS", continuation_id="cont_ctx"),
        make_fact(4, "NOP", continuation_id="cont_ctx"),
    ]
    mapping = {0: ["cont_target"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[3] == ["cont_target"]
    assert 3 in returning_branches
    assert cont_targets["cont_target"] == [4]
    assert 3 not in uncertain


def test_map_continuations_callxvarargs_unknown_tail_multi_candidate_stays_uncertain():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT", continuation_id="cont_ctx"),
        make_fact(1, "PUSHCONT", continuation_id="cont_ctx"),
        make_fact(2, "PUSHINT", continuation_id="cont_ctx", args=[DummyArg(8)]),
        make_fact(3, "PUSHINT", continuation_id="cont_ctx", args=[DummyArg(4)]),
        make_fact(4, "CALLXVARARGS", continuation_id="cont_ctx"),
    ]
    mapping = {0: ["cont_a"], 1: ["cont_b"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 4 not in branch_map
    assert 4 not in returning_branches
    assert not cont_targets
    assert 4 in uncertain


def test_map_continuations_jmpxvarargs_unknown_tail_single_candidate_fallback():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT", continuation_id="cont_ctx"),
        make_fact(1, "PUSHINT", continuation_id="cont_ctx", args=[DummyArg(8)]),  # p
        make_fact(2, "JMPXVARARGS", continuation_id="cont_ctx"),
    ]
    mapping = {0: ["cont_target"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_target"]
    assert 2 not in returning_branches
    assert not cont_targets
    assert 2 not in uncertain


def test_map_continuations_cont_context_starts_with_unknown_tail():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT", continuation_id="cont_0"),
        make_fact(
            1,
            "CALLXARGS",
            continuation_id="cont_0",
            args=[DummyArg(6), DummyArg(-1)],
        ),
        make_fact(2, "NOP", continuation_id="cont_0"),
    ]
    mapping = {0: ["callee_cont"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # Continuation contexts have caller-provided values below local stack.
    # Even when CALLXARGS expects continuation below p arguments, we should
    # still recover the PUSHCONT-tracked continuation from the known prefix.
    assert branch_map[1] == ["callee_cont"]
    assert 1 in returning_branches
    assert cont_targets["callee_cont"] == [2]
    assert 1 not in uncertain


def test_map_continuations_unknown_tail_recovers_ifjmp_continuation():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "BLKDROP2"),  # Invalidates stack -> unknown_below=True
        make_fact(1, "PUSHCONT"),
        make_fact(2, "IFJMP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 not in returning_branches
    assert "cont_0" not in cont_targets
    assert 2 not in uncertain


def test_map_continuations_unknown_tail_recovers_ifelse_continuations():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "BLKDROP2"),  # Invalidates stack -> unknown_below=True
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHCONT"),
        make_fact(3, "IFELSE"),
        make_fact(4, "NOP"),
    ]
    mapping = {1: ["cont_a"], 2: ["cont_b"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    # Top-of-stack continuation is consumed first.
    assert branch_map[3] == ["cont_b", "cont_a"]
    assert 3 in returning_branches
    assert cont_targets["cont_b"] == [4]
    assert cont_targets["cont_a"] == [4]
    assert 3 not in uncertain


def test_map_continuations_ifjmp_accepts_top_cont_layout():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(1)]),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "IFJMP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 not in returning_branches
    assert "cont_0" not in cont_targets
    assert 2 not in uncertain


def test_map_continuations_ifelse_accepts_top_cont_layout():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(1)]),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "PUSHCONT"),
        make_fact(3, "IFELSE"),
        make_fact(4, "NOP"),
    ]
    mapping = {1: ["cont_a"], 2: ["cont_b"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[3] == ["cont_b", "cont_a"]
    assert 3 in returning_branches
    assert cont_targets["cont_b"] == [4]
    assert cont_targets["cont_a"] == [4]
    assert 3 not in uncertain


def test_map_continuations_if_accepts_top_cont_layout():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHINT", args=[DummyArg(1)]),
        make_fact(1, "PUSHCONT"),
        make_fact(2, "IF"),
        make_fact(3, "NOP"),
    ]
    mapping = {1: ["cont_0"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert branch_map[2] == ["cont_0"]
    assert 2 in returning_branches
    assert cont_targets["cont_0"] == [3]
    assert 2 not in uncertain


def test_map_continuations_if_rejects_known_continuation_in_condition_slot():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # candidate target continuation
        make_fact(1, "PUSHCONT"),  # occupies condition slot (known continuation) -> VM type mismatch
        make_fact(2, "IF"),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_target"], 1: ["cont_not_bool"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 2 not in branch_map
    assert 2 not in returning_branches
    assert "cont_target" not in cont_targets
    assert 2 in uncertain
    assert any(info.reason == "type_mismatch" for info in uncertain[2])


def test_map_continuations_ifjmp_rejects_known_continuation_in_condition_slot():
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),  # candidate jump continuation
        make_fact(1, "PUSHCONT"),  # occupies condition slot (known continuation) -> VM type mismatch
        make_fact(2, "IFJMP"),
        make_fact(3, "NOP"),
    ]
    mapping = {0: ["cont_target"], 1: ["cont_not_bool"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = analyzer._continuation_resolver.map_continuations(
        facts, mapping, inline_cont_map={}
    )

    assert 2 not in branch_map
    assert 2 not in returning_branches
    assert "cont_target" not in cont_targets
    assert 2 in uncertain
    assert any(info.reason == "type_mismatch" for info in uncertain[2])


def test_map_continuations_pseudopushref_emits_entry_shape():
    """PSEUDO_PUSHREF should record current stack as entry_shape for inline continuation."""
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PUSHCONT"),       # pushes cont_a onto stack
        make_fact(1, "PSEUDO_PUSHREF"), # inlines cont_b, should emit entry_shape
        make_fact(2, "NOP"),
    ]
    pushcont_mapping = {0: ["cont_a"]}
    inline_mapping = {1: ["cont_b"]}

    cont_entry_shapes: Dict[str, List] = {}

    # Call the resolver directly to capture cont_entry_shapes
    branch_map, returning_branches, cont_targets, uncertain, _ = \
        analyzer._continuation_resolver.map_continuations(
            facts, pushcont_mapping, inline_mapping,
            cont_entry_shapes=cont_entry_shapes,
        )

    # cont_b should have an entry_shape recorded
    assert "cont_b" in cont_entry_shapes, f"cont_b not in cont_entry_shapes: {cont_entry_shapes}"
    shapes = cont_entry_shapes["cont_b"]
    assert len(shapes) == 1
    shape_stack, shape_unknown = shapes[0]
    # The stack at PSEUDO_PUSHREF time had cont_a (from PUSHCONT)
    # So entry_shape should contain cont_a
    cont_tokens = [t for t in shape_stack if t[0] == "cont"]
    assert len(cont_tokens) >= 1, f"Expected cont token in shape, got {shape_stack}"
    assert cont_tokens[0][1] == "cont_a"


def test_map_continuations_pseudopushref_pushes_cont_to_stack():
    """PSEUDO_PUSHREF should push its inline continuation as STACK_CONT for parent consumption."""
    analyzer = ProgramAnalyzer()
    facts = [
        make_fact(0, "PSEUDO_PUSHREF"),  # inlines cont_x
        make_fact(1, "IFJMP"),           # should consume cont_x from stack
    ]
    pushcont_mapping = {}
    inline_mapping = {0: ["cont_x"]}

    branch_map, returning_branches, cont_targets, uncertain, _ = \
        analyzer._continuation_resolver.map_continuations(
            facts, pushcont_mapping, inline_mapping,
        )

    # IFJMP should have resolved cont_x
    assert 1 in branch_map, f"IFJMP not in branch_map: {branch_map}"
    assert branch_map[1] == ["cont_x"]
    assert 1 not in uncertain


def test_analyze_stack_uses_parent_context_height():
    analyzer = ProgramAnalyzer()
    main_facts = [
        make_fact(0, "PUSHCONT"),
        make_fact(1, "PUSHINT"),
        make_fact(2, "IF"),
        make_fact(3, "NOP"),
    ]
    cont_fact = make_fact(4, "PUSHINT", continuation_id="cont_0", parent_index=2)
    all_facts = main_facts + [cont_fact]

    continuations = {
        "cont_0": Continuation(
            id="cont_0",
            instructions=[],
            parent_context="main",
            parent_local_index=1,
            parent_instruction_index=2,
            entry_index=0,
        )
    }

    # Use initial_height=0 to simulate empty stack scenario for testing
    states = analyzer._stack_analyzer.analyze_stack(all_facts, continuations, initial_height=0)
    cont_state = next(state for state in states if state.instruction_index == 4)

    assert cont_state.height_before == 0
    assert cont_state.unknown is False
