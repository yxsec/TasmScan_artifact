"""
Tests for stack effects database coverage.

Verifies that MANUAL_STACK_EFFECTS in stack_effects.py:
1. Cover important operations (SENDRAWMSG, dictionary ops)
2. Have correct tuple format
3. Define meaningful stack effects
"""

import json
from pathlib import Path

from tasmscan.ir.stack_effects import (
    StackEffectDatabase,
    StackEffect,
    get_stack_effect,
    get_stack_effect_db,
)


# Access MANUAL_STACK_EFFECTS from the class
MANUAL_STACK_EFFECTS = StackEffectDatabase.MANUAL_STACK_EFFECTS


class TestStackEffectsCoverage:
    """Test stack effects database coverage."""

    def test_sendrawmsg_defined(self):
        """SENDRAWMSG should have stack effect defined."""
        assert "SENDRAWMSG" in MANUAL_STACK_EFFECTS
        effect = MANUAL_STACK_EFFECTS["SENDRAWMSG"]
        # SENDRAWMSG: pops (msg, mode), pushes nothing
        pops, pushes, pop_names, push_names, variadic = effect
        assert pops == 2
        assert pushes == 0

    def test_dict_operations_defined(self):
        """Dictionary operations should have stack effects."""
        dict_ops = ["DICTGET", "DICTSET", "DICTDEL", "STDICT", "LDDICT"]
        for op in dict_ops:
            assert op in MANUAL_STACK_EFFECTS, f"{op} not defined"

    def test_dictget_stack_effect(self):
        """DICTGET should have correct stack effect."""
        assert "DICTGET" in MANUAL_STACK_EFFECTS
        pops, pushes, pop_names, push_names, variadic = MANUAL_STACK_EFFECTS["DICTGET"]
        # DICTGET: pops (key, dict, keylen), pushes (value, found)
        assert pops == 3
        assert pushes == 2

    def test_stack_effect_format(self):
        """Stack effects should have correct tuple format."""
        for opcode, effect in MANUAL_STACK_EFFECTS.items():
            assert len(effect) == 5, f"{opcode} has wrong format, expected 5-tuple"
            pops, pushes, pop_names, push_names, variadic = effect
            assert isinstance(pops, int), f"{opcode} pops should be int"
            assert isinstance(pushes, int), f"{opcode} pushes should be int"
            assert isinstance(pop_names, list), f"{opcode} pop_names should be list"
            assert isinstance(push_names, list), f"{opcode} push_names should be list"
            assert isinstance(variadic, bool), f"{opcode} variadic should be bool"

    def test_pop_names_count_matches(self):
        """pop_names list length should match pops count (when not variadic)."""
        for opcode, effect in MANUAL_STACK_EFFECTS.items():
            pops, pushes, pop_names, push_names, variadic = effect
            # For variadic/dynamic instructions, names may not match exactly
            if not variadic and pop_names:
                assert len(pop_names) == pops, \
                    f"{opcode}: pop_names length {len(pop_names)} != pops {pops}"

    def test_push_names_count_matches(self):
        """push_names list length should match pushes count (when not variadic)."""
        for opcode, effect in MANUAL_STACK_EFFECTS.items():
            pops, pushes, pop_names, push_names, variadic = effect
            # For variadic/dynamic instructions, names may not match exactly
            if not variadic and push_names:
                assert len(push_names) == pushes, \
                    f"{opcode}: push_names length {len(push_names)} != pushes {pushes}"

    def test_non_negative_values(self):
        """pops and pushes should be non-negative."""
        for opcode, effect in MANUAL_STACK_EFFECTS.items():
            pops, pushes, _, _, _ = effect
            assert pops >= 0, f"{opcode} has negative pops: {pops}"
            assert pushes >= 0, f"{opcode} has negative pushes: {pushes}"


class TestStackEffectDatabaseAPI:
    """Test StackEffectDatabase API."""

    def test_get_stack_effect_returns_effect(self):
        """get_stack_effect should return StackEffect for known opcodes."""
        effect = get_stack_effect("SENDRAWMSG")
        assert effect is not None
        assert isinstance(effect, StackEffect)
        assert effect.inputs == 2
        assert effect.outputs == 0

    def test_get_stack_effect_unknown_returns_none(self):
        """get_stack_effect should return None for unknown opcodes."""
        effect = get_stack_effect("TOTALLY_FAKE_OPCODE_XYZ123")
        assert effect is None

    def test_stack_effect_net_effect(self):
        """StackEffect.net_effect should compute correctly."""
        effect = get_stack_effect("DICTGET")
        assert effect is not None
        # DICTGET: 3 inputs, 2 outputs -> net = -1
        assert effect.net_effect == effect.outputs - effect.inputs

    def test_database_has_effect(self):
        """has_effect should return correct boolean."""
        db = get_stack_effect_db()
        assert db.has_effect("SENDRAWMSG") is True
        assert db.has_effect("TOTALLY_FAKE_OPCODE") is False

    def test_database_coverage_stats(self):
        """get_coverage_stats should return valid statistics."""
        db = get_stack_effect_db()
        stats = db.get_coverage_stats()
        assert "total_opcodes" in stats
        assert "with_stack_effects" in stats
        assert "no_stack_effect" in stats
        assert stats["total_opcodes"] > 0

    def test_alias_lookup_for_cfg_control_flow_variants(self):
        """Alias-only control-flow names should resolve to known stack effects."""
        assert get_stack_effect_db().has_effect("JMP_JMPREF") is True
        assert get_stack_effect("JMP_JMPREF") is not None

    def test_alias_lookup_for_callccargs_var(self):
        """CALLCCARGS_VAR should resolve through alias registry."""
        assert get_stack_effect_db().has_effect("CALLCCARGS_VAR") is True
        assert get_stack_effect("CALLCCARGS_VAR") is not None

    def test_instruction_table_coverage(self):
        """All opcodes in instruction_table.json should have stack effects."""
        table_path = Path(__file__).resolve().parents[1] / "tasmscan" / "disassembler" / "instruction_table.json"
        data = json.loads(table_path.read_text(encoding="utf-8"))
        names = [item["name"] for item in data.get("instructions", [])]
        missing = [name for name in names if not get_stack_effect_db().has_effect(name)]
        assert not missing, f"Missing stack effects for: {missing[:20]}"


class TestControlFlowInstructions:
    """Test stack effects for control flow instructions."""

    def test_accept_has_no_stack_effect(self):
        """ACCEPT should have no stack effect."""
        assert "ACCEPT" in MANUAL_STACK_EFFECTS
        pops, pushes, _, _, _ = MANUAL_STACK_EFFECTS["ACCEPT"]
        assert pops == 0
        assert pushes == 0

    def test_commit_has_no_stack_effect(self):
        """COMMIT should have no stack effect."""
        assert "COMMIT" in MANUAL_STACK_EFFECTS
        pops, pushes, _, _, _ = MANUAL_STACK_EFFECTS["COMMIT"]
        assert pops == 0
        assert pushes == 0


class TestContinuationInstructions:
    """Test stack effects for continuation instructions."""

    def test_popctr_defined(self):
        """POPCTR should have stack effect defined."""
        assert "POPCTR" in MANUAL_STACK_EFFECTS
        pops, pushes, _, _, _ = MANUAL_STACK_EFFECTS["POPCTR"]
        assert pops == 1  # Pops value to write to control register
        assert pushes == 0

    def test_pushctr_defined(self):
        """PUSHCTR should have stack effect defined."""
        assert "PUSHCTR" in MANUAL_STACK_EFFECTS
        pops, pushes, _, _, _ = MANUAL_STACK_EFFECTS["PUSHCTR"]
        assert pops == 0
        assert pushes == 1  # Pushes control register value


class TestDynamicInstructions:
    """Test stack effects for dynamic/variadic instructions."""

    def test_pick_is_dynamic(self):
        """PICK should be marked as dynamic."""
        assert "PICK" in MANUAL_STACK_EFFECTS
        _, _, _, _, variadic = MANUAL_STACK_EFFECTS["PICK"]
        assert variadic is True

    def test_roll_is_dynamic(self):
        """ROLL should be marked as dynamic."""
        assert "ROLL" in MANUAL_STACK_EFFECTS
        _, _, _, _, variadic = MANUAL_STACK_EFFECTS["ROLL"]
        assert variadic is True

    def test_callref_is_dynamic(self):
        """CALLREF should be marked as dynamic."""
        assert "CALLREF" in MANUAL_STACK_EFFECTS
        _, _, _, _, variadic = MANUAL_STACK_EFFECTS["CALLREF"]
        assert variadic is True
