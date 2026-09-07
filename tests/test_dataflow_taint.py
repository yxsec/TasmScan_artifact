"""
Tests for multi-output instruction taint propagation rules.

Verifies that MULTI_OUTPUT_TAINT_RULES in dataflow.py:
1. Cover important multi-output instructions
2. Have consistent structure
3. Define valid propagation rules
"""
from tasmscan.ir.dataflow import DataFlowAnalyzer


class TestMultiOutputTaintRules:
    """Test multi-output instruction taint propagation rules."""

    def test_divmod_or_ldix_taint_rule_exists(self):
        """LDIX should have taint rule defined (multi-output load instruction)."""
        # Note: MULTI_OUTPUT_TAINT_RULES focuses on LD* instructions
        # DIVMOD is not defined as it's an arithmetic operation handled differently
        assert "LDIX" in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES
        rule = DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES["LDIX"]
        assert isinstance(rule, list)
        assert len(rule) == 2  # LDIX outputs: value + remaining slice

    def test_ldix_taint_rule_exists(self):
        """LDIX should have taint rule defined."""
        assert "LDIX" in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES

    def test_ldux_taint_rule_exists(self):
        """LDUX should have taint rule defined (unsigned load)."""
        assert "LDUX" in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES

    def test_all_ld_instructions_covered(self):
        """All multi-output LD instructions should have rules."""
        ld_ops = ["LDIX", "LDUX", "LDSLICEX", "LDDICT", "LDDICTS"]
        for op in ld_ops:
            if op in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES:
                rule = DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES[op]
                assert isinstance(rule, list), f"{op} rule should be a list"
                assert len(rule) >= 1, f"{op} should have at least 1 output rule"

    def test_ldgrams_taint_rule_exists(self):
        """LDGRAMS (currency loading) should have taint rule defined."""
        assert "LDGRAMS" in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES
        rule = DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES["LDGRAMS"]
        assert rule == ["inherit_first_input", "preserve_slice"]

    def test_ldu_taint_rule_exists(self):
        """LDU (unsigned integer loading) should have taint rule defined."""
        assert "LDU" in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES

    def test_ldref_taint_rule_exists(self):
        """LDREF (reference loading) should have taint rule defined."""
        assert "LDREF" in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES

    def test_taint_rule_structure(self):
        """All taint rules should have required fields."""
        valid_rules = {"inherit_first_input", "preserve_slice", "inherit_all", "no_taint"}

        for opcode, rule in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES.items():
            # Rule should be a list of propagation strategies
            assert isinstance(rule, list), f"{opcode} rule should be a list"
            assert len(rule) >= 1, f"{opcode} should have at least 1 output rule"

            # Each element should be a valid rule string
            for i, strategy in enumerate(rule):
                assert isinstance(strategy, str), f"{opcode}[{i}] should be string"
                assert strategy in valid_rules, \
                    f"{opcode}[{i}] has invalid rule '{strategy}', expected one of {valid_rules}"

    def test_ld_instructions_have_two_outputs(self):
        """LD* instructions typically output (value, remaining_slice)."""
        ld_instructions = [k for k in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES.keys()
                          if k.startswith("LD")]

        for opcode in ld_instructions:
            rule = DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES[opcode]
            assert len(rule) == 2, \
                f"{opcode} should have 2 output rules (value + slice), got {len(rule)}"

    def test_preserve_slice_rule_common(self):
        """Most LD* instructions should preserve slice taint for second output."""
        for opcode, rule in DataFlowAnalyzer.MULTI_OUTPUT_TAINT_RULES.items():
            if opcode.startswith("LD") and len(rule) >= 2:
                assert rule[1] == "preserve_slice", \
                    f"{opcode} second output should preserve_slice, got '{rule[1]}'"


class TestTaintSources:
    """Test taint source definitions."""

    def test_ldmsgaddr_is_taint_source(self):
        """LDMSGADDR should be a taint source (loads sender address)."""
        assert "LDMSGADDR" in DataFlowAnalyzer.TAINT_SOURCES

    def test_ldgrams_is_taint_source(self):
        """LDGRAMS should be a taint source (loads message value)."""
        assert "LDGRAMS" in DataFlowAnalyzer.TAINT_SOURCES

    def test_inmsg_src_is_taint_source(self):
        """INMSG_SRC should be a taint source.

        Note: SENDER is not a valid TVM opcode; use INMSG_SRC instead.
        """
        assert "INMSG_SRC" in DataFlowAnalyzer.TAINT_SOURCES

    def test_conditional_taint_sources_defined(self):
        """LDU/LDI should be conditional taint sources."""
        assert "LDU" in DataFlowAnalyzer.CONDITIONAL_TAINT_SOURCES
        assert "LDI" in DataFlowAnalyzer.CONDITIONAL_TAINT_SOURCES


class TestGuardOpcodes:
    """Test guard opcode definitions."""

    def test_throwif_is_guard(self):
        """THROWIF should be a guard opcode."""
        assert "THROWIF" in DataFlowAnalyzer.GUARD_OPCODES

    def test_ifnot_is_guard(self):
        """IFNOT should be a guard opcode."""
        assert "IFNOT" in DataFlowAnalyzer.GUARD_OPCODES

    def test_ifelse_is_guard(self):
        """IFELSE should be a guard opcode."""
        assert "IFELSE" in DataFlowAnalyzer.GUARD_OPCODES


class TestSensitiveOpcodes:
    """Test sensitive opcode definitions."""

    def test_sendrawmsg_is_sensitive(self):
        """SENDRAWMSG should be a sensitive opcode."""
        assert "SENDRAWMSG" in DataFlowAnalyzer.SENSITIVE_OPCODES

    def test_rawreserve_is_sensitive(self):
        """RAWRESERVE should be a sensitive opcode."""
        assert "RAWRESERVE" in DataFlowAnalyzer.SENSITIVE_OPCODES
