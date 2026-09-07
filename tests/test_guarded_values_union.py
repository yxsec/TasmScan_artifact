"""
Tests for guarded_values union merge strategy in path-sensitive dataflow analysis.

Background:
    In _merge_path_states(), guarded_values was changed from intersection to union
    to reduce false positives. A value is considered guarded if it was checked on
    ANY path, not ALL paths.

    Improved handling: guarded_values is now Dict[int, int] (definition_site -> access_count)
    for LRU-based pruning instead of Set[int].

This test file verifies:
    1. Union merge behavior for guarded_values across paths
    2. State merging correctness for path-sensitive analysis
"""
from tasmscan.ir.dataflow import DataFlowState, DataFlowValue, ValueSource
from tasmscan.ir.path_sensitive_dataflow import PathState, PathSensitiveDataFlowAnalyzer


class TestGuardedValuesUnionMerge:
    """Test guarded_values union merge strategy"""

    def test_guarded_values_union_single_path_checked(self):
        """
        Test: When a value is checked on only one path, it should still be
        in merged guarded_values.

        Scenario:
            - Path A: value 10 is guarded (checked by IF/THROW)
            - Path B: value 10 is NOT guarded

        Expected: After merge, guarded_values should contain 10
        (union strategy means checked on ANY path is sufficient)
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        # Path A: value 10 is guarded
        state_a = PathState(
            path_id=0,
            dataflow_state=DataFlowState(
                stack=[],
                registers={},
                guarded_values={10: 1}  # Value 10 was checked on this path
            )
        )

        # Path B: value 10 is NOT guarded
        state_b = PathState(
            path_id=1,
            dataflow_state=DataFlowState(
                stack=[],
                registers={},
                guarded_values={}  # No values checked on this path
            )
        )

        merged = analyzer._merge_path_states([state_a, state_b])

        # Union strategy: 10 should be in guarded_values because it was checked on path A
        assert 10 in merged.dataflow_state.guarded_values

    def test_guarded_values_union_disjoint_sets(self):
        """
        Test: Different paths check different values.

        Scenario:
            - Path A: checks value 10
            - Path B: checks value 20

        Expected: Merged guarded_values keys = {10, 20}
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        state_a = PathState(
            path_id=0,
            dataflow_state=DataFlowState(
                stack=[],
                registers={},
                guarded_values={10: 1}
            )
        )

        state_b = PathState(
            path_id=1,
            dataflow_state=DataFlowState(
                stack=[],
                registers={},
                guarded_values={20: 1}
            )
        )

        merged = analyzer._merge_path_states([state_a, state_b])

        # Union: both 10 and 20 should be guarded (check keys)
        assert set(merged.dataflow_state.guarded_values.keys()) == {10, 20}

    def test_guarded_values_union_overlapping_sets(self):
        """
        Test: Paths have overlapping guarded values.

        Scenario:
            - Path A: checks {10, 20}
            - Path B: checks {20, 30}

        Expected: Merged guarded_values keys = {10, 20, 30}
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        state_a = PathState(
            path_id=0,
            dataflow_state=DataFlowState(
                stack=[],
                registers={},
                guarded_values={10: 1, 20: 2}
            )
        )

        state_b = PathState(
            path_id=1,
            dataflow_state=DataFlowState(
                stack=[],
                registers={},
                guarded_values={20: 3, 30: 1}
            )
        )

        merged = analyzer._merge_path_states([state_a, state_b])

        # Check keys for union behavior
        assert set(merged.dataflow_state.guarded_values.keys()) == {10, 20, 30}
        # Overlapping key 20 should have max access_count (3)
        assert merged.dataflow_state.guarded_values[20] == 3

    def test_guarded_values_union_multiple_paths(self):
        """
        Test: Union merge with more than two paths.

        Scenario:
            - Path A: checks {1}
            - Path B: checks {2}
            - Path C: checks {3}
            - Path D: checks {} (empty)

        Expected: Merged guarded_values keys = {1, 2, 3}
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        guarded_dicts = [{1: 1}, {2: 1}, {3: 1}, {}]
        states = [
            PathState(
                path_id=i,
                dataflow_state=DataFlowState(
                    stack=[], registers={}, guarded_values=guarded
                )
            )
            for i, guarded in enumerate(guarded_dicts)
        ]

        merged = analyzer._merge_path_states(states)

        assert set(merged.dataflow_state.guarded_values.keys()) == {1, 2, 3}


class TestPathStateMerging:
    """Test path state merging for path-sensitive analysis"""

    def test_merge_empty_states_list(self):
        """
        Test: Merging empty states list returns fresh state.

        This handles unreachable code or analysis entry points.
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        merged = analyzer._merge_path_states([])

        assert merged.dataflow_state.stack == []
        assert merged.dataflow_state.registers == {}
        assert merged.dataflow_state.guarded_values == {}

    def test_merge_single_state_unchanged(self):
        """
        Test: Merging single state returns that state unchanged.
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        original = PathState(
            path_id=42,
            dataflow_state=DataFlowState(
                stack=[
                    DataFlowValue(ValueSource.MESSAGE_SENDER, definition_site=5, tainted=True)
                ],
                registers={0: DataFlowValue(ValueSource.CONSTANT, definition_site=1)},
                guarded_values={5: 1, 10: 2}
            )
        )

        merged = analyzer._merge_path_states([original])

        assert merged is original  # Should return same object

    def test_merge_tainted_on_any_path(self):
        """
        Test: Value is tainted in merged state if tainted on ANY path.

        Scenario:
            - Path A: stack[0] is tainted
            - Path B: stack[0] is NOT tainted

        Expected: Merged stack[0] is tainted (conservative)
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        tainted_value = DataFlowValue(
            ValueSource.MESSAGE_SENDER,
            definition_site=10,
            tainted=True,
            checked=False
        )
        clean_value = DataFlowValue(
            ValueSource.MESSAGE_SENDER,
            definition_site=10,
            tainted=False,
            checked=False
        )

        state_a = PathState(
            path_id=0,
            dataflow_state=DataFlowState(
                stack=[tainted_value],
                registers={},
                guarded_values={}
            )
        )

        state_b = PathState(
            path_id=1,
            dataflow_state=DataFlowState(
                stack=[clean_value],
                registers={},
                guarded_values={}
            )
        )

        merged = analyzer._merge_path_states([state_a, state_b])

        # Should be tainted because one path has tainted value
        assert merged.dataflow_state.stack[0].tainted is True

    def test_merge_checked_requires_all_tainted_checked(self):
        """
        Test: Merged value is checked only if ALL tainted values are checked.

        Scenario:
            - Path A: tainted + checked
            - Path B: tainted + NOT checked

        Expected: Merged value is tainted but NOT checked (conservative)
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        checked_value = DataFlowValue(
            ValueSource.MESSAGE_SENDER,
            definition_site=10,
            tainted=True,
            checked=True
        )
        unchecked_value = DataFlowValue(
            ValueSource.MESSAGE_SENDER,
            definition_site=10,
            tainted=True,
            checked=False
        )

        state_a = PathState(
            path_id=0,
            dataflow_state=DataFlowState(
                stack=[checked_value],
                registers={},
                guarded_values={10: 1}
            )
        )

        state_b = PathState(
            path_id=1,
            dataflow_state=DataFlowState(
                stack=[unchecked_value],
                registers={},
                guarded_values={}
            )
        )

        merged = analyzer._merge_path_states([state_a, state_b])

        # Tainted is conservative (ANY)
        assert merged.dataflow_state.stack[0].tainted is True
        # Checked requires ALL tainted to be checked
        assert merged.dataflow_state.stack[0].checked is False
        # But guarded_values uses union - 10 should be there from path A
        assert 10 in merged.dataflow_state.guarded_values

    def test_merge_stack_uses_maximum_size(self):
        """
        Test: Stack merge uses maximum size across all paths to preserve taint info.

        Scenario:
            - Path A: stack has 3 elements
            - Path B: stack has 1 element

        Expected: Merged stack has 3 elements (maximum), preserving all taint info.
        Values at positions that don't exist on all paths are still included,
        as they may contain taint information that should not be lost.
        """
        analyzer = PathSensitiveDataFlowAnalyzer()

        val1 = DataFlowValue(ValueSource.CONSTANT, definition_site=1)
        val2 = DataFlowValue(ValueSource.MESSAGE_SENDER, definition_site=2, tainted=True)
        val3 = DataFlowValue(ValueSource.CONSTANT, definition_site=3)

        state_a = PathState(
            path_id=0,
            dataflow_state=DataFlowState(
                stack=[val1, val2, val3],  # 3 elements, val2 is tainted
                registers={},
                guarded_values={}
            )
        )

        state_b = PathState(
            path_id=1,
            dataflow_state=DataFlowState(
                stack=[val1],  # 1 element
                registers={},
                guarded_values={}
            )
        )

        merged = analyzer._merge_path_states([state_a, state_b])

        # Use maximum size to preserve taint info from all paths
        assert len(merged.dataflow_state.stack) == 3
        # Tainted value at position 1 should be preserved
        assert merged.dataflow_state.stack[1].tainted is True
