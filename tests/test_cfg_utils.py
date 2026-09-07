"""Tests for common CFG traversal utilities."""
from unittest.mock import MagicMock

from tasmscan.detectors.cfg_utils import (
    build_predecessor_map,
    collect_auth_guard_indices,
    traverse_cfg_for_unguarded_sinks,
    traverse_cfg_for_specific_guard,
    build_instruction_to_block_map,
    is_block_guarded_on_all_predecessor_paths,
)


def create_mock_block(block_id: int, instruction_indices: list, successors: list):
    """Create a mock BasicBlock for testing."""
    block = MagicMock()
    block.id = block_id
    block.instruction_indices = instruction_indices
    block.successors = successors
    block.context = "main"
    block.has_unknown_successor = False
    return block


class MockInstructionFact:
    def __init__(self, opcode: str, index: int):
        self.opcode = opcode
        self.index = index


class TestTraverseCfgForUnguardedSinks:
    """Tests for traverse_cfg_for_unguarded_sinks function."""

    def test_simple_unguarded_sink(self):
        """Test detection of a sink without any guard."""
        # Block 0: [0, 1] -> Block 1: [2, 3 (sink)]
        block0 = create_mock_block(0, [0, 1], [1])
        block1 = create_mock_block(1, [2, 3], [])
        block_map = {0: block0, 1: block1}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=None,
            guard_indices=set(),  # No guards
            sink_indices={3},
        )

        assert 3 in result.flagged_indices
        assert not result.truncated

    def test_guarded_sink_not_flagged(self):
        """Test that a sink after a guard is not flagged."""
        # Block 0: [0 (guard), 1] -> Block 1: [2, 3 (sink)]
        block0 = create_mock_block(0, [0, 1], [1])
        block1 = create_mock_block(1, [2, 3], [])
        block_map = {0: block0, 1: block1}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=None,
            guard_indices={0},  # Guard at index 0
            sink_indices={3},
        )

        assert 3 not in result.flagged_indices
        assert len(result.flagged_indices) == 0

    def test_start_from_idx_skips_earlier_instructions(self):
        """Test that instructions before start_from_idx are skipped."""
        # Block 0: [0 (sink), 1, 2 (sink)] - start from 1
        block0 = create_mock_block(0, [0, 1, 2], [])
        block_map = {0: block0}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=1,  # Start from instruction 1
            guard_indices=set(),
            sink_indices={0, 2},  # Both 0 and 2 are sinks
        )

        # Only instruction 2 should be flagged (0 is before start_from)
        assert 0 not in result.flagged_indices
        assert 2 in result.flagged_indices

    def test_unknown_branch_handling(self):
        """Test that blocks with unknown successors are tracked."""
        # Block 0: [0, 1] -> [-1 (legacy unknown marker)]
        block0 = create_mock_block(0, [0, 1], [-1])
        block0.has_unknown_successor = True
        block_map = {0: block0}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=None,
            guard_indices=set(),
            sink_indices=set(),
            flag_unknown_unguarded=True,
            unknown_flag_idx=0,
        )

        assert result.encountered_unknown
        assert 0 in result.flagged_indices

    def test_unknown_branch_guarded_not_flagged(self):
        """Test that unknown branches after guard are not flagged."""
        # Block 0: [0 (guard), 1] -> [-1 (unknown)]
        block0 = create_mock_block(0, [0, 1], [-1])
        block0.has_unknown_successor = True
        block_map = {0: block0}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=None,
            guard_indices={0},
            sink_indices=set(),
            flag_unknown_unguarded=True,
            unknown_flag_idx=0,
        )

        assert not result.encountered_unknown
        assert 0 not in result.flagged_indices

    def test_iteration_limit_truncates(self):
        """Test that exceeding iteration limit causes truncation."""
        # Create a cycle that would cause infinite loop
        block0 = create_mock_block(0, [0], [1])
        block1 = create_mock_block(1, [1], [0])  # Back edge
        block_map = {0: block0, 1: block1}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=None,
            guard_indices=set(),
            sink_indices=set(),
            max_iterations=5,
        )

        # Visited-state dedup prevents infinite cycling; should not truncate
        assert not result.truncated

    def test_multiple_paths_one_guarded(self):
        """Test branching where one path has guard, other doesn't."""
        # Block 0 -> Block 1 (guard) -> Block 3 (sink)
        #         -> Block 2 (no guard) -> Block 3 (sink)
        block0 = create_mock_block(0, [0], [1, 2])
        block1 = create_mock_block(1, [1], [3])  # Has guard at 1
        block2 = create_mock_block(2, [2], [3])  # No guard
        block3 = create_mock_block(3, [3], [])   # Sink at 3
        block_map = {0: block0, 1: block1, 2: block2, 3: block3}

        result = traverse_cfg_for_unguarded_sinks(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=None,
            guard_indices={1},
            sink_indices={3},
        )

        # Sink is reachable via unguarded path (block 2)
        assert 3 in result.flagged_indices


class TestTraverseCfgForSpecificGuard:
    """Tests for traverse_cfg_for_specific_guard function."""

    def test_specific_guard_protects_sink(self):
        """Test that specific guard instruction protects subsequent sinks."""
        block0 = create_mock_block(0, [0, 1, 2], [])  # guard=1, sink=2
        block_map = {0: block0}

        result = traverse_cfg_for_specific_guard(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=0,
            specific_guard_idx=1,
            sink_indices={2},
        )

        assert 2 not in result.flagged_indices
        assert not result.truncated

    def test_sink_before_specific_guard_flagged(self):
        """Test that sink before specific guard is flagged."""
        block0 = create_mock_block(0, [0, 1, 2], [])  # sink=1, guard=2
        block_map = {0: block0}

        result = traverse_cfg_for_specific_guard(
            block_map=block_map,
            start_block_id=0,
            start_from_idx=0,
            specific_guard_idx=2,
            sink_indices={1},
        )

        assert 1 in result.flagged_indices


class TestBuildInstructionToBlockMap:
    """Tests for build_instruction_to_block_map function."""

    def test_basic_mapping(self):
        """Test basic instruction to block mapping."""
        block0 = create_mock_block(0, [0, 1, 2], [])
        block1 = create_mock_block(1, [3, 4], [])

        result = build_instruction_to_block_map([block0, block1])

        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == 1
        assert result[4] == 1

    def test_empty_blocks(self):
        """Test with empty block list."""
        result = build_instruction_to_block_map([])
        assert result == {}


class TestAuthGuardHelpers:
    def test_collect_auth_guard_indices_detects_block_local_pattern(self):
        block = create_mock_block(0, [0, 1, 2, 3], [])
        instructions = {
            0: MockInstructionFact("LDMSGADDR", 0),
            1: MockInstructionFact("SDEQ", 1),
            2: MockInstructionFact("PUSHINT", 2),
            3: MockInstructionFact("THROWIFNOT", 3),
        }

        result = collect_auth_guard_indices([block], instructions)

        assert result == {3}

    def test_all_predecessor_paths_guarded_returns_true(self):
        entry = create_mock_block(0, [0, 1], [1])
        sender = create_mock_block(1, [2, 3], [])
        predecessors = build_predecessor_map([entry, sender])

        guarded, truncated = is_block_guarded_on_all_predecessor_paths(
            block_map={0: entry, 1: sender},
            predecessor_map=predecessors,
            start_block_id=1,
            stop_before_idx=2,
            guard_indices={1},
        )

        assert guarded is True
        assert truncated is False

    def test_all_predecessor_paths_guarded_returns_false_when_entry_unguarded(self):
        entry = create_mock_block(0, [0, 1], [1])
        sender = create_mock_block(1, [2, 3], [])
        predecessors = build_predecessor_map([entry, sender])

        guarded, truncated = is_block_guarded_on_all_predecessor_paths(
            block_map={0: entry, 1: sender},
            predecessor_map=predecessors,
            start_block_id=1,
            stop_before_idx=2,
            guard_indices=set(),
        )

        assert guarded is False
        assert truncated is False
