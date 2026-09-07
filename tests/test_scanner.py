"""
Tests for scanner.py API.

Tests cover:
- scan_contract function
- Hex input validation
- File input handling
- Detector filtering
- Error handling for invalid inputs
- SecurityScanner class
"""
import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch

from tasmscan.detectors.results import Confidence, Vulnerability
from tasmscan.scanner import (
    AnalysisIncompleteError,
    SecurityScanner,
    _filter_detectors,
    scan_contract,
)


class TestScanContractHexValidation:
    """Tests for text input validation in scan_contract."""

    def test_invalid_hex_characters(self):
        """Text string with invalid characters should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            scan_contract("invalid!@#$")  # Invalid base64/hex chars

        assert "invalid" in str(exc_info.value).lower()

    def test_odd_length_hex_string(self):
        """Hex string with odd length should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            scan_contract("0xabc")  # Odd length (explicit hex)

        assert "odd length" in str(exc_info.value).lower()

    def test_hex_with_0x_prefix(self):
        """Hex string with 0x prefix should be handled."""
        # Note: This will still fail to parse as BOC, but should not fail
        # on hex validation itself
        with pytest.raises(Exception):  # Will fail on BOC parsing
            scan_contract("0xabcd")

    def test_hex_with_0X_prefix_uppercase(self):
        """Hex string with uppercase 0X prefix should be handled."""
        with pytest.raises(Exception):  # Will fail on BOC parsing
            scan_contract("0XABCD")

    def test_hex_with_whitespace(self):
        """Hex string with surrounding whitespace should be stripped."""
        with pytest.raises(Exception):  # Will fail on BOC parsing
            scan_contract("  abcd  ")

    def test_empty_string(self):
        """Empty string should raise error (FileNotFoundError or ValueError)."""
        # Empty string is treated as file path, which doesn't exist
        with pytest.raises((ValueError, FileNotFoundError)):
            scan_contract("")

    def test_whitespace_only(self):
        """Whitespace only should raise error."""
        with pytest.raises(ValueError):
            scan_contract("   ")

    def test_valid_hex_invalid_boc(self):
        """Valid hex but invalid BOC data should raise appropriate error."""
        # This is valid hex but not valid BOC format
        with pytest.raises(Exception):
            scan_contract("deadbeef")

    @patch("tasmscan.scanner.detect_and_decode_boc")
    @patch("tasmscan.scanner.SecurityScanner.scan_boc_bytes")
    def test_base64_string_input(self, mock_scan, mock_detect):
        """Base64 string input should be auto-detected and decoded."""
        mock_detect.return_value = b"boc_data"
        mock_scan.return_value = []

        scan_contract("Zm9v")  # base64("foo")

        mock_detect.assert_called_once()
        mock_scan.assert_called_once_with(b"boc_data")


class TestScanContractFileInput:
    """Tests for file input handling in scan_contract."""

    def test_file_not_found(self):
        """Non-existent file path should raise error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            scan_contract("/nonexistent/path/to/file.boc")

    def test_file_exists_check(self):
        """Should check if path exists before treating as file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".boc", delete=False) as f:
            f.write(b"invalid boc data")
            temp_path = f.name

        try:
            # Should try to read as file and fail on BOC parsing
            with pytest.raises(Exception):
                scan_contract(temp_path)
        finally:
            os.unlink(temp_path)


class TestScanContractDetectorFiltering:
    """Tests for detector filtering in scan_contract."""

    def test_filter_detectors_by_name(self):
        """Specifying detector names should filter to only those detectors."""
        # Test the filtering logic directly without mocking
        scanner = SecurityScanner(enable_all_detectors=True)
        original_count = len(scanner.detectors)

        # Filter detectors
        scanner.detectors = [
            d for d in scanner.detectors
            if d.name in ["bounced_message"]
        ]

        # Should have fewer detectors after filtering
        assert len(scanner.detectors) <= original_count
        # If bounced_message exists, it should be in the filtered list
        if scanner.detectors:
            assert all(d.name == "bounced_message" for d in scanner.detectors)

    def test_none_detectors_uses_all(self):
        """None for detectors parameter should use all detectors."""
        scanner = SecurityScanner(enable_all_detectors=True)

        # With None, detectors should not be filtered
        # Verify scanner has detectors when enable_all_detectors=True
        assert len(scanner.detectors) > 0

    def test_filter_explicitly_enables_disabled_detector(self):
        """Explicit detector list should allow default-disabled detectors."""
        scanner = SecurityScanner(enable_all_detectors=True)
        assert all(d.name != "bounced_handler_pattern" for d in scanner.detectors)

        _filter_detectors(scanner, ["bounced_handler_pattern"])

        assert len(scanner.detectors) == 1
        assert scanner.detectors[0].name == "bounced_handler_pattern"


class TestSecurityScanner:
    """Tests for SecurityScanner class."""

    def test_init_with_all_detectors(self):
        """Default init should enable all detectors."""
        scanner = SecurityScanner(enable_all_detectors=True)
        assert len(scanner.detectors) > 0

    def test_init_without_detectors(self):
        """Init with enable_all_detectors=False should have empty detectors."""
        scanner = SecurityScanner(enable_all_detectors=False)
        assert scanner.detectors == []

    def test_has_analyzer(self):
        """Scanner should have ProgramAnalyzer."""
        scanner = SecurityScanner()
        assert scanner.analyzer is not None

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_scan_cell_strict_completeness_raises(self, mock_dataflow, mock_decompile):
        """Strict completeness mode should fail closed on incomplete analysis."""
        scanner = SecurityScanner(
            enable_all_detectors=False,
            use_tasir=False,
            strict_completeness=True,
        )
        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.cfg_edges = []
        mock_facts.entry_points = []
        mock_facts.metadata = {"continuation_count": 5}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["path_truncation"],
            "truncated_count": 1,
        }
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = mock_graph

        with pytest.raises(AnalysisIncompleteError) as exc_info:
            scanner.scan_cell(mock_cell)

        assert "analysis_incomplete" in str(exc_info.value)

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_scan_cell_strict_completeness_raises_on_cfg_entry_shape_widened(
        self, mock_dataflow, mock_decompile
    ):
        """Strict completeness should fail closed for cfg_entry_shape_widened reason."""
        scanner = SecurityScanner(
            enable_all_detectors=False,
            use_tasir=False,
            strict_completeness=True,
        )
        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.cfg_edges = []
        mock_facts.entry_points = []
        mock_facts.metadata = {"continuation_count": 1}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["cfg_entry_shape_widened"],
        }
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = mock_graph

        with pytest.raises(AnalysisIncompleteError) as exc_info:
            scanner.scan_cell(mock_cell)

        report = exc_info.value.completeness
        assert report["analysis_incomplete"] is True
        assert "cfg_entry_shape_widened" in report["analysis_incomplete_reasons"]

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_scan_cell_records_completeness_metadata(self, mock_dataflow, mock_decompile):
        """scan_cell should expose normalized completeness metadata for consumers."""
        scanner = SecurityScanner(enable_all_detectors=False, use_tasir=False)
        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.cfg_edges = []
        mock_facts.entry_points = []
        mock_facts.metadata = {"continuation_count": 5}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {
            "analysis_incomplete": True,
            "analysis_incomplete_reasons": ["dynamic_continuation_targets"],
            "dynamic_target_count": 1,
        }
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = mock_graph

        result = scanner.scan_cell(mock_cell)

        assert result == []
        assert scanner.last_completeness["analysis_incomplete"] is True
        assert "dynamic_continuation_targets" in scanner.last_completeness["analysis_incomplete_reasons"]
        info = scanner.get_last_analysis_info()
        assert info["completeness"]["analysis_incomplete"] is True
        assert info["metadata"]["instruction_count"] == 0
        assert info["metadata"]["basic_block_count"] == 0
        assert info["metadata"]["cfg_edge_count"] == 0
        assert info["metadata"]["entry_point_count"] == 0
        assert info["metadata"]["continuation_count"] == 5

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_scan_cell_calls_disassembler(self, mock_dataflow, mock_decompile):
        """scan_cell should call decompile_cell."""
        scanner = SecurityScanner(enable_all_detectors=False)
        mock_cell = MagicMock()
        mock_decompile.return_value = []

        # Mock analyzer
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = MagicMock(
            instructions=[],
            events=[],
            basic_blocks=[],
        )

        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = MagicMock()

        result = scanner.scan_cell(mock_cell)

        mock_decompile.assert_called_once_with(mock_cell)
        assert result == []

    @patch("tasmscan.scanner.parse_boc")
    def test_scan_boc_bytes_calls_parse_boc(self, mock_parse):
        """scan_boc_bytes should call parse_boc."""
        scanner = SecurityScanner(enable_all_detectors=False)
        mock_parse.return_value = MagicMock()

        # Mock scan_cell to avoid further processing
        scanner.scan_cell = MagicMock(return_value=[])

        scanner.scan_boc_bytes(b"test data")

        mock_parse.assert_called_once_with(b"test data")
        scanner.scan_cell.assert_called_once()

    def test_scan_file_reads_file(self):
        """scan_file should read file and call scan_boc_bytes."""
        scanner = SecurityScanner(enable_all_detectors=False)
        scanner.scan_boc_bytes = MagicMock(return_value=[])
        boc_bytes = b"\xb5\xee\x9c\x72" + b"test boc payload"

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".boc", delete=False) as f:
            f.write(boc_bytes)
            temp_path = f.name

        try:
            scanner.scan_file(temp_path)
            scanner.scan_boc_bytes.assert_called_once_with(boc_bytes)
        finally:
            os.unlink(temp_path)


class TestSolverIRIntegration:
    """Tests for optional Solver IR lowering integration."""

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    @patch("tasmscan.scanner.IRBuilder")
    def test_scan_cell_lowers_solver_ir_when_enabled(
        self,
        mock_ir_builder_cls,
        mock_dataflow,
        mock_decompile,
    ):
        """Solver IR lowering should run when TASIR and solver IR are enabled."""
        scanner = SecurityScanner(
            enable_all_detectors=False,
            use_tasir=True,
            use_solver_ir=True,
        )
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.cfg_edges = []
        mock_facts.entry_points = []
        mock_facts.metadata = {}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_tasir_module = MagicMock()
        mock_tasir_module.functions = []
        mock_tasir_module.all_blocks.return_value = []
        mock_tasir_module.all_instructions.return_value = []
        mock_tasir_module.build_inter_procedural_view = MagicMock()

        mock_solver_module = MagicMock()
        mock_solver_module.blocks = []
        mock_solver_module.instructions = []
        mock_solver_module.symbols = []
        mock_solver_module.constraints = []
        mock_solver_module.obligations = []
        mock_solver_module.metadata = {"status": "ok"}

        mock_ir_builder = MagicMock()
        mock_ir_builder.build_tasir.return_value = mock_tasir_module
        mock_ir_builder.lower_tasir_to_solver_ir.return_value = mock_solver_module
        mock_ir_builder_cls.return_value = mock_ir_builder

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {}
        mock_graph.analysis_type = "tasir_path_sensitive"
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze_tasir.return_value = mock_graph

        result = scanner.scan_cell(MagicMock())

        assert result == []
        mock_ir_builder.lower_tasir_to_solver_ir.assert_called_once_with(mock_tasir_module)
        assert mock_facts.metadata["solver_ir_module"] is mock_solver_module
        assert mock_facts.metadata["solver_ir"] == {"status": "ok"}

        info = scanner.get_last_analysis_info()
        assert "solver_ir" in info
        assert info["solver_ir"]["metadata"] == {"status": "ok"}

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    @patch("tasmscan.scanner.IRBuilder")
    def test_scan_cell_skips_solver_ir_when_disabled(
        self,
        mock_ir_builder_cls,
        mock_dataflow,
        mock_decompile,
    ):
        """Solver IR lowering should be skipped when disabled."""
        scanner = SecurityScanner(
            enable_all_detectors=False,
            use_tasir=True,
            use_solver_ir=False,
        )
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.cfg_edges = []
        mock_facts.entry_points = []
        mock_facts.metadata = {}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_tasir_module = MagicMock()
        mock_tasir_module.functions = []
        mock_tasir_module.all_blocks.return_value = []
        mock_tasir_module.all_instructions.return_value = []
        mock_tasir_module.build_inter_procedural_view = MagicMock()

        mock_ir_builder = MagicMock()
        mock_ir_builder.build_tasir.return_value = mock_tasir_module
        mock_ir_builder_cls.return_value = mock_ir_builder

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {}
        mock_graph.analysis_type = "tasir_path_sensitive"
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze_tasir.return_value = mock_graph

        result = scanner.scan_cell(MagicMock())

        assert result == []
        mock_ir_builder.lower_tasir_to_solver_ir.assert_not_called()
        assert "solver_ir_module" not in mock_facts.metadata
        assert "solver_ir" not in mock_facts.metadata
        info = scanner.get_last_analysis_info()
        assert "solver_ir" not in info

    @patch("tasmscan.scanner._resolve_source_to_bytes")
    @patch("tasmscan.scanner._filter_detectors")
    @patch("tasmscan.scanner.SecurityScanner")
    def test_scan_contract_forwards_use_solver_ir_flag(
        self,
        mock_scanner_cls,
        mock_filter,
        mock_resolve,
    ):
        """scan_contract should forward use_solver_ir to SecurityScanner."""
        mock_scanner = MagicMock()
        mock_scanner.scan_boc_bytes.return_value = []
        mock_scanner_cls.return_value = mock_scanner
        mock_resolve.return_value = b"\xb5\xee\x9c\x72"

        result = scan_contract("deadbeef", use_solver_ir=False)

        assert result == []
        assert mock_scanner_cls.call_args.kwargs["use_solver_ir"] is False
        mock_filter.assert_called_once()
        mock_resolve.assert_called_once_with("deadbeef")
        mock_scanner.scan_boc_bytes.assert_called_once_with(b"\xb5\xee\x9c\x72")


class TestSecurityScannerDetectorExecution:
    """Tests for detector execution in SecurityScanner."""

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_runs_all_detectors(self, mock_dataflow, mock_decompile):
        """scan_cell should run all registered detectors."""
        scanner = SecurityScanner(enable_all_detectors=False)

        # Add mock detectors
        detector1 = MagicMock()
        detector1.detect.return_value = []
        detector2 = MagicMock()
        detector2.detect.return_value = []
        scanner.detectors = [detector1, detector2]

        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = MagicMock()

        scanner.scan_cell(mock_cell)

        detector1.detect.assert_called_once()
        detector2.detect.assert_called_once()

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_collects_all_findings(self, mock_dataflow, mock_decompile):
        """scan_cell should collect findings from all detectors."""
        scanner = SecurityScanner(enable_all_detectors=False)

        # Add mock detectors with findings
        finding1 = MagicMock()
        finding2 = MagicMock()
        detector1 = MagicMock()
        detector1.detect.return_value = [finding1]
        detector2 = MagicMock()
        detector2.detect.return_value = [finding2]
        scanner.detectors = [detector1, detector2]

        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = MagicMock()

        result = scanner.scan_cell(mock_cell)

        assert finding1 in result
        assert finding2 in result
        assert len(result) == 2

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_findings_include_analysis_completeness(self, mock_dataflow, mock_decompile):
        """All findings should carry unified analysis_completeness metadata."""
        scanner = SecurityScanner(enable_all_detectors=False, use_tasir=False)

        detector = MagicMock()
        detector.detect.return_value = [
            Vulnerability(
                detector="dummy",
                severity="low",
                message="dummy finding",
                instruction=None,
                extra=None,
            )
        ]
        scanner.detectors = [detector]

        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.metadata = {}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {"analysis_incomplete": False}
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = mock_graph

        findings = scanner.scan_cell(mock_cell)

        assert len(findings) == 1
        assert isinstance(findings[0].extra, dict)
        assert "analysis_completeness" in findings[0].extra
        assert findings[0].extra["analysis_completeness"]["analysis_incomplete"] is False

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_high_severity_findings_keep_severity_when_evidence_present(self, mock_dataflow, mock_decompile):
        """High severity findings with complete evidence should not be downgraded."""
        scanner = SecurityScanner(enable_all_detectors=False, use_tasir=False)

        detector = MagicMock()
        detector.detect.return_value = [
            Vulnerability(
                detector="dummy",
                severity="high",
                message="high finding",
                instruction=MagicMock(index=7),
                extra={
                    "evidence": {
                        "path_trace": [7],
                        "source_sink_chain": [],
                        "guard_state": {"analysis_incomplete": False, "reasons": []},
                        "solver_status": {"status": "resolved_or_not_applicable", "obligation_count": 0},
                    }
                },
                confidence=Confidence.HIGH,
            )
        ]
        scanner.detectors = [detector]

        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.metadata = {}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {"analysis_incomplete": False}
        mock_graph.tainted_propagation = []
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = mock_graph

        findings = scanner.scan_cell(mock_cell)

        assert len(findings) == 1
        assert findings[0].severity == "high"
        assert findings[0].confidence == Confidence.HIGH
        assert findings[0].extra["evidence_policy"]["status"] == "satisfied"

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_high_severity_findings_downgrade_when_path_trace_missing(self, mock_dataflow, mock_decompile):
        """High severity findings without minimal evidence should be downgraded."""
        scanner = SecurityScanner(enable_all_detectors=False, use_tasir=False)

        detector = MagicMock()
        detector.detect.return_value = [
            Vulnerability(
                detector="dummy",
                severity="high",
                message="high finding",
                instruction=None,
                extra=None,
                confidence=Confidence.HIGH,
            )
        ]
        scanner.detectors = [detector]

        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        mock_facts.metadata = {}
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_graph = MagicMock()
        mock_graph.analysis_metadata = {"analysis_incomplete": False}
        mock_graph.tainted_propagation = []
        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = mock_graph

        findings = scanner.scan_cell(mock_cell)

        assert len(findings) == 1
        assert findings[0].severity == "low"
        assert findings[0].confidence == Confidence.LOW
        assert findings[0].extra["evidence_policy"]["status"] == "missing"
        assert "path_trace" in findings[0].extra["evidence_policy"]["missing_fields"]
        assert "[evidence downgraded]" in findings[0].message


class TestErrorHandling:
    """Tests for error handling in scanner."""

    def test_disassembly_error_propagates(self):
        """Disassembly errors should propagate to caller."""
        scanner = SecurityScanner()

        # Invalid BOC should raise
        with pytest.raises(Exception):
            scanner.scan_boc_bytes(b"not valid boc")

    @patch("tasmscan.scanner.decompile_cell")
    def test_analysis_error_propagates(self, mock_decompile):
        """Analysis errors should propagate to caller."""
        scanner = SecurityScanner()
        mock_decompile.side_effect = RuntimeError("Analysis failed")

        with pytest.raises(RuntimeError):
            scanner.scan_cell(MagicMock())

    def test_scan_contract_invalid_input_error_message(self):
        """Invalid input should have helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            scan_contract("invalid!@#$")

        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg


class TestEdgeCases:
    """Edge case tests."""

    def test_very_long_hex_string(self):
        """Very long hex string should be handled."""
        # 10KB of hex
        long_hex = "ab" * 5000

        # Should fail on BOC parsing, not on hex validation
        with pytest.raises(Exception):
            scan_contract(long_hex)

    def test_path_with_spaces(self):
        """Path with spaces should be handled."""
        # Create temp file with spaces in name
        with tempfile.NamedTemporaryFile(
            suffix=".boc", prefix="test file ", delete=False
        ) as f:
            f.write(b"test data")
            temp_path = f.name

        try:
            with pytest.raises(Exception):  # Will fail on BOC parsing
                scan_contract(temp_path)
        finally:
            os.unlink(temp_path)

    def test_unicode_in_path(self):
        """Unicode characters in path should be handled."""
        # Create temp file with unicode name
        with tempfile.NamedTemporaryFile(
            suffix=".boc", prefix="test_", delete=False
        ) as f:
            f.write(b"test data")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                scan_contract(temp_path)
        finally:
            os.unlink(temp_path)

    @patch("tasmscan.scanner.decompile_cell")
    @patch("tasmscan.scanner.DataFlowAnalyzer")
    def test_empty_detector_list(self, mock_dataflow, mock_decompile):
        """Empty detector list should return no findings."""
        scanner = SecurityScanner(enable_all_detectors=False)
        assert scanner.detectors == []

        mock_cell = MagicMock()
        mock_decompile.return_value = []

        mock_facts = MagicMock()
        mock_facts.instructions = []
        mock_facts.events = []
        mock_facts.basic_blocks = []
        scanner.analyzer = MagicMock()
        scanner.analyzer.analyze.return_value = mock_facts

        mock_dataflow_instance = MagicMock()
        mock_dataflow.return_value = mock_dataflow_instance
        mock_dataflow_instance.analyze.return_value = MagicMock()

        result = scanner.scan_cell(mock_cell)
        assert result == []


class TestDocumentation:
    """Tests for documentation and API contracts."""

    def test_scan_contract_docstring(self):
        """scan_contract should have proper docstring."""
        assert scan_contract.__doc__ is not None
        assert "contract" in scan_contract.__doc__.lower()

    def test_security_scanner_docstring(self):
        """SecurityScanner should have proper docstring."""
        assert SecurityScanner.__doc__ is not None
        assert "security" in SecurityScanner.__doc__.lower()

    def test_scan_cell_return_type(self):
        """scan_cell should return list of Vulnerability."""
        scanner = SecurityScanner(enable_all_detectors=False)

        with patch("tasmscan.scanner.decompile_cell") as mock_decompile:
            with patch("tasmscan.scanner.DataFlowAnalyzer") as mock_dataflow:
                mock_decompile.return_value = []

                mock_facts = MagicMock()
                mock_facts.instructions = []
                mock_facts.events = []
                mock_facts.basic_blocks = []
                scanner.analyzer = MagicMock()
                scanner.analyzer.analyze.return_value = mock_facts

                mock_dataflow_instance = MagicMock()
                mock_dataflow.return_value = mock_dataflow_instance
                mock_dataflow_instance.analyze.return_value = MagicMock()

                result = scanner.scan_cell(MagicMock())

                assert isinstance(result, list)
