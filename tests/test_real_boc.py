import glob

import pytest

pytest.importorskip("pytoniq_core")

from tasmscan.analyzer.program_analyzer import ProgramAnalyzer
from tasmscan.detectors.no_accept import NoAcceptBeforeSendDetector
from tasmscan.detectors.unchecked_sender import UncheckedSenderDetector
from tasmscan.disassembler import TvmDisassembler


@pytest.mark.integration
def test_real_boc_samples():
    boc_files = glob.glob("boc_out/*.code.boc")
    if not boc_files:
        pytest.skip("No BOC files found in boc_out/")

    disasm = TvmDisassembler()
    analyzer = ProgramAnalyzer()

    for boc_path in boc_files[:3]:
        result = disasm.disassemble(boc_path)
        facts = analyzer.analyze(result.instructions, result.root_cell)

        assert facts.instructions

        detector1 = NoAcceptBeforeSendDetector()
        detector2 = UncheckedSenderDetector()

        _ = detector1.detect(facts)
        _ = detector2.detect(facts)
