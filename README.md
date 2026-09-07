# TasmScan Artifact

TasmScan is a standalone TVM bytecode security scanner for TON smart
contracts. This release contains the implementation, unit tests, example
configuration, and evaluation drivers.

## Layout

- `tasmscan/`: scanner implementation (BOC decoding, CFG/continuation
  analysis, TASIR, data-flow analysis, and detectors)
- `tests/`: unit and regression tests
- `scripts/`: evaluation and validation utilities; inputs and outputs are
  supplied externally
- `example_config.json`: detector configuration example
- `docs/TECHNICAL.md`: algorithm, interface, and experiment-setting reference

## Installation and use

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
tasmscan --list-detectors
tasmscan <CONTRACT.boc> --json
```

The CLI accepts BOC files and Base64/hex BOC text. Results are printed to the
terminal or emitted as JSON.

Run the test suite with:

```bash
PYTHONPATH=. python3 -m pytest -q
```

## Citation

```bibtex
@inproceedings{Liu2026TCA,
  author = {Liu, Yixuan and Wu, Yin and Li, Yi},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  month = oct,
  title = {{TasmScan}: Continuation-Aware Taint Analysis for {TVM} Bytecode with Savelist Abstraction},
  year = {2026}
}
```
