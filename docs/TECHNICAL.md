# TasmScan Technical Guide

This is the single technical reference for the released artifact. The
release contains code and experiment settings only; datasets, labels,
precomputed findings, logs, and benchmark outputs must be supplied externally.

## Analysis pipeline

```text
BOC / Base64 / hex input
        │
        ▼
BOC parser and TVM instruction decoder
        │
        ▼
CFG construction + stack analysis + continuation resolution
        │
        ▼
TASIR lifting + SaveList linking + path-sensitive data flow
        │
        ▼
security and quality detectors
        │
        ▼
text / JSON findings
```

The implementation is organized into `tasmscan/disassembler/`,
`tasmscan/analyzer/`, `tasmscan/ir/`, `tasmscan/solver/`, and
`tasmscan/detectors/`. The scanner exposes both the `tasmscan` CLI and the
`scan_contract` Python API.

## Core components

- The disassembler parses BOC cells and decodes TVM instructions using the
  bundled instruction table and `spec/cp0.json`.
- The analyzer builds basic blocks and CFG edges, models stack effects, and
  resolves continuation targets conservatively when they are dynamic.
- TASIR provides typed instruction semantics, opcode mappings, SaveList-aware
  continuation linking, and path-sensitive taint propagation.
- Detectors report sender and message checks, send/accept ordering, bounced
  messages, randomness, dynamic calls, parsing, message modes, dictionary/TL-B
  issues, destination validation, and related classes.

## Interfaces

```bash
tasmscan CONTRACT.boc
tasmscan CONTRACT.boc --json
tasmscan CONTRACT.boc --config example_config.json
tasmscan CONTRACT.boc --analysis-mode security_audit
tasmscan CONTRACT.boc --path-insensitive
tasmscan --list-detectors
```

```python
from tasmscan import scan_contract

findings = scan_contract("contract.boc", analysis_mode="balanced")
```

The scanner also accepts a Base64 or hexadecimal BOC string. Configuration
supports JSON (and YAML when an external YAML parser is installed), detector
selection, analysis-mode presets, path budgets, TASIR selection, and strict
completeness handling.

## Experiment settings and drivers

- `example_config.json` is a minimal detector configuration.
- `scripts/ablation_rq1.py` and the validation scripts support ablation and
  correctness checks over externally supplied inputs.
- `scripts/benchmark_test.py` supports benchmark smoke tests.
- The evaluation configuration uses `MAX_LOOP_UNROLL=3` and
  `TAINT_STACK_DEPTH=64` (`K_s` in the paper).  The RQ1 ablation stages are
  baseline resolution, intercontinuation propagation, reachability pruning,
  and savelist propagation.
- The scripts under `scripts/` write only to caller-selected or local runtime
  paths; no datasets or generated reports are included here.

## Reproducibility checks

```bash
PYTHONPATH=. python3 -m pytest -q
python3 scripts/validate_release.py
python3 -m tasmscan --list-detectors
```

The artifact intentionally does not claim a precomputed metric or include a
training/evaluation corpus. Any reported result must identify the external
input corpus, detector configuration, and analysis budgets used.

## Known limitations

Dynamic continuation targets and runtime-constructed cells may remain
conservative or unresolved. Path-sensitive and fixpoint analyses are bounded
by configurable resource limits; when a limit is reached, metadata marks the
analysis as incomplete rather than silently treating it as exact.
