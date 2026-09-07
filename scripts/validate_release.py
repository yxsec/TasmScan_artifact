"""Validate the result-free TasmScan artifact before release."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = {
    "README.md",
    "docs/TECHNICAL.md",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "setup.py",
    "example_config.json",
}
FORBIDDEN_NAMES = {
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "benchmark_results",
    "evaluation",
    "results",
    "result",
    "dataset",
    "datasets",
    "tmp",
}
FORBIDDEN_SUFFIXES = {".pyc", ".log", ".jsonl", ".db", ".sqlite", ".sqlite3"}


def main() -> None:
    missing = [name for name in sorted(REQUIRED_FILES) if not (ROOT / name).is_file()]
    assert not missing, f"missing required files: {missing}"
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        assert not FORBIDDEN_NAMES.intersection(relative.parts), relative
        if path.is_file():
            assert path.suffix not in FORBIDDEN_SUFFIXES, relative
    print("TasmScan release validation passed")


if __name__ == "__main__":
    main()
