from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
USER_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "API.md",
    ROOT / "docs" / "ECOSYSTEM_DEVELOPMENT.md",
    ROOT / "docs" / "FQ_MODULE.md",
    ROOT / "docs" / "HYBRID_RUNTIME_ARCHITECTURE.md",
    ROOT / "examples" / "README.md",
    ROOT / "examples" / "single_machine_quantum_ai" / "README.md",
    ROOT / "examples" / "mps_hamiltonian_identification" / "README.md",
    ROOT / "examples" / "tutorials" / "README.md",
)
LEGACY_CODE_PATTERNS = {
    "legacy quantum device": re.compile(r"\b(?:Distributed)?QuantumDevice\b"),
    "legacy quantum layer": re.compile(r"\bQuantumTorchLayer\b"),
    "legacy encoder": re.compile(r"\bGeneralEncoder\b"),
    "legacy measurement helper": re.compile(r"\bmeasure_allZ\b"),
    "legacy module gate": re.compile(r"\bfq\.(?:CX|RY)\b"),
    "internal runtime import": re.compile(r"\bfrom\s+flagquantum\.runtime"),
}
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _code_blocks(path: Path) -> tuple[str, ...]:
    return tuple(FENCE.findall(path.read_text(encoding="utf-8")))


@pytest.mark.unit
def test_user_markdown_code_uses_current_public_api() -> None:
    violations: list[str] = []
    for path in USER_DOCUMENTS:
        assert path.is_file()
        for index, code in enumerate(_code_blocks(path), start=1):
            for label, pattern in LEGACY_CODE_PATTERNS.items():
                if pattern.search(code):
                    violations.append(
                        f"{path.relative_to(ROOT)} code block {index}: {label}"
                    )
    assert not violations, "\n".join(violations)


@pytest.mark.unit
def test_markdown_example_commands_use_n_qubits() -> None:
    legacy_command = re.compile(r"python(?:\s+-\S+)*\s+examples/\S+\.py[^\n]*--n-wires")
    violations = []
    for path in sorted(ROOT.rglob("*.md")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if legacy_command.search(line):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.unit
def test_markdown_does_not_reference_removed_scratch_notebook() -> None:
    violations = []
    for path in sorted(ROOT.rglob("*.md")):
        if "tutorials/tt.ipynb" in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, "\n".join(violations)
