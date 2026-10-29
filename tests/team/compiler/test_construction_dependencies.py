"""Check the narrow compiler profile and the proposed reuse direction."""

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_construction_profile_has_no_execution_or_algorithm_imports() -> None:
    package = importlib.import_module("flagquantum.compiler._construction")
    root = Path(package.__file__).parent
    for source in root.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = " ".join(alias.name for alias in node.names)
            else:
                continue
            parts = set(module.replace(" ", ".").split("."))
            assert not parts.intersection(
                {"runtime", "simulation", "algorithms", "circuit"}
            ), source


def test_existing_algorithm_can_import_the_shared_compiler_boundary() -> None:
    arithmetic = importlib.import_module("flagquantum.algorithms.arithmetic")
    profile = importlib.import_module("flagquantum.compiler._construction.profile")
    ir = arithmetic.adder_circuit(1).to_ir().validate()
    assert profile.uint_type(1).kind == "quantum.uint"
    assert ir.n_wires == arithmetic.adder_qubits(1).n_qubits == 4
    # Production arithmetic remains untouched until the CS-02 kernel extraction.
    assert len(ir.instructions) == 9
