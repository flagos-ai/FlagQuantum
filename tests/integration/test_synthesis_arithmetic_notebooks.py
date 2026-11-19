"""Exact integer/fixed-point arithmetic through complete native Notebooks."""

import importlib
import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "workflow,stage,case,registers,expected",
    [
        (
            "multiplication",
            "multiply_words",
            "multiplication_2vars_example",
            ("a", "b", "res"),
            {(4, 5, 20): 1.0},
        ),
        (
            "multiplication",
            "multiply_scaled",
            "multiplication_float_example",
            ("a", "res"),
            {(a, Fraction(7, 2) * a): 1 / 4 for a in range(4)},
        ),
        (
            "subtraction",
            "subtract_words",
            "subtraction_2vars_example",
            ("a", "b", "res"),
            {(4, 5, -1): 1.0},
        ),
        (
            "subtraction",
            "subtract_scaled",
            "subtraction_float_example",
            ("a", "res"),
            {(4, Fraction(1, 2)): 1.0},
        ),
        (
            "extremum",
            "minimum",
            "minimum_2vars_example",
            ("a", "b", "res"),
            {(4, b, min(4, b)): 1 / 8 for b in range(8)},
        ),
        (
            "extremum",
            "maximum",
            "maximum_float_example",
            ("a", "res"),
            {(a, max(Fraction(7, 2), a)): 1 / 8 for a in range(8)},
        ),
    ],
)
def test_complete_notebook_states_and_exact_numeric_readout(
    workflow: str,
    stage: str,
    case: str,
    registers: tuple[str, ...],
    expected: dict[tuple[int | Fraction, ...], float],
) -> None:
    directory = ROOT / f"examples/synthesis/functions/arithmetic/{workflow}/overview"
    namespace, cells = execute_code_cells(directory / "walkthrough.ipynb")
    assert cells == 5
    program = namespace[f"{stage}_program"]
    assert isinstance(program, synth.Program)
    reference = importlib.import_module(
        f"examples.synthesis.functions.arithmetic.{workflow}.{case}.comparisons.circuit"
    )
    build_reference = cast(Callable[[], fq.Circuit], reference.reference_circuit)
    compare_states(program, build_reference(), registers)
    probabilities = cast(
        dict[tuple[int | Fraction, ...], float], namespace[f"{stage}_probabilities"]
    )
    counts = cast(dict[tuple[int | Fraction, ...], int], namespace[f"{stage}_counts"])
    assert probabilities == pytest.approx(expected, abs=1e-12)
    assert sum(counts.values()) == 1000
    assert set(counts) == set(expected)


@pytest.mark.parametrize("workflow", ["multiplication", "subtraction", "extremum"])
def test_launcher_executes_both_stages(
    workflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module(
        f"examples.synthesis.functions.arithmetic.{workflow}.overview.main"
    )
    run = cast(Callable[[], None], module.run_example)
    run()
    output = capsys.readouterr().out
    assert output.count("Probabilities:") == 2
    assert output.count("Counts:") == 2
    assert output.count("Resources:") == 2


@pytest.mark.parametrize("workflow", ["multiplication", "subtraction", "extremum"])
def test_native_notebook_source_and_provenance(workflow: str) -> None:
    directory = ROOT / f"examples/synthesis/functions/arithmetic/{workflow}/overview"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert len(provenance["reviewed_code_cells"]) == 4
    assert all(len(c["sha256"]) == 64 for c in provenance["reviewed_code_cells"])
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
