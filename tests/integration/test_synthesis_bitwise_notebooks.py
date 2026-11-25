"""Execute both native Notebook stages and verify independent complete states."""

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


@pytest.mark.parametrize("operation,answer", [("or", 5), ("xor", 1)])
def test_notebook_cells_preserve_both_original_examples(
    operation: str, answer: int, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = (
        ROOT / f"examples/synthesis/functions/arithmetic/bitwise_{operation}/overview"
    )
    namespace, executed = execute_code_cells(directory / "walkthrough.ipynb")
    assert executed == 5
    assert namespace["register_probabilities"] == {(4, 5, answer): 1.0}
    assert namespace["literal_probabilities"] == {(4, 7): 1.0}
    assert namespace["register_samples"] == {(4, 5, answer): 1000}
    assert namespace["literal_samples"] == {(4, 7): 1000}
    for stage, case, registers in (
        ("register", "2vars", ("a", "b", "res")),
        ("literal", "integer", ("a", "res")),
    ):
        reference = importlib.import_module(
            f"examples.synthesis.functions.arithmetic.bitwise_{operation}."
            f"bitwise_{operation}_{case}_example.comparisons.circuit"
        )
        circuit = cast(Callable[[], fq.Circuit], reference.reference_circuit)
        program = namespace[f"{stage}_program"]
        assert isinstance(program, synth.Program)
        compare_states(program, circuit(), registers)
    output = capsys.readouterr().out
    assert output.count("Probabilities:") == 2
    assert output.count("Counts:") == 2


@pytest.mark.parametrize("operation,answer", [("or", 5), ("xor", 1)])
def test_python_launcher_keeps_source_example_order_and_sampling(
    operation: str, answer: int, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow = importlib.import_module(
        f"examples.synthesis.functions.arithmetic.bitwise_{operation}.overview.main"
    )
    run = cast(Callable[[], None], workflow.run_example)
    run()
    lines = capsys.readouterr().out.splitlines()
    signed = (Fraction(4), Fraction(5), Fraction(answer))
    assert [line for line in lines if line.startswith("Probabilities:")] == [
        f"Probabilities: { {signed: 1.0} }",
        "Probabilities: {(4, 7): 1.0}",
    ]
    assert [line for line in lines if line.startswith("Counts:")] == [
        f"Counts: { {signed: 1000} }",
        "Counts: {(4, 7): 1000}",
    ]


@pytest.mark.parametrize("operation", ["or", "xor"])
def test_workflow_provenance_covers_every_original_code_cell(operation: str) -> None:
    directory = (
        ROOT / f"examples/synthesis/functions/arithmetic/bitwise_{operation}/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [cell["index"] for cell in proof["reviewed_code_cells"]] == [2, 3, 5, 6]
    assert all(len(cell["sha256"]) == 64 for cell in proof["reviewed_code_cells"])
    assert proof["native_notebook"] == str(
        (directory / "walkthrough.ipynb").relative_to(ROOT)
    )
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source
        assert "source_format" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
