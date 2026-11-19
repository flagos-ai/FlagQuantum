"""Full native comparison and signed-negation Notebook workflows."""

import importlib
import json
from collections.abc import Callable
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
    "workflow,stage,case,registers,cells",
    [
        (
            "comparator",
            "equal_words",
            "comparator_2vars_example",
            ("a", "b", "res"),
            5,
        ),
        ("comparator", "threshold", "comparator_integer_example", ("a", "res"), 5),
        ("negation", "negate", "negation_example", ("a", "b"), 3),
    ],
)
def test_all_notebook_cells_and_independent_complete_state(
    workflow: str,
    stage: str,
    case: str,
    registers: tuple[str, ...],
    cells: int,
) -> None:
    directory = ROOT / f"examples/synthesis/functions/arithmetic/{workflow}/overview"
    namespace, executed = execute_code_cells(directory / "walkthrough.ipynb")
    assert executed == cells
    program = namespace[f"{stage}_program"]
    assert isinstance(program, synth.Program)
    reference = importlib.import_module(
        f"examples.synthesis.functions.arithmetic.{workflow}.{case}.comparisons.circuit"
    )
    build_reference = cast(Callable[[], fq.Circuit], reference.reference_circuit)
    compare_states(program, build_reference(), registers)
    probabilities = cast(
        dict[tuple[int, ...], float], namespace[f"{stage}_probabilities"]
    )
    counts = cast(dict[tuple[int, ...], int], namespace[f"{stage}_counts"])
    assert sum(counts.values()) == 1000 and set(counts) == set(probabilities)
    if stage == "equal_words":
        assert probabilities == {(0, 0, 1): 1.0}
    elif stage == "threshold":
        assert probabilities == pytest.approx(
            {(value, int(value <= 2)): 1 / 8 for value in range(8)}, abs=1e-12
        )
    else:
        assert probabilities == pytest.approx(
            {(value, -value): 1 / 8 for value in range(-4, 4)}, abs=1e-12
        )
        assert tuple(map(len, program.registers.values())) == (3, 4)
        assert (-4, 4) in probabilities and (3, -3) in probabilities


@pytest.mark.parametrize("workflow,stages", [("comparator", 2), ("negation", 1)])
def test_python_workflow_launcher_executes_every_stage(
    workflow: str, stages: int, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module(
        f"examples.synthesis.functions.arithmetic.{workflow}.overview.main"
    )
    run = cast(Callable[[], None], module.run_example)
    run()
    output = capsys.readouterr().out
    assert output.count("Probabilities:") == stages
    assert output.count("Counts:") == stages
    assert output.count("Resources:") == stages
    if workflow == "negation":
        assert "Fraction(-4, 1), Fraction(4, 1)" in output
        assert "Fraction(3, 1), Fraction(-3, 1)" in output


@pytest.mark.parametrize(
    "workflow,source_cells", [("comparator", [2, 3, 5, 6]), ("negation", [2, 3])]
)
def test_source_cell_provenance_and_native_notebook_content(
    workflow: str, source_cells: list[int]
) -> None:
    directory = ROOT / f"examples/synthesis/functions/arithmetic/{workflow}/overview"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert [c["index"] for c in provenance["reviewed_code_cells"]] == source_cells
    assert all(len(c["sha256"]) == 64 for c in provenance["reviewed_code_cells"])
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
