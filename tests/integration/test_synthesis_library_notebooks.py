"""Complete native state preparation, control and evolution Notebook workflows."""

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.linear_pauli_rotations.comparisons.circuit import (
    reference_circuit as linear_reference,
)
from flagquantum import synth
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    ("state_preparation.prepare_exponential_state", ("x",), 3, (2, 3)),
    ("mcx.mcx_example", ("cntrl", "target"), 3, (3, 4)),
    ("hamiltonian_evolution.qdrift", ("qba",), 2, (2,)),
    ("linear_pauli_rotations", ("x", "ind"), 3, (3, 4)),
)


def directory(module: str) -> Path:
    return ROOT / "examples/synthesis/functions" / module.replace(".", "/") / "overview"


def mcx_reference() -> fq.Circuit:
    permutation = torch.eye(256, dtype=torch.complex128)
    permutation[-2:, -2:] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for bit in range(7):
        circuit.h(bit)
    circuit.s(6).h(7).rz(7, theta=0.73)
    circuit.any(*range(8), matrix=permutation)
    return circuit


@pytest.mark.parametrize("module,registers,cells,source_cells", WORKFLOWS)
def test_every_cell_and_original_full_complex_state(
    module: str, registers: tuple[str, ...], cells: int, source_cells: tuple[int, ...]
) -> None:
    namespace, count = execute_code_cells(directory(module) / "walkthrough.ipynb")
    assert count == cells
    program = namespace["program"]
    assert isinstance(program, synth.Program)
    reference = importlib.import_module(
        f"examples.synthesis.functions.{module}.comparisons.circuit"
    )
    build_reference = cast(Callable[[], fq.Circuit], reference.reference_circuit)
    compare_states(program, build_reference(), registers)
    if module == "state_preparation.prepare_exponential_state":
        result = fq.run(program)
        probabilities = dict(program.probabilities(result, registers=("x",)))
        weights = [math.exp(-0.1 * value) for value in range(32)]
        assert probabilities == pytest.approx(
            {
                (value,): weight / math.fsum(weights)
                for value, weight in enumerate(weights)
            },
            abs=1e-12,
        )
        assert len(program.registers["x"]) == 5
    elif module == "linear_pauli_rotations":
        assert namespace["SLOPES"] == (2.1, 1.0, 7.0)
        assert namespace["OFFSETS"] == (0.1, 0.3, 0.33)
        assert tuple(map(len, program.registers.values())) == (6, 3)
    elif module == "mcx.mcx_example":
        assert tuple(map(len, program.registers.values())) == (7, 1)
    else:
        assert len(program.registers["qba"]) == 2


@pytest.mark.parametrize("module", ["mcx.mcx_example", "linear_pauli_rotations"])
def test_coherent_supplement_preserves_absolute_phase_and_cleans_work(
    module: str,
) -> None:
    namespace, _ = execute_code_cells(directory(module) / "walkthrough.ipynb")
    program = namespace["coherent_program"]
    assert isinstance(program, synth.Program)
    if module == "mcx.mcx_example":
        compare_states(program, mcx_reference(), ("cntrl", "target"))
        launcher = importlib.import_module(
            "examples.synthesis.functions.mcx.mcx_example.overview.main"
        )
        compare_states(
            synth.synthesize(launcher.coherent), mcx_reference(), ("cntrl", "target")
        )
    else:
        compare_states(program, linear_reference(coherent=True), ("x", "ind"))


@pytest.mark.parametrize(
    "module,stages",
    [
        (row[0], 2 if row[0] in {"mcx.mcx_example", "linear_pauli_rotations"} else 1)
        for row in WORKFLOWS
    ],
)
def test_python_launcher_executes_all_stages(
    module: str, stages: int, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher = importlib.import_module(
        f"examples.synthesis.functions.{module}.overview.main"
    )
    run = cast(Callable[[], None], launcher.run_example)
    run()
    output = capsys.readouterr().out
    assert output.count("Probabilities:") == stages
    assert output.count("Resources:") == stages


@pytest.mark.parametrize("module,registers,cells,source_cells", WORKFLOWS)
def test_source_cells_and_clean_native_notebook(
    module: str, registers: tuple[str, ...], cells: int, source_cells: tuple[int, ...]
) -> None:
    path = directory(module)
    provenance = json.loads((path / "provenance.json").read_text())
    assert (
        tuple(cell["index"] for cell in provenance["reviewed_code_cells"])
        == source_cells
    )
    assert all(len(cell["sha256"]) == 64 for cell in provenance["reviewed_code_cells"])
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert sum(cell["cell_type"] == "code" for cell in notebook["cells"]) == cells
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        assert "Staged prototype" not in source and "Staged outside" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
