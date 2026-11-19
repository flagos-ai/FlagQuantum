"""Retain original sampling, discarded environments and both QPE stages."""

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.hadamard_test.comparisons.circuit import (
    reference_circuit as hadamard_reference,
)
from examples.synthesis.algorithms.quantum_primitives.swap_test.comparisons.circuit import (
    reference_circuit as swap_reference,
)
from examples.synthesis.qpe_reference import dense_qpe
from flagquantum import synth
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    ("algorithms/quantum_primitives/hadamard_test", 3, (8, 10, 12, 19, 21)),
    (
        "tutorials/basics/quantum_primitives/hadamard_test_tutorial",
        3,
        (8, 10, 12, 19, 21),
    ),
    ("algorithms/quantum_primitives/swap_test", 2, (2, 4, 7, 9, 10, 11)),
    ("functions/qpe", 5, (3, 4, 6, 8, 9, 11)),
)


def notebook(relative: str) -> Path:
    return ROOT / "examples/synthesis" / relative / "overview/walkthrough.ipynb"


@pytest.mark.parametrize("relative", [row[0] for row in WORKFLOWS[:2]])
def test_hadamard_complete_joint_state_and_mixed_readout(relative: str) -> None:
    namespace, cells = execute_code_cells(notebook(relative))
    assert cells == 3
    program = namespace["program"]
    assert isinstance(program, synth.Program)
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(hadamard_reference()).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert tuple(program.registers) == ("expectation_value",)
    assert actual.numel() == 32 and program.report["workspace_qubits"] == 0
    probabilities = dict(
        program.probabilities(fq.run(program), registers=("expectation_value",))
    )
    assert probabilities == pytest.approx({(0,): 0.625, (1,): 0.375}, abs=1e-12)
    counts = cast(dict[tuple[int, ...], int], namespace["counts"])
    assert sum(counts.values()) == 100000 and set(counts) == {(0,), (1,)}
    estimate = cast(float, namespace["expectation_value"])
    assert estimate == 2 * counts[(0,)] / 100000 - 1
    assert abs(estimate - 0.25) < 0.02
    # Retained hidden environment is deliberately entangled, not clean work.
    joint = actual.reshape(2, 16)
    reduced = joint @ joint.mH
    assert float(torch.trace(reduced @ reduced).real) == pytest.approx(0.53125)


def test_swap_original_seed_data_joint_state_and_overlap() -> None:
    np = pytest.importorskip("numpy")
    namespace, cells = execute_code_cells(notebook(WORKFLOWS[2][0]))
    assert cells == 2
    rng = np.random.RandomState(0)
    first, second = 1 - 2 * rng.rand(8), 1 - 2 * rng.rand(8)
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    np.testing.assert_array_equal(namespace["amps1"], first)
    np.testing.assert_array_equal(namespace["amps2"], second)
    program = namespace["program"]
    assert isinstance(program, synth.Program)
    actual = fq.run(program).statevector().flatten()
    expected = (
        fq.run(swap_reference(tuple(first), tuple(second))).statevector().flatten()
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert actual.numel() == 128 and tuple(program.registers) == ("test",)
    assert program.report["workspace_qubits"] == 0
    exact = float(abs(first @ second))
    probabilities = dict(program.probabilities(fq.run(program), registers=("test",)))
    assert probabilities[(0,)] == pytest.approx((1 + exact**2) / 2, abs=1e-12)
    counts = cast(dict[tuple[int, ...], int], namespace["counts"])
    assert sum(counts.values()) == 100000
    estimate = cast(float, namespace["overlap_from_swap_test"])
    assert estimate == math.sqrt(2 * counts[(0,)] / 100000 - 1)
    assert abs(estimate - exact) < 0.05


def test_qpe_both_original_stages_and_absolute_evolution_phase() -> None:
    namespace, cells = execute_code_cells(notebook(WORKFLOWS[3][0]))
    assert cells == 5
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    controlled_z = torch.diag(torch.tensor([1, -1j, 1, 1j], dtype=torch.complex128))
    energies = torch.tensor([0, 0.5, 0.25, 0.75], dtype=torch.float64)
    evolution = torch.diag(torch.exp(2j * math.pi * energies))
    for stage, matrix, preparation, outcomes in (
        ("first", controlled_z, torch.kron(x, x), {(3, 1): 1.0}),
        (
            "second",
            evolution,
            torch.kron(hadamard, hadamard),
            {(0, 0): 0.25, (1, 2): 0.25, (2, 1): 0.25, (3, 3): 0.25},
        ),
    ):
        program = namespace[stage + "_program"]
        assert isinstance(program, synth.Program)
        result = fq.run(program)
        state = result.statevector().flatten()
        assert state.numel() == 16 and program.report["workspace_qubits"] == 0
        wires = (*program.registers["state"], *program.registers["phase_"])
        actual = state.reshape(2, 2, 2, 2).permute(wires).flatten()
        expected = (
            fq.run(dense_qpe(matrix, preparation, 2))
            .statevector()
            .reshape(4, 4)
            .T.flatten()
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        probabilities = dict(
            program.probabilities(result, registers=("state", "phase_"))
        )
        assert probabilities == pytest.approx(outcomes, abs=1e-12)
        counts = cast(dict[tuple[int, ...], int], namespace[stage + "_counts"])
        assert sum(counts.values()) == 1000 and set(counts) == set(outcomes)


@pytest.mark.parametrize("relative,cells,source_cells", WORKFLOWS)
def test_source_provenance_and_clean_native_notebook(
    relative: str, cells: int, source_cells: tuple[int, ...]
) -> None:
    path = notebook(relative)
    proof = json.loads(path.with_name("provenance.json").read_text())
    assert tuple(cell["index"] for cell in proof["reviewed_code_cells"]) == source_cells
    assert all(len(cell["sha256"]) == 64 for cell in proof["reviewed_code_cells"])
    document = json.loads(path.read_text())
    assert sum(cell["cell_type"] == "code" for cell in document["cells"]) == cells
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        assert "Staged only" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize("relative,cells,source_cells", WORKFLOWS)
def test_python_launcher_runs_original_stages(
    relative: str,
    cells: int,
    source_cells: tuple[int, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if "swap_test" in relative:
        pytest.importorskip("numpy")
    launcher = importlib.import_module(
        "examples.synthesis." + relative.replace("/", ".") + ".overview.main"
    )
    run = cast(Callable[[], None], launcher.run_example)
    run()
    output = capsys.readouterr().out
    if "hadamard_test" in relative:
        assert "Exact Re(<U>): 0.25" in output and "Re(<U>):" in output
    elif "swap_test" in relative:
        assert "Sampled overlap:" in output and "Exact overlap:" in output
    else:
        assert output.count("Fractional phase:") == 2
        assert output.count("1000 seeded local samples:") == 2
