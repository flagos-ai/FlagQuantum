"""Accept both coherent LCU constructions and every Hamiltonian sweep point."""

import importlib
import json
import linecache
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.overview import (
    main as evolution,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.pauli_evolution import reference_circuit, reference_sequence
from examples.synthesis.tutorials.basics.quantum_primitives.linear_combination_of_unitaries.comparisons.circuit import (
    reference_circuit as lcu_reference,
)
from flagquantum import synth
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    (
        "tutorials/basics/quantum_primitives/linear_combination_of_unitaries",
        3,
        (7, 11, 13, 15, 18),
    ),
    (
        "algorithms/hamiltonian_simulation/hamiltonian_simulation_guide",
        5,
        (12, 14, 21, 33, 35, 36, 37, 39),
    ),
)
TERMS = (
    (0.3, ((0, "Z"), (1, "Z"))),
    (0.7, ((0, "X"),)),
    (0.2, ((1, "X"),)),
)


def notebook(relative: str) -> Path:
    return ROOT / "examples/synthesis" / relative / "overview/walkthrough.ipynb"


def independent_exponential(time: float) -> torch.Tensor:
    """Use basis transitions and scaled Taylor, independently of spectral loading."""
    hamiltonian = torch.zeros((4, 4), dtype=torch.complex128)
    for word in range(4):
        hamiltonian[word, word] = 0.3 * (1 if (word & 1) == ((word >> 1) & 1) else -1)
        hamiltonian[word ^ 1, word] = 0.7
        hamiltonian[word ^ 2, word] = 0.2
    generator = -1j * time * hamiltonian
    norm = float(generator.abs().sum(dim=1).max())
    scaling = max(0, math.ceil(math.log2(norm))) if norm else 0
    generator = generator / (2**scaling)
    term = torch.eye(4, dtype=torch.complex128)
    total = term.clone()
    for degree in range(1, 33):
        term = term @ generator / degree
        total = total + term
    for _ in range(scaling):
        total = total @ total
    return total


def expected_state(method: str, time: float, repetitions: int = 30) -> torch.Tensor:
    if method == "Exponentiate":
        circuit = fq.Circuit(2, dtype=torch.complex128).any(
            0, 1, matrix=independent_exponential(time)
        )
    else:
        sequence = (
            tuple(
                (index, time * coefficient / repetitions)
                for _ in range(repetitions)
                for index, (coefficient, _) in enumerate(TERMS)
            )
            if method == "Suzuki-Trotter 1st order"
            else reference_sequence(TERMS, time, 288, seed=0)
        )
        circuit = reference_circuit(2, TERMS, sequence)
    state = fq.run(circuit).statevector()
    assert isinstance(state, torch.Tensor)
    return state.flatten()


def test_both_lcu_constructions_full_state_and_clean_workspace() -> None:
    namespace, cells = execute_code_cells(notebook(WORKFLOWS[0][0]))
    assert cells == 3
    programs = cast(list[synth.Program], namespace["programs"])
    assert len(programs) == 2 and programs[0] is not programs[1]
    for program in programs:
        compare_states(program, lcu_reference(), ("controller", "psi"))
        probabilities = dict(
            program.probabilities(fq.run(program), registers=("controller", "psi"))
        )
        success = sum(
            value for (controller, _), value in probabilities.items() if not controller
        )
        assert success == pytest.approx(0.75, abs=1e-12)


@pytest.fixture(scope="module")
def sweep() -> tuple[dict[str, list[torch.Tensor]], dict[str, list[float]]]:
    return evolution.run_sweep(evolution.TIME_POINTS)


def test_three_initial_methods_keep_original_repetitions_and_sample_sequence() -> None:
    programs = evolution.initial_programs()
    assert set(programs) == set(evolution.METHODS)
    for method, program in programs.items():
        actual = fq.run(program).statevector().flatten()
        torch.testing.assert_close(
            actual, expected_state(method, 1.0, 10), atol=1e-11, rtol=0
        )
        assert actual.numel() == 4 and program.report["workspace_qubits"] == 0


def test_every_time_and_method_matches_independent_full_state_and_observable(
    sweep: tuple[dict[str, list[torch.Tensor]], dict[str, list[float]]],
) -> None:
    states, curves = sweep
    times = evolution.TIME_POINTS
    assert len(times) == 100 and times[0] == 0 and times[-1] == 2
    assert times == pytest.approx(
        tuple(2 * index / 99 for index in range(100)), abs=1e-15
    )
    assert set(states) == set(curves) == set(evolution.METHODS)
    diagonal = torch.tensor([1, 0, 0, -1], dtype=torch.float64)
    ideal = evolution.ideal_curve(times)
    for method in evolution.METHODS:
        assert len(states[method]) == len(curves[method]) == 100
        for index, time in enumerate(times):
            expected = expected_state(method, time)
            actual = states[method][index]
            assert actual.numel() == 4
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
            expected_magnetization = float((expected.abs().square() * diagonal).sum())
            assert curves[method][index] == pytest.approx(
                expected_magnetization, abs=1e-11
            )
    for index, time in enumerate(times):
        expected = independent_exponential(time)[:, 0]
        assert ideal[index] == pytest.approx(
            float((expected.abs().square() * diagonal).sum()), abs=1e-11
        )


def test_complete_notebook_nested_factory_and_all_four_plot_series(
    sweep: tuple[dict[str, list[torch.Tensor]], dict[str, list[float]]],
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    path = notebook(WORKFLOWS[1][0])
    try:
        namespace, cells = execute_code_cells(path)
        assert cells == 5
        figure = namespace["figure"]
        assert isinstance(figure, Figure)
        axes = figure.axes[0]
        assert (axes.get_xlabel(), axes.get_ylabel()) == (
            "Evolution coefficient",
            "Magnetization",
        )
        assert [line.get_label() for line in axes.lines] == [
            *evolution.METHODS,
            "Ideal Simulation",
        ]
        states = cast(dict[str, list[torch.Tensor]], namespace["states"])
        curves = cast(dict[str, list[float]], namespace["curves"])
        for index, method in enumerate(evolution.METHODS):
            for actual, expected in zip(states[method], sweep[0][method], strict=True):
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
            assert curves[method] == pytest.approx(sweep[1][method], abs=1e-12)
            assert torch.equal(
                torch.as_tensor(axes.lines[index].get_xdata(), dtype=torch.float64),
                torch.tensor(evolution.TIME_POINTS, dtype=torch.float64),
            )
            assert torch.equal(
                torch.as_tensor(axes.lines[index].get_ydata(), dtype=torch.float64),
                torch.tensor(curves[method], dtype=torch.float64),
            )
        assert torch.equal(
            torch.as_tensor(axes.lines[-1].get_ydata(), dtype=torch.float64),
            torch.tensor(namespace["ideal"], dtype=torch.float64),
        )
        assert torch.as_tensor(axes.lines[-1].get_xdata()).shape == (100,)
    finally:
        plt.close("all")
    assert not any(name.startswith(str(path)) for name in linecache.cache)


@pytest.mark.parametrize("relative,cells,source_cells", WORKFLOWS)
def test_source_provenance_and_clean_native_notebook(
    relative: str, cells: int, source_cells: tuple[int, ...]
) -> None:
    path = notebook(relative)
    proof = json.loads(path.with_name("provenance.json").read_text())
    assert tuple(cell["index"] for cell in proof["reviewed_code_cells"]) == source_cells
    document = json.loads(path.read_text())
    assert sum(cell["cell_type"] == "code" for cell in document["cells"]) == cells
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        assert "Staged only" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_lcu_launcher_executes_both_constructions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = importlib.import_module(
        "examples.synthesis." + WORKFLOWS[0][0].replace("/", ".") + ".overview.main"
    )
    run = cast(Callable[[], None], launcher.run_example)
    run()
    output = capsys.readouterr().out
    assert output.count("Stage:") == output.count("Resources:") == 2
