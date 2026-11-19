"""Accept real VQE optimization, all Grover drafts and Jacobi approximation."""

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.jacobi_anger_expansion.overview import (
    main as jacobi,
)
from examples.synthesis.community.basics.vqe.overview import main as vqe
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.workshops.grover_workshop.grover_workshop.overview import (
    main as grover,
)
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    ("community/basics/vqe", 3, (1, 3, 4)),
    (
        "tutorials/workshops/grover_workshop/grover_workshop",
        3,
        (2, 3, 7, 9, 10, 11, 28, 37, 40, 42, 44, 46, 48, 51),
    ),
    (
        "algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/jacobi_anger_expansion",
        2,
        (4, 6),
    ),
)


def test_vqe_optimizes_real_executions_and_every_state_matches_independent_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = vqe.energy
    angles: list[float] = []

    def checked_energy(parameters: torch.Tensor) -> torch.Tensor:
        angle = float(parameters[0])
        angles.append(angle)
        program = vqe.build_program(angle)
        compare_states(
            program, fq.Circuit(1, dtype=torch.complex128).ry(0, theta=angle), ("q",)
        )
        assert program.report["workspace_qubits"] == 0
        value = objective(parameters)
        assert float(value) == pytest.approx(
            math.cos(angle) + 0.1 * math.sin(angle), abs=1e-12
        )
        return value

    monkeypatch.setattr(vqe, "energy", checked_energy)
    result = vqe.run_workflow()
    assert angles[0] == 0 and len(angles) == result.evaluations > 2
    assert result.converged and 0 < result.iterations <= 1000
    assert result.value == pytest.approx(-math.sqrt(1.01), abs=1e-10)
    assert result.value < float(objective(torch.zeros(1, dtype=torch.float64)))


def test_all_five_grover_stages_preserve_drafts_solutions_register_order_and_clean_work() -> (
    None
):
    programs = grover.run_workflow()
    assert set(programs) == {
        "uniform",
        "draft_width",
        "draft_depth",
        "solution_width",
        "solution_depth",
    }
    for name, program in programs.items():
        if name == "uniform":
            reference = fq.Circuit(5, dtype=torch.complex128)
            for bit in range(5):
                reference.h(bit)
            registers: tuple[str, ...] = ("x",)
            assert len(program.registers["x"]) == 5
        else:
            registers = ("a", "b", "c")
            assert tuple(len(program.registers[key]) for key in registers) == (2, 2, 3)
            reference = (
                fq.Circuit(7, dtype=torch.complex128)
                if name.startswith("draft")
                else reference_circuit()
            )
        compare_states(program, reference, registers)
        assert program.report["workspace_qubits"] == 0
        assert tuple(program.registers) == registers
        if name.startswith("draft"):
            assert dict(
                program.probabilities(fq.run(program), registers=registers)
            ) == {(0, 0, 0): 1.0}


def independent_bessel(order: int, time: float) -> float:
    """Use a high-precision power series independently of cosine projection."""
    with localcontext() as context:
        context.prec = 75
        half = Decimal(str(time)) / 2
        term = (
            Decimal(1) if order == 0 else half**order / Decimal(math.factorial(order))
        )
        total = term
        for index in range(1, 1000):
            term *= -(half * half) / (Decimal(index) * Decimal(order + index))
            total += term
            if abs(term) < Decimal("1e-65"):
                return float(total)
    raise AssertionError("Independent Bessel series did not converge")


@pytest.mark.parametrize("time", [0.0, 3.0, 22.0])
def test_jacobi_coefficients_against_decimal_bessel_and_actual_dense_grid(
    time: float,
) -> None:
    degree = jacobi.degree_for_error(time, 1e-7)
    cosine, sine = jacobi.coefficients(degree, time)
    for order in range(degree + 1):
        expected = (
            independent_bessel(order, time)
            * (1 if order == 0 else 2)
            * (-1) ** (order // 2)
        )
        actual = float(cosine[order] if order % 2 == 0 else sine[order])
        assert actual == pytest.approx(expected, abs=1e-13)
        assert float(sine[order] if order % 2 == 0 else cosine[order]) == 0
    # This bounds exact coefficient truncation, separately from numerical error.
    assert 2 * math.exp(0.75 * abs(time)) / (2**degree) <= 1e-7
    grid = torch.linspace(-1, 1, 4097, dtype=torch.float64)
    torch.testing.assert_close(
        jacobi.evaluate(cosine, grid), torch.cos(time * grid), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        jacobi.evaluate(sine, grid), torch.sin(time * grid), atol=1e-7, rtol=0
    )


def test_jacobi_retains_source_time_error_and_all_plot_points() -> None:
    degree, xs, cosine, sine = jacobi.run_workflow()
    assert jacobi.EVOLUTION_TIME == 22 and jacobi.EPS == 1e-7
    assert degree == 49 and xs.shape == cosine.shape == sine.shape == (100,)
    torch.testing.assert_close(
        xs, torch.linspace(0, 1, 100, dtype=torch.float64), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "time,error", [(math.inf, 1e-7), (22.0, 0.0), (22.0, 1.0), (22.0, math.nan)]
)
def test_jacobi_rejects_invalid_approximation_request(
    time: float, error: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        jacobi.degree_for_error(time, error)


@pytest.mark.parametrize(
    "degree,time", [(-1, 22.0), (129, 22.0), (True, 22.0), (49, math.inf), (49, 33.0)]
)
def test_jacobi_rejects_unbounded_projection(degree: int, time: float) -> None:
    with pytest.raises(ValueError, match="local example"):
        jacobi.coefficients(degree, time)


@pytest.mark.parametrize("relative,cells,source_cells", WORKFLOWS)
def test_complete_notebooks_and_source_provenance(
    relative: str, cells: int, source_cells: tuple[int, ...]
) -> None:
    if relative.endswith("jacobi_anger_expansion"):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
    path = ROOT / "examples/synthesis" / relative / "overview/walkthrough.ipynb"
    proof = json.loads(path.with_name("provenance.json").read_text())
    assert tuple(cell["index"] for cell in proof["reviewed_code_cells"]) == source_cells
    document = json.loads(path.read_text())
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
    namespace, executed = execute_code_cells(path)
    assert executed == cells
    if relative.endswith("vqe"):
        assert float(cast(float, namespace["minimal_energy"])) == pytest.approx(
            -math.sqrt(1.01), abs=1e-10
        )
    elif relative.endswith("grover_workshop"):
        assert len(cast(dict[str, object], namespace["programs"])) == 5
    else:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        try:
            figure = namespace["figure"]
            assert isinstance(figure, Figure)
            assert len(figure.axes) == 2
            first, second = figure.axes
            assert [line.get_label() for line in first.lines] == [
                "cos",
                "approx. cos",
                "sin",
                "approx. sin",
            ]
            assert [line.get_label() for line in second.lines] == [
                "cos error",
                "sin error",
            ]
            assert tuple(figure.get_size_inches()) == (14.0, 5.0)
            xs = cast(torch.Tensor, namespace["xs"])
            for axes in (first, second):
                assert axes.get_xlim() == pytest.approx((-0.1, 1.1))
                for line in axes.lines:
                    torch.testing.assert_close(
                        torch.as_tensor(line.get_xdata(), dtype=torch.float64),
                        xs,
                        atol=0,
                        rtol=0,
                    )
            for line, expected in zip(
                second.lines,
                (
                    torch.cos(22 * xs) - cast(torch.Tensor, namespace["cosine"]),
                    torch.sin(22 * xs) - cast(torch.Tensor, namespace["sine"]),
                ),
                strict=True,
            ):
                torch.testing.assert_close(
                    torch.as_tensor(line.get_ydata(), dtype=torch.float64),
                    expected,
                    atol=0,
                    rtol=0,
                )
        finally:
            plt.close("all")
