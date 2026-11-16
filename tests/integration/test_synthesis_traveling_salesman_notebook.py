"""Check the independent whole-space objective, bound circuit, and quantile rule."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.logistics.traveling_salesman_problem.comparisons.circuit import (
    BINDINGS,
    cost,
    reference_circuit,
)
from examples.synthesis.applications.logistics.traveling_salesman_problem.overview.model import (
    exact_path,
    make_model,
    route,
    visits,
)
from examples.synthesis.applications.logistics.traveling_salesman_problem.overview.quantum import (
    build_program,
)
from examples.synthesis.applications.logistics.traveling_salesman_problem.overview.workflow import (
    low_tail_mean,
)

pytestmark = pytest.mark.integration


def test_all_65536_costs_and_exact_open_route() -> None:
    model = make_model()
    expected = torch.tensor(
        [
            -cost(tuple(bit for row in visits(word) for bit in row))
            for word in range(1 << 16)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(model.cost, expected, atol=0, rtol=0)
    assert int((model.residual == 0).sum()) == 24
    assert exact_path(model) == (3.0, (0, 1, 2, 3))
    assert route(model, 0) is None
    assert route(model, (1 << 0) | (1 << 5) | (1 << 10) | (1 << 15)) == (0, 1, 2, 3)


@pytest.mark.parametrize("angles", BINDINGS)
def test_absolute_state_in_two_nonzero_bindings(angles: tuple[float, ...]) -> None:
    program = build_program(make_model(), angles)
    actual = fq.run(program).statevector()
    expected = fq.run(reference_circuit(angles)).statevector()
    assert tuple(actual.shape) == (1, 65536)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_finite_fractional_tail_and_source_clean_cells() -> None:
    from examples.synthesis.applications.logistics.traveling_salesman_problem.overview import (
        __file__ as overview_path,
    )

    value = low_tail_mean(torch.tensor([6, 2, 1, -5], dtype=torch.float64), 0.625)
    assert float(value) == -1.2
    assert overview_path is not None
    path = Path(overview_path).parent
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 17
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
