"""Verify the generated LABS cost, measured CX selection and finite sampling."""

import json
from pathlib import Path

import pytest

import flagquantum as fq
from examples.synthesis.applications.optimization.low_autocorrelation_binary_sequences_problem.evidence_scaling_labs.overview import (
    checks,
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_full_model_and_exact_minimum() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.applications.optimization.low_autocorrelation_binary_sequences_problem.evidence_scaling_labs.overview import (
        reference,
    )

    terms = model.cost_terms()
    assert len(terms) == 162 and dict(terms)[()] == 246
    expected = reference.direct_costs()
    assert expected.shape == (8192,)
    for word, cost in enumerate(expected):
        assert model.source_cost(word) == cost
        assert (
            sum(
                coefficient * (-1) ** sum((word >> bit) & 1 for bit in support)
                for support, coefficient in terms
            )
            == cost
        )
    assert min(expected) == 174
    assert int(np.count_nonzero(expected == 174)) == 4


def test_measured_cx_choice_preserves_all_complex_amplitudes() -> None:
    candidates, selected = quantum.choose()
    assert tuple(candidate.cx for candidate in candidates) == (2466, 1506, 1164)
    assert selected.schedule is quantum.Schedule.NEAREST
    assert all(item.resources.n_qubits == 13 for item in candidates)
    assert all(
        set(item.resources.operation_counts) == {"cx", "u3"} for item in candidates
    )
    errors = checks.verify(candidates)
    assert len(errors) == 6 and max(errors) < 1e-11

    np = pytest.importorskip("numpy")
    from examples.synthesis.applications.optimization.low_autocorrelation_binary_sequences_problem.evidence_scaling_labs.overview import (
        reference,
    )

    for binding in checks.BINDINGS:
        state = fq.run(quantum.bind(selected, binding)).statevector().flatten()
        np.testing.assert_allclose(
            state.numpy(), reference.direct_state(binding), rtol=0, atol=1e-11
        )


def test_budgeted_execution_keeps_actual_draws() -> None:
    result = workflow.run(budget=2, shots=100)
    assert len(result.trace) == 2 and result.budget_stop and not result.converged
    assert result.chosen.cost == min(query.cost for query in result.trace)
    assert sum(row.count for row in result.samples) == 100
    assert all(row.cost == model.source_cost(row.word) for row in result.samples)
    assert result.exact_minimum == 174 and len(result.exact_words) == 4
    with pytest.raises(ValueError, match="1..60"):
        workflow.run(budget=61)
    with pytest.raises(ValueError, match="six finite"):
        quantum.bind(result.selected, (0.0,))


def test_original_stages_remain_runnable() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 9
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
