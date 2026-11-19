"""Check both source selectors, signed Taylor readout and bounded synthesis."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.overview import (
    model,
    quantum,
    reference,
    workflow,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("time", model.GRID)
def test_all_times_match_full_absolute_dense_state(time: float) -> None:
    encoding = model.taylor_encoding(time)
    program = quantum.build_program(encoding)
    assert tuple(program.registers) == ("x", "ancilla", "y")
    actual = quantum.logical_state(program)
    expected = fq.run(reference.reference_circuit(time)).statevector().flatten()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    packed = actual.reshape(2, 8, 2)
    torch.testing.assert_close(
        packed[:, :, 1], torch.zeros_like(packed[:, :, 1]), rtol=0, atol=1e-12
    )
    selected = packed[:, 0, 0]
    assert float(selected.imag.abs().max()) < 1e-12
    torch.testing.assert_close(
        selected.real * encoding.scale,
        reference.direct_taylor(time),
        rtol=0,
        atol=1e-12,
    )


def test_workflow_keeps_signed_zero_crossing_and_source_energies() -> None:
    result = workflow.run_workflow()
    assert [row.time for row in result] == list(model.GRID)
    assert result[7].velocity > 0 > result[8].velocity
    assert result[-1].velocity == pytest.approx(-0.3011904761904763, abs=1e-12)
    for row in result:
        assert 0 < row.postselection_probability <= 1.0 + 1e-12
        assert row.potential == row.position**2 / 2
        assert row.kinetic == row.velocity**2 / 2
        error = math.hypot(
            row.position - row.exact_position, row.velocity - row.exact_velocity
        )
        assert error <= math.sqrt(2) * row.time**8 / math.factorial(8) + 1e-12


@pytest.mark.parametrize("time", [-0.01, 1.01, float("nan"), float("inf")])
def test_reject_out_of_scope_time(time: float) -> None:
    with pytest.raises(ValueError, match="source interval"):
        model.taylor_encoding(time)


def test_resource_budget_refuses_insufficient_total_width() -> None:
    encoding = model.taylor_encoding(1.0)
    with pytest.raises(synth.SynthesisError, match="max_qubits=6, actual=7"):
        quantum.build_program(encoding, max_qubits=6)


def test_notebook_preserves_all_clean_source_stages() -> None:
    root = Path(workflow.__file__).parent
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 12
    assert [row["native_code_cell"] for row in proof["reviewed_code_cells"]] == list(
        range(1, 13)
    )
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
