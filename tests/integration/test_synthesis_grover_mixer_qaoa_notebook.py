"""Verify both four-qubit mixers and actual shot accounting."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.overview import (
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def ordinary_reference(parameters: tuple[float, ...]) -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    values = torch.tensor(
        [
            (bits.count("1") - 1) ** 2
            + sum(
                weight * int(bit)
                for weight, bit in zip((12, -1, 2, 1), bits, strict=True)
            )
            for bits in (f"{word:04b}" for word in range(16))
        ],
        dtype=torch.float64,
    )
    for gamma, beta in zip(parameters[:3], parameters[3:], strict=True):
        circuit.any(*range(4), matrix=torch.diag(torch.exp(1j * gamma * values)))
        for wire in range(4):
            circuit.rx(wire, theta=beta)
    return circuit


@pytest.mark.parametrize("grover", [False, True])
@pytest.mark.parametrize(
    "parameters",
    [(0.12, 0.26, -0.39, 0.43, -0.58, 0.71), (-0.31, 0.44, 0.17, -0.52, 0.63, 0.28)],
)
def test_full_state_and_named_bit_order(
    parameters: tuple[float, ...], grover: bool
) -> None:
    program = quantum.build_program(parameters, grover=grover)
    expected = fq.run(
        (reference_circuit if grover else ordinary_reference)(parameters)
    ).statevector()
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, rtol=0, atol=2e-12
    )
    torch.testing.assert_close(
        torch.tensor(quantum.distribution(program), dtype=torch.float64),
        expected.flatten().abs().square(),
        rtol=0,
        atol=2e-12,
    )
    assert program.report["workspace_qubits"] == 0
    if grover:
        assert (
            sum(
                p
                for word, p in enumerate(quantum.distribution(program))
                if word.bit_count() != 1
            )
            < 1e-22
        )


@pytest.mark.parametrize("grover", [False, True])
def test_budgeted_optimization_with_finite_samples(grover: bool) -> None:
    result = workflow.optimize(grover=grover)
    assert len(result.trace) == 30 and result.budget_reached and not result.converged
    assert result.parameters == min(result.trace, key=lambda row: row.cost).parameters
    for row in result.trace:
        # An integer sum over 1000 outcomes must stay on the discrete shot grid.
        assert row.cost * 1000 == pytest.approx(round(row.cost * 1000), abs=1e-9)
    assert sum(row.count for row in result.observations) == 1000
    assert all(
        row.cost == workflow.cost(row.word, grover=grover)
        for row in result.observations
    )
    feasible = sum(row.count for row in result.observations if row.one_hot)
    if grover:
        assert feasible == 1000
    else:
        assert 0 < feasible < 1000


def test_preserves_all_source_stages() -> None:
    root = Path(quantum.__file__).parent
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    report = json.loads((root / "provenance.json").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(report["reviewed_code_cells"]) == 15
    assert not any(c["outputs"] or c["execution_count"] is not None for c in code)
