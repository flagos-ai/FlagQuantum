"""Reproduce the 3-SAT size sweep, oracle semantics and cleanup."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.oracle_generation.three_sat_oracles.overview import (
    oracle,
)

pytestmark = pytest.mark.integration


def test_full_original_size_sweep() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.tutorials.technology_demonstrations.oracle_generation.three_sat_oracles.overview import (
        benchmark,
    )

    rows = benchmark.run()
    assert tuple(row.variables for row in rows) == (
        *range(11, 23),
        24,
        27,
        30,
        33,
        36,
        40,
        45,
        50,
        55,
        61,
        68,
    )
    assert len(benchmark.HISTORICAL_QISKIT) == 12
    assert all(
        row.resources.n_qubits >= row.variables and row.seconds > 0 for row in rows
    )
    assert rows[-1].route == "Python qfunc"
    assert rows[-1].resources.n_qubits == 152
    assert {row.route for row in rows} == {"Python qfunc", "Clause Circuit"}


def test_every_coherent_amplitude_in_both_routes() -> None:
    formula = oracle.Formula((1, 2, 3, 0), (2, 3, 0, 1))
    direct = oracle.clause_circuit(formula, uniform=True)
    program = oracle.build_program(formula, uniform=True)
    phases = torch.tensor(
        [(-1 if formula.satisfies(word) else 1) for word in range(16)],
        dtype=torch.complex128,
    )
    assert phases.real.min() == -1 and phases.real.max() == 1
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.any(*range(4), matrix=torch.diag(phases))
    expected = fq.run(reference).statevector()
    for experiment in (direct, program):
        actual = fq.run(experiment).statevector().reshape(16, -1)
        torch.testing.assert_close(actual[:, 0], expected.flatten(), atol=1e-12, rtol=0)
        assert float(actual[:, 1:].abs().square().sum()) < 1e-24
        assert math.isclose(float(actual.abs().square().sum()), 1, abs_tol=1e-12)


def test_validation_and_source_stage_pins() -> None:
    with pytest.raises(ValueError, match="permutations"):
        oracle.Formula((1, 2, 0), (1, 1, 2))
    with pytest.raises(ValueError, match="distinct"):
        oracle.Formula((1, 2, 0), (1, 2, 0))
    formula = oracle.Formula((1, 2, 0), (2, 0, 1))
    with pytest.raises(ValueError, match="outside"):
        oracle.native_source(formula, word=8)
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/tutorials/technology_demonstrations/oracle_generation/three_sat_oracles/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 9
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
