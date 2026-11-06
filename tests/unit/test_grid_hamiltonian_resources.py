"""Verify the locked notebook fixtures under both resource conventions."""

from dataclasses import asdict

import pytest

from examples.algorithms.grid_hamiltonian_resources import main
from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
    source_grid_inverse_sqrt_newton_cost,
    source_grid_inverse_sqrt_polynomial_cost,
)
from flagquantum.algorithms.primitives.grid_kinetic import source_grid_kinetic_cost
from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_pair_potential_cost,
    source_grid_potential_cost,
)
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler._component_trace.terminal_t_cost import rotation_t_cost

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case",
    [
        "kinetic",
        "potential",
        "pair",
        "polynomial",
        "newton",
        "diagram_kinetic",
        "diagram_potential",
    ],
)
def test_grid_notebook_legacy_and_qec_queries(case):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        NewtonRaphsonApproxInverseSquareRoot,
        PolynmomialEvaluationInverseSquareRoot,
    )
    from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PotentialEnergy
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    if case == "kinetic":
        source = KineticEnergy(2, 21)
        modern = legacy = source_grid_kinetic_cost(2, 21)
    elif case == "potential":
        source = PotentialEnergy(2, 21)
        modern = source_grid_potential_cost(2, 21)
        legacy = source_grid_potential_cost(2, 21, legacy_shims=True)
    elif case == "pair":
        source = next(
            instance.bloq
            for instance, _, _ in PotentialEnergy(2, 21)
            .decompose_bloq()
            .iter_bloqsoqs()
        )
        modern = source_grid_pair_potential_cost(6)
        legacy = source_grid_pair_potential_cost(6, legacy_shims=True)
    elif case == "polynomial":
        source = PolynmomialEvaluationInverseSquareRoot(14, 15, 24)
        modern = source_grid_inverse_sqrt_polynomial_cost(14, 15, 24)
        legacy = source_grid_inverse_sqrt_polynomial_cost(14, 15, 24, legacy_shims=True)
    elif case == "diagram_kinetic":
        source = KineticEnergy(12, 17)
        modern = legacy = source_grid_kinetic_cost(12, 17)
    elif case == "diagram_potential":
        source = PotentialEnergy(12, 17)
        modern = source_grid_potential_cost(12, 17)
        legacy = source_grid_potential_cost(12, 17, legacy_shims=True)
    else:
        source = NewtonRaphsonApproxInverseSquareRoot(14, 15, 24)
        modern = source_grid_inverse_sqrt_newton_cost(14, 15, 24)
        legacy = source_grid_inverse_sqrt_newton_cost(14, 15, 24, legacy_shims=True)

    for model, shims in ((modern, False), (legacy, True)):
        actual = estimate_component_resources(model)
        expected = get_cost_value(source, QECGatesCost(legacy_shims=shims))
        assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())
        assert {
            key: value for key, value in asdict(actual.gates).items() if value
        } == expected.asdict()
    budget = legacy_gate_budget(estimate_component_resources(legacy).gates)
    reference = source.t_complexity()
    assert budget == reference.asdict()
    for precision in (1e-4, 1e-11):
        assert budget["t"] + budget["rotations"] * rotation_t_cost(
            precision
        ) == reference.t_incl_rotations(precision)


def test_native_grid_example_reports_separate_costs(capsys):
    main()
    output = capsys.readouterr().out
    assert "QEC T equivalents 9440" in output
    assert "Legacy gates {'t': 9176, 'rotations': 24, 'clifford': 2646}" in output
    assert "Legacy T including synthesized rotations 10424" in output
