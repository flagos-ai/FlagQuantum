from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.integration


def _hamiltonian() -> Hamiltonian:
    return Hamiltonian(
        (
            pauli_term(0.5, "X", (0,)),
            pauli_term(-0.25, "XZ", (0, 1)),
            pauli_term(0.75, "Y", (0,)),
            pauli_term(0.1, "", ()),
        )
    )


def test_pauli_measurement_plan_rotates_each_group_into_z_basis() -> None:
    circuit = fq.Circuit(2).ry(0, theta=0.37).rx(1, theta=-0.21)
    hamiltonian = _hamiltonian()

    plan = fqd.create_pauli_measurement_plan(
        circuit,
        hamiltonian,
        shots=100,
        optimize=False,
    )

    assert isinstance(plan, fqd.PauliMeasurementPlan)
    assert len(plan.groups) == len(plan.packages) == 2
    assert plan.groups[0].basis == ((0, "x"), (1, "z"))
    assert plan.groups[1].basis == ((0, "y"),)
    assert tuple(item.name for item in plan.packages[0].ir.instructions[-1:]) == ("h",)
    assert tuple(item.name for item in plan.packages[1].ir.instructions[-2:]) == (
        "sdg",
        "h",
    )
    for group, package in zip(plan.groups, plan.packages, strict=True):
        rotated = fq.Circuit.from_ir(package.ir)
        for term_index in group.term_indices:
            term = hamiltonian.terms[term_index]
            if not term.ops:
                continue
            measured = rotated.expectation_ps(z=tuple(wire for wire, _name in term.ops))
            coefficient = float(torch.as_tensor(term.coefficient))
            expected = term.expectation(circuit) / coefficient
            torch.testing.assert_close(measured, expected, atol=1e-6, rtol=0)


def test_grouped_counts_reconstruct_general_pauli_hamiltonian() -> None:
    plan = fqd.create_pauli_measurement_plan(
        fq.Circuit(2),
        _hamiltonian(),
        shots=10,
        optimize=False,
    )
    counts = (
        {"00": 10},
        {"10": 10},
    )

    value = fqd.hamiltonian_expectation_from_grouped_counts(counts, plan)

    torch.testing.assert_close(value, torch.tensor([-0.4]))
    torch.testing.assert_close(plan.expectation(counts), value)


def test_local_provider_executes_grouped_measurement_packages() -> None:
    torch.manual_seed(19)
    circuit = fq.Circuit(2).ry(0, theta=0.37).rx(1, theta=-0.21)
    hamiltonian = _hamiltonian()
    plan = fqd.create_pauli_measurement_plan(
        circuit,
        hamiltonian,
        shots=8192,
        optimize=False,
    )
    provider = InMemoryRemoteTarget()

    results = tuple(provider.run(package) for package in plan.packages)
    measured = plan.expectation(tuple(result.counts for result in results))
    exact = hamiltonian.expectation(circuit)

    torch.testing.assert_close(measured, exact, atol=0.04, rtol=0)
    assert all(result.shots == 8192 for result in results)


def test_grouped_measurement_aggregation_fails_closed() -> None:
    plan = fqd.create_pauli_measurement_plan(
        fq.Circuit(2),
        _hamiltonian(),
        shots=10,
        optimize=False,
    )

    with pytest.raises(ValueError, match="one result per measurement group"):
        plan.expectation(({"00": 10},))
    with pytest.raises(ValueError, match="returned 9 shots"):
        plan.expectation(({"00": 9}, {"00": 10}))
    with pytest.raises(ValueError, match="bitstring width"):
        plan.expectation(({"0": 10}, {"00": 10}))
