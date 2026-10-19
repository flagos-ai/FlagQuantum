"""Static Hamiltonian terms preserve existing parameter and gradient semantics."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables import boson
from flagquantum.simulation.lindblad import (
    EvolutionValidationError,
    evolve_density_matrix,
)
from flagquantum.simulation.matrix_free_hamiltonian import PauliSum

pytestmark = pytest.mark.unit


def _exchange() -> boson.BosonOperator:
    return boson.create(0) * boson.annihilate(1) + boson.create(1) * boson.annihilate(0)


@pytest.mark.integration
@pytest.mark.parametrize("model", ["qutrit", "boson"])
@pytest.mark.parametrize(
    "method", ["runge-kutta", "crank-nicolson", "krylov-exponential"]
)
def test_shared_named_binding_and_analytic_coupling_gradient(
    model: str, method: str
) -> None:
    variable = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    name = fq.Parameter("drive")
    coefficient = (2 * name - 0.7).bind({name: variable})
    if model == "qutrit":
        operator = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.complex128
        )
        state, dimensions, target = [0.0, 1.0, 0.0], (3,), 2
    else:
        operator = _exchange()
        state, dimensions, target = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (2, 3), 1
    plan = fql.plan(
        [{"operator": operator, "coefficient": coefficient}],
        state,
        torch.linspace(0.0, 0.5, 201, dtype=torch.float64),
        dimensions=dimensions,
        method=method,
        options=fq.ExecutionOptions(require_gradients=True),
    )
    probability = fql.run(plan).populations[-1, target]
    (gradient,) = torch.autograd.grad(probability, variable)
    tolerance = 5e-6 if method == "crank-nicolson" else 2e-9
    assert probability.item() == pytest.approx(math.sin(0.35) ** 2, abs=tolerance)
    assert gradient.item() == pytest.approx(math.sin(0.7), abs=tolerance)
    with pytest.raises(EvolutionValidationError, match="gradient"):
        fql.run(fql.LindbladPlan.from_json(plan.to_json()))


@pytest.mark.parametrize("dimensions", [(2,), (3,)])
def test_bound_terms_and_dense_inputs_seal_identical_plans(
    dimensions: tuple[int, ...],
) -> None:
    side = dimensions[0]
    operator = torch.zeros((side, side), dtype=torch.complex128)
    operator[0, 1] = operator[1, 0] = 1
    terms = [
        {"operator": operator, "coefficient": 0.2},
        {"operator": operator, "coefficient": 0.3},
    ]
    state = [0.0, 1.0] + [0.0] * (side - 2)
    plan = fql.plan(terms, state, [0.0, 0.1], dimensions=dimensions)
    dense = fql.plan(0.5 * operator, state, [0.0, 0.1], dimensions=dimensions)
    assert plan.to_dict() == dense.to_dict()
    restored = fql.LindbladPlan.from_json(plan.to_json())
    assert restored.identity == plan.identity
    torch.testing.assert_close(
        fql.run(plan).populations, fql.run(restored).populations, atol=0, rtol=0
    )


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_complex_conjugate_terms_preserve_both_coefficient_derivatives(
    precision: str,
) -> None:
    alpha = torch.tensor(
        0.4 + 0.3j, dtype=getattr(torch, precision), requires_grad=True
    )
    forward = boson.create(1) * boson.annihilate(0)
    terms = [
        {"operator": forward, "coefficient": alpha},
        {"operator": forward.dagger(), "coefficient": alpha.conj()},
    ]
    result = fql.run(
        terms,
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.5],
        dimensions=(2, 2),
        method="krylov-exponential",
        options=fq.ExecutionOptions(precision=precision, require_gradients=True),
    )
    (derivative,) = torch.autograd.grad(result.populations[-1, 1], alpha)
    expected = (0.4 + 0.3j) / 0.5 * 0.5 * math.sin(0.5)
    tolerance = 3e-5 if precision == "complex64" else 2e-11
    assert derivative.item() == pytest.approx(expected, abs=tolerance)


@pytest.mark.integration
def test_a_real_optimizer_update_reduces_the_boson_population_objective() -> None:
    coupling = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    name = fq.Parameter("coupling")
    optimizer = torch.optim.Adam([coupling], lr=0.04)
    target = math.sin(0.35) ** 2
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        result = fql.run(
            [
                {
                    "operator": _exchange(),
                    "coefficient": name.bind({"coupling": coupling}),
                }
            ],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5],
            dimensions=(2, 3),
            method="krylov-exponential",
            options=fq.ExecutionOptions(require_gradients=True),
        )
        loss = (result.populations[-1, 1] - target).square()
        loss.backward()
        assert coupling.grad is not None and bool(torch.isfinite(coupling.grad))
        losses.append(loss.item())
        optimizer.step()
    assert losses[-1] < losses[0] / 1000
    assert coupling.item() == pytest.approx(0.7, abs=0.015)


def test_batched_initial_states_and_initial_density_adjoint_reuse_the_bound_generator() -> (
    None
):
    value = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    operator = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    terms = [{"operator": operator, "coefficient": value}]
    ground = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex128))
    excited = ground.flip((0, 1))
    times = torch.linspace(0.0, 0.5, 51, dtype=torch.float64)
    result = fql.run(
        terms,
        torch.stack((ground, excited)),
        times,
        n_qubits=1,
        options=fq.ExecutionOptions(batch_size=2, require_gradients=True),
    )
    assert result.populations.shape == (51, 2, 2)
    (derivative,) = torch.autograd.grad(result.populations[-1, 0, 1], value)
    assert derivative.item() == pytest.approx(0.5 * math.sin(0.7), abs=2e-9)
    cotangent = fql.adjoint_gradient(
        terms, ground, times, lambda rho: rho[-1, 1, 1].real, n_qubits=1
    )
    expected = torch.tensor(
        [
            [math.sin(0.35) ** 2, 1j * math.sin(0.7) / 2],
            [-1j * math.sin(0.7) / 2, math.cos(0.35) ** 2],
        ],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(cotangent, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize(
    "coefficient",
    [
        True,
        torch.tensor(True),
        [0.2],
        torch.tensor([0.2]),
        float("nan"),
        float("inf"),
        0.1 + float("inf") * 1j,
        ".2",
        fq.Parameter("missing"),
    ],
)
def test_invalid_or_unbound_coefficients_fail_during_planning(
    coefficient: object,
) -> None:
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(
            [{"operator": boson.number(0), "coefficient": coefficient}],
            [1.0, 0.0, 0.0],
            [0.0, 0.1],
            dimensions=(3,),
        )
    assert caught.value.code == "invalid_coefficient"
    assert caught.value.field == "hamiltonian[0].coefficient"


@pytest.mark.parametrize(
    "terms",
    [
        [{"operator": boson.annihilate(0), "coefficient": 0.2}],
        [{"operator": boson.number(0)}, {"pauli": "Z"}],
        [{"operator": boson.number(0), "pauli": "Z"}],
        [{"operator": boson.number(0), "qubits": [0]}],
        [{"operator": boson.number(0), "qubit": 0}],
        [{"operator": boson.number(0)}, None],
        [{"operator": torch.zeros((2, 2))}],
    ],
)
def test_invalid_full_space_terms_fail_before_evolution(terms: list[object]) -> None:
    with pytest.raises(EvolutionValidationError):
        fql.plan(terms, [1.0, 0.0, 0.0], [0.0, 0.1], dimensions=(3,))


def test_defaults_and_constant_scalars_keep_numeric_plan_behavior() -> None:
    direct = fql.plan(
        [{"operator": boson.number(0)}], [0.0, 1.0, 0.0], [0.0, 0.1], dimensions=(3,)
    )
    dense = fql.plan(boson.number(0), [0.0, 1.0, 0.0], [0.0, 0.1], dimensions=(3,))
    assert direct.to_dict() == dense.to_dict()


def test_existing_numeric_pauli_descriptors_still_evolve_without_dense_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_dense(self: PauliSum) -> torch.Tensor:
        pytest.fail("numeric Pauli evolution must keep its matrix-free path")

    monkeypatch.setattr(PauliSum, "dense", refuse_dense)
    result = evolve_density_matrix(
        [{"pauli": "X", "qubits": [0], "coefficient": 0.5}], "0", 1, [0.0, 0.1]
    )
    assert result.populations[-1, 1].item() == pytest.approx(
        math.sin(0.05) ** 2, abs=2e-9
    )
