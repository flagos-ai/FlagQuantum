"""VQE optimizer injection preserves training and parameter ownership."""

from collections.abc import Iterable

import pytest
import torch

from flagquantum.algorithms import (
    Hamiltonian,
    HamiltonianTerm,
    OptimizerFactory,
    run_adapt_vqe,
    run_vqe,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit


def _vqe_circuit(parameters: torch.Tensor) -> Circuit:
    return Circuit(1).gate("rx", 0, theta=parameters[0])


def _adapt_circuit(operators: tuple[object, ...], parameters: torch.Tensor) -> Circuit:
    circuit = Circuit(1, dtype=torch.complex128).gate("h", 0)
    for operator, parameter in zip(operators, parameters, strict=True):
        assert isinstance(operator, str)
        circuit.gate(operator, 0, theta=parameter)
    return circuit


@pytest.mark.parametrize("optimizer_factory", [torch.optim.Adam, torch.optim.SGD])
def test_vqe_accepts_optimizer_constructors(
    optimizer_factory: OptimizerFactory,
) -> None:
    initial = torch.tensor([1.0])
    hamiltonian = Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),))
    result = run_vqe(
        _vqe_circuit,
        initial,
        hamiltonian,
        steps=20,
        lr=0.2,
        optimizer_factory=optimizer_factory,
    )
    assert result.energy < hamiltonian.expectation(_vqe_circuit(initial))
    assert result.history[-1] < result.history[0]
    torch.testing.assert_close(initial, torch.tensor([1.0]))


@pytest.mark.parametrize("optimizer_factory", [torch.optim.Adam, torch.optim.SGD])
def test_adapt_vqe_accepts_optimizer_constructors(
    optimizer_factory: OptimizerFactory,
) -> None:
    result = run_adapt_vqe(
        _adapt_circuit,
        ("rx", "ry"),
        Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),)),
        max_adapt_iterations=1,
        optimization_steps=20,
        lr=0.1,
        optimizer_factory=optimizer_factory,
    )
    assert result.selected_pool_indices == (1,)
    assert result.energy < result.initial_energy


def test_custom_factory_receives_owned_parameters_and_learning_rates() -> None:
    calls: list[tuple[tuple[torch.Tensor, ...], float, torch.optim.Optimizer]] = []

    def factory(params: Iterable[torch.Tensor], *, lr: float) -> torch.optim.Optimizer:
        parameters = tuple(params)
        optimizer = torch.optim.SGD(parameters, lr=lr)
        calls.append((parameters, lr, optimizer))
        return optimizer

    hamiltonian = Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),))
    initial = torch.tensor([1.0])
    run_vqe(
        _vqe_circuit,
        initial,
        hamiltonian,
        steps=3,
        lr=0.2,
        optimizer_factory=factory,
    )
    run_adapt_vqe(
        _adapt_circuit,
        ("ry",),
        hamiltonian,
        max_adapt_iterations=1,
        optimization_steps=3,
        lr=0.1,
        optimizer_factory=factory,
    )
    assert len(calls) == 2
    assert [lr for _, lr, _ in calls] == [0.2, 0.1]
    assert calls[0][0][0].data_ptr() != initial.data_ptr()
    assert calls[0][0][0].dtype == torch.float32
    assert calls[1][0][0].dtype == torch.float64
    for parameters, _, optimizer in calls:
        assert len(parameters) == 1
        parameter = parameters[0]
        assert parameter.is_leaf and parameter.requires_grad
        assert parameter.grad is not None
        assert optimizer.param_groups[0]["params"][0] is parameter


def test_legacy_optimizer_cls_keyword_is_not_supported() -> None:
    hamiltonian = Hamiltonian((HamiltonianTerm(1.0, {0: "z"}),))
    with pytest.raises(TypeError):
        run_vqe(
            _vqe_circuit,
            torch.tensor([1.0]),
            hamiltonian,
            optimizer_cls=torch.optim.Adam,  # type: ignore[call-arg]
        )
