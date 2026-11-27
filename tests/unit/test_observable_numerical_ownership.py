"""Replace numerical owners while Runtime keeps readout policy and cache state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.training.train_rydberg_controls import moving_pair_program
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.runtime.executors.jax.statevector import records as sv_records
from flagquantum.runtime.executors.jax.tensor_network import records as tn_records
from flagquantum.runtime.executors.statevector import reverse_adjoint
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.simulation import native_cpu
from flagquantum.simulation.dynamics import ising_braid, rydberg_presence
from flagquantum.simulation.numerics import probabilities
from flagquantum.simulation.statevector import adjoint

pytestmark = pytest.mark.unit


def _result(
    kind: str, state: torch.Tensor
) -> sv_records.JAXShardedStatevectorResult | tn_records.JAXShardedTensorNetworkResult:
    if kind == "statevector":
        return sv_records.JAXShardedStatevectorResult(
            shards=(),
            plan=SimpleNamespace(
                n_qubits=3, bsz=2, total_amplitudes=8, complex_bytes=16
            ),
            jax_plan=cast(Any, None),
            backend_policy=cast(Any, None),
            local_gate_count=0,
            distributed_gate_count=0,
            simulated_communication_count=0,
            simulated_communication_bytes=0,
            _torch_state_cache=state,
        )
    return tn_records.JAXShardedTensorNetworkResult(
        rank_partials=(),
        reduced_output=object(),
        slicing=None,
        jax_plan=cast(Any, None),
        backend_policy=cast(Any, None),
        n_qubits=3,
        bsz=2,
        complex_bytes=16,
        local_world_size=1,
        node_count=1,
        _torch_state_cache=state,
    )


@pytest.mark.parametrize("kind", ("statevector", "tensor_network"))
def test_jax_probability_methods_use_simulation_and_keep_the_state_cache(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = torch.zeros((2, 8), dtype=torch.complex128)
    result = _result(kind, state)
    expected = torch.full((2, 8), 0.125, dtype=torch.float64)
    calls: list[torch.Tensor] = []

    def read(value: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return expected

    monkeypatch.setattr(probabilities, "_statevector_readout_probabilities", read)
    assert result.probabilities() is expected
    assert result.probability() is expected
    assert len(calls) == 2 and all(value is state for value in calls)
    assert result.state() is state


@pytest.mark.parametrize("kind", ("statevector", "tensor_network"))
def test_jax_z_readout_delegates_ordered_numerics(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = torch.zeros((2, 8), dtype=torch.complex128)
    result = _result(kind, state)
    expected = torch.full((2, 1), 0.25, dtype=torch.float64)
    calls: list[tuple[int, ...]] = []

    def read(
        value: torch.Tensor,
        qubits: tuple[int, ...],
        *,
        n_qubits: int,
        total_amplitudes: int,
    ) -> torch.Tensor:
        assert value is state
        assert n_qubits == 3 and total_amplitudes == 8
        calls.append(tuple(qubits))
        return expected

    monkeypatch.setattr(probabilities, "_statevector_readout_z_expectations", read)
    assert result.expectation_z(2) is expected
    assert calls == [(2,)]
    assert result.state() is state


@pytest.mark.parametrize("kind", ("statevector", "tensor_network"))
def test_jax_rejects_invalid_qubits_before_reconstruction_or_numerical_dispatch(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    result = _result(kind, torch.zeros((2, 8), dtype=torch.complex128))
    result._torch_state_cache = None

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("invalid readout must stop before reconstruction/numerics")

    monkeypatch.setattr(probabilities, "_statevector_readout_z_expectations", forbidden)
    monkeypatch.setattr(
        sv_records, "_reconstruct_torch_state_from_jax_shards", forbidden
    )
    monkeypatch.setattr(tn_records, "_jax_reduced_tn_output_to_torch_state", forbidden)
    for qubit in (-1, 3):
        with pytest.raises(ValueError, match="observable qubit index out of range"):
            result.expectation_z(qubit)


def test_ising_output_uses_its_physical_numerical_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []
    expected = torch.tensor([0.25], dtype=torch.float64)

    def read(state: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(state.shape))
        assert observable.shape == (2, 2) and observable.device == state.device
        return expected

    monkeypatch.setattr(ising_braid, "_ising_output_expectation", read)
    program = IsingBraidProgram().exchange(1).exchange(0, clockwise=True)
    result = fq.run(
        program, outputs=fq.expectation(program.projector("psi"), name="psi")
    )
    assert calls == [(1, 2, 2)]
    assert torch.equal(result.expectation("psi"), expected)
    assert result.state is not None and result.state.shape == (1, 2, 2)


def test_presence_output_uses_its_physical_numerical_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []
    expected = torch.full((129,), 0.25, dtype=torch.float64)

    def read(trajectory: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(trajectory.shape))
        assert operator.shape == (9, 9) and operator.device == trajectory.device
        return expected

    monkeypatch.setattr(rydberg_presence, "_presence_output_expectation", read)
    source = moving_pair_program().measure_presence("a", at=0.5, efficiency=0.8)
    state = source.basis_state(("g", "g"))
    initial = (
        0.9 * torch.outer(state, state.conj())
        + torch.eye(9, dtype=torch.complex128) / 90
    )
    plan = fql.plan(
        source,
        initial,
        torch.linspace(0, 1, 129, dtype=torch.float64),
        parameter_values=[{"omega": 1.2}] * 64 + [{"omega": 0.4}] * 64,
        outputs=fq.expectation(source.survival_operator(), name="survival"),
        options=fq.ExecutionOptions(shots=1, seed=17),
        return_density_matrices=True,
    )
    result = fq.run(plan)
    assert calls == [(129, 9, 9)]
    assert torch.equal(result.expectation("survival"), expected)
    assert result.state is not None and result.state.shape == (1, 9, 9)


def test_runtime_retains_weight_cache_while_dispatching_to_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE", "1")
    reverse_adjoint._clear_observable_weight_cache()
    executed = execute_torch_distributed_statevector(
        fq.Circuit(2, dtype=torch.complex128).h(0), dtype=torch.complex128
    )
    expected = torch.tensor(0.25, dtype=torch.float64)
    calls: list[torch.Tensor] = []

    def read(amplitudes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        assert amplitudes is executed.shard_state.amplitudes
        calls.append(weights)
        return expected

    monkeypatch.setattr(adjoint, "_weighted_observable_expectation", read)
    try:
        for _ in range(2):
            value, weights = (
                reverse_adjoint._local_expectation_z_hamiltonian_and_weights(
                    executed.shard_state,
                    plan=executed.plan,
                    n_wires=2,
                    terms=((1.0, (0,)),),
                )
            )
            assert value is expected and weights is calls[-1]
        assert len(calls) == 2 and calls[0] is calls[1]
    finally:
        reverse_adjoint._clear_observable_weight_cache()


@pytest.mark.parametrize("native_result", (False, True))
def test_weighted_readout_keeps_native_provider_and_existing_tensor_route(
    monkeypatch: pytest.MonkeyPatch, native_result: bool
) -> None:
    amplitudes = torch.tensor([[1.0, 2.0j]], dtype=torch.complex128)
    weights = torch.tensor([0.5, -0.25], dtype=torch.float64)
    marker = torch.tensor(123.0, dtype=torch.float64)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def provider(ket: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor | None:
        calls.append((ket, diagonal))
        return marker if native_result else None

    monkeypatch.setattr(native_cpu, "fused_observable_expectation", provider)
    value = adjoint._weighted_observable_expectation(amplitudes, weights)
    assert len(calls) == 1 and calls[0][0] is amplitudes and calls[0][1] is weights
    if native_result:
        assert value is marker
    else:
        assert value.shape == () and value.dtype == torch.float64
        assert value.item() == -0.5
