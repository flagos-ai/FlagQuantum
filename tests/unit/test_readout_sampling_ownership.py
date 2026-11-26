"""Replacement checks for Runtime readout and Simulation sampling ownership."""

from __future__ import annotations

import pytest
import torch

from flagquantum.core.noise import NoiseModel, ReadoutError
from flagquantum.runtime import measurements
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.runtime.results.trajectories import MPSMonteCarloResult
from flagquantum.simulation.mps import sampling as mps_sampling
from flagquantum.simulation.numerics import probabilities
from flagquantum.simulation.statevector import measurement_ops

pytestmark = pytest.mark.unit


def test_pauli_readout_normalizes_before_creating_its_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    distribution = torch.tensor([[0.25, 0.75]])
    normalized = distribution.clone()
    expected = torch.tensor([[[1], [0]]])
    generator = torch.Generator().manual_seed(7)

    def resolve(*args: object, **kwargs: object) -> torch.Tensor:
        events.append("resolve")
        return distribution

    def normalize(value: torch.Tensor) -> torch.Tensor:
        assert value is distribution
        events.append("normalize")
        return normalized

    def create(*args: object) -> torch.Generator:
        events.append("generator")
        return generator

    def draw(
        value: torch.Tensor,
        *,
        shots: int,
        n_qubits: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        assert value is normalized
        assert shots == 2
        assert n_qubits == 1
        assert generator is not None and generator.initial_seed() == 7
        events.append("draw")
        return expected

    monkeypatch.setattr(measurements, "_pauli_basis_distribution", resolve)
    monkeypatch.setattr(
        probabilities, "_normalize_pauli_sampling_distribution", normalize
    )
    monkeypatch.setattr(measurements, "_generator", create)
    monkeypatch.setattr(probabilities, "_sample_pauli_distribution", draw)
    result = measurements._sample_pauli_product(
        object(),
        shots=2,
        wires=(0,),
        metadata={"z": (0,)},
        state_output=object(),
        n_wires=1,
    )
    assert result is expected
    assert events == ["resolve", "normalize", "generator", "draw"]


def test_public_mixture_sampling_keeps_validation_before_numerical_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = object()
    trajectories = (state,)
    result = MPSMonteCarloResult(
        trajectories=trajectories,
        expectation_z_mean=torch.zeros(1),
        expectation_z_variance=torch.zeros(1),
        n_trajectories=1,
    )
    generator = torch.Generator().manual_seed(11)
    expected = torch.zeros((1, 2, 1), dtype=torch.int64)
    calls: list[int] = []

    def draw(
        values: object,
        shots: int,
        *,
        generator: torch.Generator | None,
        format: str,
    ) -> torch.Tensor:
        assert values is trajectories
        assert generator is not None and generator.initial_seed() == 11
        assert format == "bits"
        calls.append(shots)
        return expected

    monkeypatch.setattr(mps_sampling, "_sample_retained_trajectory_mixture", draw)
    assert result.sample(2, generator=generator) is expected
    with pytest.raises(ValueError, match="positive integer"):
        result.sample(0, generator=generator)
    with pytest.raises(ValueError, match="sample format"):
        result.sample(2, generator=generator, format="invalid")
    assert calls == [2]


@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_dynamic_final_draw_delegates_but_runtime_keeps_readout_error(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    calls: list[tuple[int, ...]] = []

    def draw(
        state: torch.Tensor,
        n_qubits: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert n_qubits == 2
        assert isinstance(generator, torch.Generator)
        calls.append(tuple(state.shape))
        shape = (n_qubits,) if strategy == "trajectory" else (state.shape[0], n_qubits)
        return torch.zeros(shape, dtype=torch.int64, device=state.device)

    owner = (
        "_sample_trajectory_state"
        if strategy == "trajectory"
        else "_sample_batched_trajectory_states"
    )
    monkeypatch.setattr(measurement_ops, owner, draw)
    model = NoiseModel().add_readout(0, ReadoutError(((0.0, 1.0), (1.0, 0.0))))
    result = run_dynamic(
        DynamicCircuit(2).h(0),
        shots=7,
        seed=13,
        strategy=strategy,
        noise_model=model,
    )
    assert len(calls) == (7 if strategy == "trajectory" else 1)
    assert torch.equal(result.samples[:, 0], torch.ones(7, dtype=torch.int64))
    assert torch.equal(result.samples[:, 1], torch.zeros(7, dtype=torch.int64))
