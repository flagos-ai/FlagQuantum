"""Contracts for the finite, nonnegative MPS full-width probability route."""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import asdict

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import MeasurementNode
from flagquantum.runtime.measurements import execute_measurements
from flagquantum.simulation.mps.bounded_probabilities import (
    _bounded_full_width_probabilities,
)
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def _product(n: int, theta: torch.Tensor) -> MPSState:
    first = torch.stack((torch.cos(theta), 1j * torch.sin(theta)))
    tensors = [first.reshape(1, 1, 2, 1)]
    for _ in range(n - 1):
        tensors.append(torch.tensor([1, 0], dtype=torch.complex128).reshape(1, 1, 2, 1))
    return MPSState(tensors)


@pytest.mark.parametrize("n", (1, 2, 3))
def test_product_probabilities_have_analytic_weights_and_gradients(n: int) -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    state = _product(n, theta)
    original = [tensor.clone() for tensor in state.tensors]
    configuration = asdict(state.config)
    for wires in itertools.permutations(range(n)):
        values = _bounded_full_width_probabilities(state, wires)
        assert values is not None
        assert values.shape == (1, 2**n) and values.dtype == torch.float64
        assert bool((values >= 0).all()) and bool(torch.isfinite(values).all())
        one_position = 1 << (n - wires.index(0) - 1)
        expected = torch.zeros_like(values)
        expected[0, 0] = math.cos(0.37) ** 2
        expected[0, one_position] = math.sin(0.37) ** 2
        torch.testing.assert_close(values, expected, atol=1e-12, rtol=0)
        derivative = torch.autograd.grad(
            values[0, one_position], theta, retain_graph=True
        )[0]
        assert float(derivative) == pytest.approx(math.sin(2 * 0.37), abs=1e-12, rel=0)
    for before, after in zip(original, state.tensors, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert asdict(state.config) == configuration
    assert state.truncation_errors == [] and state.orthogonality_center is None


def test_complex_entangled_state_matches_independent_ordered_born_weights() -> None:
    vector = torch.tensor(
        [1, 2j, -3, 4 - 2j, 0, -1j, 2 + 3j, 1], dtype=torch.complex128
    )
    vector = vector / torch.linalg.vector_norm(vector)
    state = MPSState.from_statevector(vector.reshape(1, -1), 3)
    tensors = [tensor.clone() for tensor in state.tensors]
    records = list(state.truncation_records)
    for wires in itertools.permutations(range(3)):
        values = _bounded_full_width_probabilities(state, wires)
        assert values is not None
        expected = vector.abs().square().reshape(2, 2, 2).permute(wires).reshape(1, 8)
        torch.testing.assert_close(values, expected, atol=1e-12, rtol=0)
    for before, after in zip(tensors, state.tensors, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert state.truncation_records == records


def test_existing_amplitude_contraction_is_bounded_and_never_materializes_dense_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _product(3, torch.tensor(0.37, dtype=torch.float64))
    amplitude = state.amplitude
    calls: list[tuple[int, ...]] = []

    def record(bits: list[int]) -> torch.Tensor:
        calls.append(tuple(bits))
        return amplitude(bits)

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("dense reconstruction or canonicalization is forbidden")

    monkeypatch.setattr(state, "amplitude", record)
    for name in (
        "to_statevector",
        "state",
        "probabilities",
        "canonicalize",
        "orthogonalize_left",
    ):
        monkeypatch.setattr(state, name, forbidden)
    result = execute_measurements(
        state, (MeasurementNode("probabilities", (2, 0, 1)),), n_qubits=3
    )
    assert isinstance(result[0].value, torch.Tensor)
    assert result[0].value.shape == (1, 8)
    assert len(calls) == len(set(calls)) == 8
    assert calls[1] == (0, 1, 0)


@pytest.mark.parametrize(
    "boundary",
    (
        "partial",
        "wide",
        "batch",
        "precision",
        "cutoff",
        "max_bond",
        "truncated",
        "unnormalized",
    ),
)
def test_ineligible_native_results_decline_before_amplitude_contraction(
    boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = MPSState.zero(4 if boundary == "wide" else 2, dtype=torch.complex128)
    wires = tuple(range(state.n_qubits))
    if boundary == "partial":
        wires = (0,)
    elif boundary == "batch":
        state = MPSState.zero(2, bsz=2, dtype=torch.complex128)
    elif boundary == "precision":
        state = MPSState.zero(2, dtype=torch.complex64)
    elif boundary == "cutoff":
        state.config = MPSConfig(cutoff=1e-8)
    elif boundary == "max_bond":
        state.config = MPSConfig(max_bond=2)
    elif boundary == "truncated":
        state.truncation_errors.append(1e-8)
    elif boundary == "unnormalized":
        state.tensors[0] = state.tensors[0] * 2

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("ineligible requests must not contract amplitudes")

    monkeypatch.setattr(state, "amplitude", forbidden)
    assert _bounded_full_width_probabilities(state, wires) is None


def test_expectation_only_target_keeps_existing_parity_route() -> None:
    class Target:
        def __init__(self) -> None:
            self.calls = 0

        def expectation_ps(self, *, z: tuple[int, ...]) -> torch.Tensor:
            self.calls += 1
            return torch.ones(1, dtype=torch.float64)

    target = Target()
    assert _bounded_full_width_probabilities(target, (0, 1)) is None
    result = execute_measurements(
        target, (MeasurementNode("probabilities", (0, 1)),), n_qubits=2
    )
    torch.testing.assert_close(
        result[0].value,
        torch.tensor([[1, 0, 0, 0]], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert target.calls == 3


def test_non_cpu_native_state_declines_without_reading_values() -> None:
    state = MPSState([torch.empty(1, 1, 2, 1, dtype=torch.complex128, device="meta")])
    assert _bounded_full_width_probabilities(state, (0,)) is None


def test_noise_and_sampling_do_not_enter_the_bounded_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.mps.bounded_probabilities as bounded

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("noise or sampling must preserve the existing route")

    monkeypatch.setattr(bounded, "_bounded_full_width_probabilities", forbidden)
    state = MPSState.zero(1, dtype=torch.complex128)
    execute_measurements(
        state,
        (MeasurementNode("probabilities", (0,)),),
        n_qubits=1,
        noise_model=object(),
    )
    execute_measurements(state, (MeasurementNode("sample", (0,), shots=2),), n_qubits=1)


def test_public_exact_mps_probabilities_preserve_phase_sensitive_analytic_weights() -> (
    None
):
    result = fq.run(
        fq.Circuit(1).h(0).t(0).h(0),
        outputs=fq.probabilities((0,)),
        options=fq.ExecutionOptions(
            mode="mps",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    phase = cmath.exp(1j * math.pi / 4)
    expected = torch.tensor(
        [[abs(1 + phase) ** 2 / 4, abs(1 - phase) ** 2 / 4]], dtype=torch.float64
    )
    assert result.probabilities is not None
    torch.testing.assert_close(result.probabilities, expected, atol=1e-12, rtol=0)
    assert bool((result.probabilities >= 0).all())
    assert result.runtime["mode"] == "mps" and result.runtime["truncation_error"] == 0


def test_nonfinite_contraction_fails_without_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = MPSState.zero(1, dtype=torch.complex128)

    def nonfinite(bits: list[int]) -> torch.Tensor:
        return torch.tensor([complex(float("inf"), 0)], dtype=torch.complex128)

    monkeypatch.setattr(state, "amplitude", nonfinite)
    with pytest.raises(ValueError, match="nonfinite"):
        _bounded_full_width_probabilities(state, (0,))
