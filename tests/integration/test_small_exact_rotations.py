"""Nonzero amplitudes must survive exact compiler and execution paths (#704)."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import optimize
from flagquantum.observables import X

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("epsilon", [1e-10, 1e-12, 1e-13, 1e-14])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_small_entangled_amplitude_is_not_an_identity(
    epsilon: float, custom: bool, level: int
) -> None:
    theta = 2 * math.asin(epsilon)
    circuit = fq.Circuit(2, dtype=torch.complex128)
    if custom:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        circuit.any(0, matrix=torch.tensor([[c, -s], [s, c]], dtype=torch.complex128))
    else:
        circuit.ry(0, theta)
    circuit.cx(0, 1)
    compiled = fq.Circuit.from_ir(optimize(circuit, optimization_level=level))
    result = fq.run(
        compiled,
        outputs=[fq.probabilities()],
        options=fq.ExecutionOptions(
            mode="statevector",
            backend="pytorch",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    actual = result.statevector().reshape(-1)
    # A default absolute tolerance of 1e-12 would accept the buggy zero vector entry.
    assert actual[3].item() == pytest.approx(epsilon, rel=1e-12, abs=0)
    assert actual[1].item() == 0
    assert actual[2].item() == 0
    assert result.probabilities.reshape(-1)[3].item() == pytest.approx(
        epsilon**2, rel=1e-12, abs=0
    )


@pytest.mark.parametrize("epsilon", [-1e-14, 1e-14, -1e-12, 1e-12])
def test_small_complex_amplitude_and_phase_survive_folding(epsilon: float) -> None:
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        precision="complex128",
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    rx = fq.Circuit(2, dtype=torch.complex128).rx(0, 2 * math.asin(epsilon)).cx(0, 1)
    state = (
        fq.run(rx, outputs=[fq.probabilities()], options=options)
        .statevector()
        .reshape(-1)
    )
    assert state[3].item() == pytest.approx(-1j * epsilon, rel=1e-12, abs=0)

    phase = fq.Circuit(1, dtype=torch.complex128).h(0).phase(0, epsilon)
    state = (
        fq.run(phase, outputs=[fq.probabilities()], options=options)
        .statevector()
        .reshape(-1)
    )
    assert state[1].imag.item() == pytest.approx(
        math.sin(epsilon) / math.sqrt(2), rel=1e-12, abs=0
    )

    matrix = torch.eye(2, dtype=torch.complex128) * complex(
        math.cos(epsilon), math.sin(epsilon)
    )
    custom = fq.Circuit(1, dtype=torch.complex128).h(0).any(0, matrix=matrix)
    state = (
        fq.run(custom, outputs=[fq.probabilities()], options=options)
        .statevector()
        .reshape(-1)
    )
    assert state[0].imag.item() == pytest.approx(
        math.sin(epsilon) / math.sqrt(2), rel=1e-12, abs=0
    )


def test_nearly_canceling_rotations_keep_the_nonzero_residual() -> None:
    first, second = 0.37, -0.37 + 2e-14
    circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, first).ry(0, second).cx(0, 1)
    result = fq.run(
        circuit,
        outputs=[fq.probabilities()],
        options=fq.ExecutionOptions(
            mode="statevector",
            backend="pytorch",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    assert result.statevector().reshape(-1)[3].item() == pytest.approx(
        math.sin((first + second) / 2), rel=1e-12, abs=0
    )


def test_trainable_zero_rotation_is_not_deleted() -> None:
    angle = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, angle)
    compiled = optimize(circuit)
    assert compiled.instructions[0].params["theta"] is angle
    result = fq.run(circuit, outputs=[fq.expectation(X(0))])
    result.expectation().backward()
    assert angle.grad.item() == pytest.approx(1.0)


@pytest.mark.parametrize("delta", [-2e-14, 2e-14])
@pytest.mark.parametrize("custom", [False, True])
def test_nearly_antidiagonal_rotations_preserve_small_diagonal(
    delta: float, custom: bool
) -> None:
    theta = math.pi + delta
    circuit = fq.Circuit(2, dtype=torch.complex128)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if custom:
        circuit.any(0, matrix=torch.tensor([[c, -s], [s, c]], dtype=torch.complex128))
    else:
        circuit.ry(0, theta)
    circuit.cx(0, 1)
    result = fq.run(
        circuit,
        outputs=[fq.probabilities()],
        options=fq.ExecutionOptions(
            mode="statevector",
            backend="pytorch",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    assert result.statevector().reshape(-1)[0].item() == pytest.approx(
        c, rel=1e-12, abs=0
    )
