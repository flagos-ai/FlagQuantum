"""Numerical unit tests for the JAX tensor-network kernels."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.jax,
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
]


def test_jax_contract_nodes_greedy_matches_einsum():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.primitives import (
        jax_contract_tensor_nodes,
    )

    rng = np.random.default_rng(4)
    a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(
        np.complex64
    )
    b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(
        np.complex64
    )

    nodes = [(jnp.asarray(a), (0, 1)), (jnp.asarray(b), (1, 2))]
    got = jax_contract_tensor_nodes(nodes, (0, 2), "highest")
    expected = np.einsum("ij,jk->ik", a, b)
    assert np.allclose(np.asarray(got), expected, atol=1e-4)


def test_jax_tensor_network_loss_state_norm_matches_reference():
    import jax.numpy as jnp
    import numpy as np

    from flagquantum.simulation.jax.tensor_network.kernels import (
        jax_tensor_network_loss_from_output,
    )

    rng = np.random.default_rng(5)
    n_wires = 3
    bsz = 2
    state = (
        rng.standard_normal((bsz, 2**n_wires))
        + 1j * rng.standard_normal((bsz, 2**n_wires))
    ).astype(np.complex64)

    got = jax_tensor_network_loss_from_output(
        jnp.asarray(state),
        n_qubits=n_wires,
        bsz=bsz,
        observable="state_norm",
        observable_qubits=None,
    )
    expected = np.sum(np.conj(state) * state)
    assert np.isclose(float(np.asarray(got)), float(expected.real), atol=1e-3)


@pytest.mark.parametrize("mode", ["statevector", "tensor_network"])
@pytest.mark.parametrize("width", [1, 2, 4, 5, 6])
@pytest.mark.parametrize("selection", ["all", "subset", "duplicates"])
def test_fused_z_sum_and_vjp_match_independent_dense_matrices(
    mode: str,
    width: int,
    selection: str,
) -> None:
    import numpy as np
    import torch

    import flagquantum as fq
    from flagquantum.runtime.executors.jax import compile_quantum_kernel

    selected = (
        tuple(range(width))
        if selection == "all"
        else (width - 1,) if selection == "subset" else (0, width - 1, 0)
    )
    edges = [(wire, wire + 1) for wire in range(width - 1)]
    if width > 1:
        edges.append((width - 1, 0))

    def build(parameters: Any) -> fq.Circuit:
        circuit = fq.Circuit(width, dtype=torch.complex128)
        for wire in range(width):
            circuit.rx(wire, parameters[wire])
            circuit.ry(wire, parameters[width + wire])
            circuit.rz(wire, parameters[wire] * parameters[width + wire])
        for control, target in edges:
            circuit.cx(control, target)
        return circuit

    pauli = (
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.diag([1, -1]).astype(np.complex128),
    )

    def reference(parameters: Any) -> float:
        state = np.zeros(2**width, dtype=np.complex128)
        state[0] = 1
        for wire in range(width):
            angles = (
                parameters[wire],
                parameters[width + wire],
                parameters[wire] * parameters[width + wire],
            )
            for generator, angle in zip(pauli, angles, strict=True):
                rotation = (
                    np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * generator
                )
                matrix = np.ones((1, 1), dtype=np.complex128)
                for position in range(width):
                    matrix = np.kron(
                        matrix, rotation if position == wire else np.eye(2)
                    )
                state = matrix @ state
        for control, target in edges:
            matrix = np.zeros((2**width, 2**width), dtype=np.complex128)
            for index in range(2**width):
                output = (
                    index ^ (1 << (width - 1 - target))
                    if index & (1 << (width - 1 - control))
                    else index
                )
                matrix[output, index] = 1
            state = matrix @ state
        diagonal = sum(
            1 - 2 * ((np.arange(2**width) >> (width - 1 - wire)) & 1)
            for wire in selected
        )
        return float(np.sum(np.abs(state) ** 2 * diagonal))

    parameters = torch.linspace(
        -0.4, 0.6, 2 * width, dtype=torch.float64, requires_grad=True
    )
    kernel = compile_quantum_kernel(
        build,
        parameters,
        n_qubits=width,
        mode=mode,
        observable="z_sum",
        observable_qubits=selected,
        compute_dtype="complex128",
    )
    value = kernel(parameters)
    gradient = torch.autograd.grad(1.3 * value, parameters)[0]
    initial = parameters.detach().numpy().copy()
    np.testing.assert_allclose(
        value.detach().numpy(), reference(initial), atol=1e-11, rtol=1e-11
    )
    expected = []
    for index in range(initial.size):
        plus, minus = initial.copy(), initial.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        expected.append(1.3 * (reference(plus) - reference(minus)) / 2e-6)
    np.testing.assert_allclose(gradient.numpy(), expected, atol=2e-6, rtol=2e-6)


def test_dense_statevector_contraction_respects_intermediate_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    import flagquantum as fq
    from flagquantum.simulation.jax.statevector.kernels import (
        jax_statevector_from_circuit,
    )

    circuit = fq.Circuit(6)
    for wire in range(6):
        circuit.ry(wire, 0.2).rz(wire, -0.3)
    for wire in range(5):
        circuit.cx(wire, wire + 1)
    diagnostics: dict[str, Any] = {}
    dense = jax_statevector_from_circuit(circuit, 6, None, diagnostics=diagnostics)
    assert diagnostics["algorithm"] == "dense_output_contraction"
    assert diagnostics["largest_intermediate_elements_per_entry"] <= 64

    # A cheap path with excessive storage must still use bounded gate updates.
    monkeypatch.setattr(
        jnp,
        "einsum_path",
        lambda *args, **kwargs: (
            [],
            SimpleNamespace(largest_intermediate=128, opt_cost=1),
        ),
    )
    bounded = jax_statevector_from_circuit(circuit, 6, None, diagnostics=diagnostics)
    assert diagnostics["algorithm"] == "fused_gate_updates"
    assert diagnostics["largest_intermediate_elements_per_entry"] == 64
    np.testing.assert_allclose(dense, bounded, atol=2e-6, rtol=2e-6)
