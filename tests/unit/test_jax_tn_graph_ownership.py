"""Replace TN numerical owners through unchanged sliced-gradient consumers."""

from collections import Counter
from dataclasses import replace

import pytest

from flagquantum.runtime.executors.jax.tensor_network import contraction, gradients
from flagquantum.simulation.numerics import jax as numerics
from flagquantum.simulation.tensor_network.jax import contraction as local_contraction
from flagquantum.simulation.tensor_network.jax import kernels

pytestmark = pytest.mark.unit


@pytest.fixture
def engines():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    np = pytest.importorskip("numpy")
    previous_x64 = jax.config.x64_enabled
    previous_dtype = numerics._set_active_jax_compute_dtype("complex128")
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax, jnp, np
    finally:
        numerics._set_active_jax_compute_dtype(previous_dtype)
        jax.config.update("jax_enable_x64", previous_x64)


@pytest.mark.parametrize("observable", ["z_sum", "state_norm", "hamiltonian"])
@pytest.mark.parametrize("complex_bytes", [8, 16])
@pytest.mark.parametrize(("world_size", "jit"), [(1, False), (2, True)])
def test_sliced_gradient_uses_replaced_graph_and_local_reduction(
    engines, monkeypatch, observable, complex_bytes, world_size, jit
):
    import torch

    import flagquantum as fq

    _, jnp, np = engines
    parameters = torch.tensor([0.21, -0.37], dtype=torch.float64)

    def build(theta):
        return fq.Circuit(2).ry(0, theta=theta[0]).rx(1, theta=theta[1]).cx(0, 1)

    from flagquantum.simulation.tensor_network.local import build_local_tensor_network
    from flagquantum.simulation.tensor_network.observables import (
        build_tensor_network_expectation,
    )

    plan = build_tensor_network_expectation(
        build_local_tensor_network(build(parameters)), z=(0,)
    )
    counts = Counter(label for node in plan.nodes for label in node.labels)
    sliced_label = next(
        label
        for label, count in counts.items()
        if count >= 2 and label not in plan.output_labels
    )
    options = dict(
        n_qubits=2,
        world_size=world_size,
        sliced_labels=(sliced_label,),
        complex_bytes=complex_bytes,
        observable=observable,
        hamiltonian_terms=((0.7, ((0, "z"),)), (-0.2, ((1, "x"),))),
        jit=jit,
    )
    original = gradients.jax_sliced_tensor_network_parameter_value_and_grad(
        build, parameters, **options
    )
    state_owner = kernels.jax_parameterized_tn_state_nodes
    expectation_owner = kernels.jax_parameterized_tn_expectation_nodes
    states, expectations, reductions = [], [], []

    def reverse_storage(*args, **kwargs):
        nodes, outputs, last = state_owner(*args, **kwargs)
        states.append(kwargs["dtype"])
        # The same labelled network permits a different node storage order.
        return list(reversed(nodes)), outputs, last

    def reconstruct_nodes(*args, **kwargs):
        nodes, outputs = expectation_owner(*args, **kwargs)
        expectations.append(dict(kwargs["ops"]))
        return (
            tuple(
                replace(
                    node,
                    metadata=None if node.metadata is None else dict(node.metadata),
                )
                for node in nodes
            ),
            outputs,
        )

    def sum_partials(partials):
        reductions.append(len(partials))
        return jnp.add.reduce(jnp.stack(partials, axis=0), axis=0)

    monkeypatch.setattr(kernels, "jax_parameterized_tn_state_nodes", reverse_storage)
    monkeypatch.setattr(
        kernels, "jax_parameterized_tn_expectation_nodes", reconstruct_nodes
    )
    monkeypatch.setattr(local_contraction, "sum_tensor_network_partials", sum_partials)
    replaced = gradients.jax_sliced_tensor_network_parameter_value_and_grad(
        build, parameters, **options
    )
    assert states and expectations and reductions
    assert set(states) == {jnp.complex128 if complex_bytes == 16 else jnp.complex64}
    np.testing.assert_allclose(replaced.value, original.value, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        replaced.gradient, original.gradient, atol=2e-6, rtol=2e-6
    )
    assert replaced.summary() == original.summary()


def test_empty_partial_refusal_precedes_numerical_owner(engines, monkeypatch):
    def unexpected(_partials):
        raise AssertionError("Empty ranks must be refused before numerical reduction.")

    monkeypatch.setattr(local_contraction, "sum_tensor_network_partials", unexpected)
    with pytest.raises(ValueError, match="Cannot reduce an empty partial list"):
        contraction._jax_reduce_rank_partials([], collective_backend="local_simulated")
