"""Core FlagQuantum import and autograd must not import optional JAX."""

import sys

import pytest
import torch


@pytest.mark.smoke
def test_core_import_and_autograd_do_not_load_jax():
    assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)

    import flagquantum as fq

    theta = torch.tensor(0.3, requires_grad=True)
    circuit = fq.Circuit(2).h(0).cx(0, 1).rx(0, theta=theta)
    value = circuit.expectation_z(0).sum()
    value.backward()

    assert theta.grad is not None
    assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
