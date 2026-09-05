"""JAX numerical ownership checks that do not require the optional JAX package."""

import pytest

from flagquantum.runtime.backends.jax import kernel
from flagquantum.simulation import jax_gate_primitives

pytestmark = pytest.mark.unit


def test_runtime_reuses_simulation_owned_jax_gate_primitives():
    assert kernel._jax_apply_matrix is jax_gate_primitives._jax_apply_matrix
    assert kernel._jax_complex_dtype is jax_gate_primitives._jax_complex_dtype
