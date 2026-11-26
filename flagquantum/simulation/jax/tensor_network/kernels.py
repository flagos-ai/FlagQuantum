"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.tensor_network.jax.kernels import (
    _JAX_EINSUM_CHARS as _JAX_EINSUM_CHARS,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    _jax_einsum_by_labels as _jax_einsum_by_labels,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    _jax_einsum_reorder as _jax_einsum_reorder,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_slice_tensor_by_labels as jax_slice_tensor_by_labels,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_tensor_network_expectation_product_ops as jax_tensor_network_expectation_product_ops,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_tensor_network_hamiltonian_expectation as jax_tensor_network_hamiltonian_expectation,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_tensor_network_loss_from_output as jax_tensor_network_loss_from_output,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_tensor_network_pauli_string_expectation as jax_tensor_network_pauli_string_expectation,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_tensor_network_z_sum as jax_tensor_network_z_sum,
)
from flagquantum.simulation.tensor_network.jax.kernels import (
    jax_tensor_network_z_values as jax_tensor_network_z_values,
)

_sys.modules[__name__] = _import_module(
    "flagquantum.simulation.tensor_network.jax.kernels"
)
