"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_adjoint import __all__ as __all__
from flagquantum.simulation.kernels.triton.statevector_adjoint import (
    _complex64_local_1q_vjp_adjoint_kernel as _complex64_local_1q_vjp_adjoint_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_adjoint import (
    _complex64_sharded_1q_vjp_adjoint_kernel as _complex64_sharded_1q_vjp_adjoint_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_adjoint import (
    fused_complex64_local_1q_reversible_vjp as fused_complex64_local_1q_reversible_vjp,
)
from flagquantum.simulation.kernels.triton.statevector_adjoint import (
    fused_complex64_local_1q_vjp_adjoint as fused_complex64_local_1q_vjp_adjoint,
)
from flagquantum.simulation.kernels.triton.statevector_adjoint import (
    fused_complex64_sharded_1q_vjp_adjoint as fused_complex64_sharded_1q_vjp_adjoint,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_adjoint"
)
_sys.modules[__name__] = _implementation
