"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.single_qubit_loop import __all__ as __all__
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _LoopContext as _LoopContext,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _RepeatedRXRZ as _RepeatedRXRZ,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _rx_rz_loop_backward_kernel as _rx_rz_loop_backward_kernel,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _rx_rz_loop_kernel as _rx_rz_loop_kernel,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _rx_rz_loop_tangent_kernel as _rx_rz_loop_tangent_kernel,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _tensor_jacobian as _tensor_jacobian,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    _TensorJacobian as _TensorJacobian,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    repeated_rx_rz as repeated_rx_rz,
)
from flagquantum.simulation.kernels.triton.single_qubit_loop import (
    repeated_rx_rz_tangents as repeated_rx_rz_tangents,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.single_qubit_loop"
)
_sys.modules[__name__] = _implementation
