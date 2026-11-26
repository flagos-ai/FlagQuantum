"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.simulation.kernels.triton import (
    _EXPORT_MODULES as _EXPORT_MODULES,
)
from flagquantum.simulation.kernels.triton import (
    FLAT_LOCAL_MAX_AMPLITUDES as FLAT_LOCAL_MAX_AMPLITUDES,
)
from flagquantum.simulation.kernels.triton import __all__ as __all__
from flagquantum.simulation.kernels.triton import (
    cx_sequence as cx_sequence,
)
from flagquantum.simulation.kernels.triton import (
    fused_complex_bmm as fused_complex_bmm,
)
from flagquantum.simulation.kernels.triton import (
    fused_complex_layout_bmm as fused_complex_layout_bmm,
)
from flagquantum.simulation.kernels.triton import (
    fused_mps_one_site as fused_mps_one_site,
)
from flagquantum.simulation.kernels.triton import (
    fused_mps_two_site as fused_mps_two_site,
)
from flagquantum.simulation.kernels.triton import (
    heisenberg_hva_forward_tangents as heisenberg_hva_forward_tangents,
)
from flagquantum.simulation.kernels.triton import (
    repeated_rx_rz as repeated_rx_rz,
)
from flagquantum.simulation.kernels.triton import (
    repeated_rx_rz_tangents as repeated_rx_rz_tangents,
)
from flagquantum.simulation.kernels.triton import (
    repeated_rxx_ryy_rzz_tangents as repeated_rxx_ryy_rzz_tangents,
)
from flagquantum.simulation.kernels.triton import (
    ry_rz_pair as ry_rz_pair,
)
from flagquantum.simulation.kernels.triton import (
    single_qubit_matrix as single_qubit_matrix,
)
from flagquantum.simulation.kernels.triton import (
    statevector_marginal_probabilities as statevector_marginal_probabilities,
)
from flagquantum.simulation.kernels.triton import (
    statevector_pauli_expectation as statevector_pauli_expectation,
)
from flagquantum.simulation.kernels.triton import (
    statevector_probabilities as statevector_probabilities,
)

_implementation = _import_module("flagquantum.simulation.kernels.triton")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
