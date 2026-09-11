"""Private Triton kernels used by FlagQuantum simulation backends.

Backend routing and CPU fallbacks remain in :mod:`flagquantum.simulation`;
this package owns CUDA kernel implementations and their launch wrappers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "fused_complex_bmm": "complex_bmm",
    "fused_complex_layout_bmm": "complex_bmm",
    "fused_mps_two_site": "mps_two_site",
    "heisenberg_hva_forward_tangents": "hva_forward_tangent",
    "cx_sequence": "statevector_gates",
    "repeated_rx_rz": "single_qubit_loop",
    "repeated_rx_rz_tangents": "single_qubit_loop",
    "ry_rz_pair": "statevector_gates",
    "single_qubit_matrix": "statevector_gates",
    "repeated_rxx_ryy_rzz_tangents": "two_qubit_pauli_tangent",
}


def __getattr__(name: str) -> Any:
    """Load an accelerator kernel only when that kernel is requested."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "fused_complex_bmm",
    "fused_complex_layout_bmm",
    "fused_mps_two_site",
    "heisenberg_hva_forward_tangents",
    "cx_sequence",
    "repeated_rx_rz",
    "repeated_rx_rz_tangents",
    "ry_rz_pair",
    "single_qubit_matrix",
    "repeated_rxx_ryy_rzz_tangents",
]
