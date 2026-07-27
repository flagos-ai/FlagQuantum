"""Private Triton kernels used by FlagQuantum simulation backends.

Backend routing and CPU fallbacks remain in :mod:`flagquantum.simulation`;
this package owns CUDA kernel implementations and their launch wrappers.
"""

from .complex_bmm import fused_complex_bmm, fused_complex_layout_bmm
from .hva_forward_tangent import heisenberg_hva_forward_tangents
from .mps_two_site import fused_mps_two_site
from .single_qubit_loop import repeated_rx_rz, repeated_rx_rz_tangents
from .statevector_gates import cx_sequence, ry_rz_pair, single_qubit_matrix
from .two_qubit_pauli_tangent import repeated_rxx_ryy_rzz_tangents

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
