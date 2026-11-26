"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.statevector_gates import __all__ as __all__
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_control_one_pack_kernel as _complex64_control_one_pack_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_control_one_unpack_kernel as _complex64_control_one_unpack_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_local_1q_kernel as _complex64_local_1q_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_local_2q_kernel as _complex64_local_2q_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_local_cx_kernel as _complex64_local_cx_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_local_cx_segment_kernel as _complex64_local_cx_segment_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _complex64_transpose_1q_kernel as _complex64_transpose_1q_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _cx_sequence_kernel as _cx_sequence_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _CXSequence as _CXSequence,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _GateContext as _GateContext,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _launch_cx_sequence as _launch_cx_sequence,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _launch_single_qubit_matrix as _launch_single_qubit_matrix,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _launch_single_qubit_matrix_backward as _launch_single_qubit_matrix_backward,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _local_cx_segment_positions as _local_cx_segment_positions,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _MatrixGateContext as _MatrixGateContext,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _require_flat_local_address as _require_flat_local_address,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _ry_rz_pair_kernel as _ry_rz_pair_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _single_qubit_matrix_backward_kernel as _single_qubit_matrix_backward_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _single_qubit_matrix_kernel as _single_qubit_matrix_kernel,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    _SingleQubitMatrix as _SingleQubitMatrix,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    apply_complex64_local_1q as apply_complex64_local_1q,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    apply_complex64_local_2q as apply_complex64_local_2q,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    apply_complex64_local_cx_inplace as apply_complex64_local_cx_inplace,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    apply_complex64_local_cx_segment as apply_complex64_local_cx_segment,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    apply_complex64_transpose_1q_inplace as apply_complex64_transpose_1q_inplace,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    cx_sequence as cx_sequence,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    pack_complex64_control_one as pack_complex64_control_one,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    ry_rz_pair as ry_rz_pair,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    single_qubit_matrix as single_qubit_matrix,
)
from flagquantum.simulation.kernels.triton.statevector_gates import (
    unpack_complex64_control_one as unpack_complex64_control_one,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.statevector_gates"
)
_sys.modules[__name__] = _implementation
