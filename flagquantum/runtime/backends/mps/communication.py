"""Shared tensor and transport primitives used by MPS backend algorithms.

This stable internal boundary combines canonical MPS tensor operations with
the state-owning distributed transport API.
"""

from ....simulation.mps.rank_local import (
    apply_one_mps_tensor as _apply_one_mps_tensor,
)
from ....simulation.mps.rank_local import (
    apply_two_mps_tensors_with_info as _apply_two_mps_tensors_with_info,
)
from ....simulation.mps.rank_local import (
    instruction_matrix_for_mps as _instruction_matrix_for_mps,
)
from ....simulation.mps.rank_local import (
    tensor_nbytes as _tensor_nbytes,
)
from ...distributed.mps_transport import (
    _recv_tensor_batch_p2p,
    _recv_tensor_p2p,
    _recv_tensor_static_p2p,
    _send_tensor_batch_p2p,
    _send_tensor_p2p,
    _send_tensor_static_p2p,
    warmup_mps_neighbor_communicators,
)

__all__ = (
    "_apply_one_mps_tensor",
    "_apply_two_mps_tensors_with_info",
    "_instruction_matrix_for_mps",
    "_recv_tensor_p2p",
    "_recv_tensor_batch_p2p",
    "_recv_tensor_static_p2p",
    "_send_tensor_p2p",
    "_send_tensor_batch_p2p",
    "_send_tensor_static_p2p",
    "_tensor_nbytes",
    "warmup_mps_neighbor_communicators",
)
