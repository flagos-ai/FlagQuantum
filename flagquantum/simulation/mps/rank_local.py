"""Pure tensor operations shared by all MPS executor entry points."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch.profiler import record_function

from ...core.ir import Instruction
from ...ops.gate_matrix import gate_matrix
from .factorization import _split_pair_matrix
from .models import MPSConfig


def instruction_matrix_for_mps(
    instruction: Instruction,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return gate_matrix(instruction, bsz=bsz, device=device, dtype=dtype).to(
        device=device, dtype=dtype
    )


def apply_one_mps_tensor(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)
    if matrix.ndim == 2:
        return torch.einsum("pq,blqr->blpr", matrix, tensor)
    return torch.einsum("bpq,blqr->blpr", matrix, tensor)


def apply_two_mps_tensors_with_info(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor,
    config: MPSConfig,
    *,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    with record_function("flagquantum::mps::two_site_split"):
        matrix = matrix.to(device=left.device, dtype=left.dtype)
        theta = torch.einsum("blsm,bmtr->blstr", left, right)
        if reverse:
            theta = theta.transpose(2, 3)
        theta = theta.reshape(left.shape[0], left.shape[1], 4, right.shape[3])
        if matrix.ndim == 2:
            theta = torch.einsum("ij,bljr->blir", matrix, theta)
        else:
            theta = torch.einsum("bij,bljr->blir", matrix, theta)
        theta = theta.reshape(left.shape[0], left.shape[1], 2, 2, right.shape[3])
        if reverse:
            theta = theta.transpose(2, 3)
        bsz, left_dim, _, _, right_dim = theta.shape
        pair_matrix = theta.reshape(bsz, left_dim * 2, 2 * right_dim)
        return _split_pair_matrix(
            pair_matrix,
            left_dim=left_dim,
            right_dim=right_dim,
            config=config,
        )


def apply_two_mps_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    matrix: torch.Tensor,
    config: MPSConfig,
    *,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    left_tensor, right_tensor, split_info = apply_two_mps_tensors_with_info(
        left, right, matrix, config, reverse=reverse
    )
    return left_tensor, right_tensor, float(split_info["discarded_weight"])


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def apply_rank_local_mps_instruction(
    instruction: Instruction,
    tensors: Sequence[torch.Tensor],
    config: MPSConfig,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any] | None]:
    """Apply one already-routed instruction to its rank-local site tensors."""

    if len(tensors) != len(instruction.wires):
        raise ValueError(
            "instruction wires and rank-local tensors must have equal arity"
        )
    matrix = instruction_matrix_for_mps(
        instruction,
        bsz=bsz,
        device=device,
        dtype=dtype,
    )
    if len(tensors) == 1:
        return (apply_one_mps_tensor(tensors[0], matrix),), None
    if len(tensors) == 2:
        left, right, split_info = apply_two_mps_tensors_with_info(
            tensors[0],
            tensors[1],
            matrix,
            config,
            reverse=int(instruction.wires[0]) > int(instruction.wires[1]),
        )
        return (left, right), split_info
    raise ValueError("rank-local MPS execution supports only one- and two-site gates")


__all__ = (
    "apply_one_mps_tensor",
    "apply_rank_local_mps_instruction",
    "apply_two_mps_tensors",
    "apply_two_mps_tensors_with_info",
    "instruction_matrix_for_mps",
    "tensor_nbytes",
)
