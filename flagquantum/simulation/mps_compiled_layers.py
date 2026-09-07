"""Pure batched numerics for compiled rank-local MPS layers."""

from __future__ import annotations

from typing import Sequence

import torch

from ..core.ir import Instruction
from .mps.factorization import _split_pair_matrix, _split_pair_matrix_bucket
from .mps.models import MPSConfig
from .mps.rank_local import apply_one_mps_tensor, instruction_matrix_for_mps
from .mps.site_kernels import apply_rxx_contraction_bucket, apply_ry_bucket


def _packed_instruction_matrices(
    instructions: Sequence[Instruction],
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrices = []
    for instruction in instructions:
        matrix = instruction_matrix_for_mps(
            instruction,
            bsz=bsz,
            device=device,
            dtype=dtype,
        )
        matrices.append(matrix.expand(bsz, -1, -1) if matrix.ndim == 2 else matrix)
    return torch.stack(matrices)


def apply_compiled_mps_one_site_bucket(
    instructions: Sequence[Instruction],
    tensors: Sequence[torch.Tensor],
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
    compiled: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Apply one equal-shape bucket of independent one-site instructions."""

    if any(instruction.name != "ry" for instruction in instructions):
        raise ValueError("compiled one-site MPS buckets support only RY instructions")
    return apply_mps_one_site_bucket(
        instructions,
        tensors,
        bsz=bsz,
        device=device,
        dtype=dtype,
        compile_ry=compiled,
    )


def apply_mps_one_site_bucket(
    instructions: Sequence[Instruction],
    tensors: Sequence[torch.Tensor],
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
    compile_ry: bool,
) -> tuple[torch.Tensor, ...]:
    """Apply independent one-site instructions, compiling homogeneous RY buckets."""

    if not instructions or len(instructions) != len(tensors):
        raise ValueError(
            "one-site instructions and tensors must be non-empty and equal"
        )
    matrices = _packed_instruction_matrices(
        instructions,
        bsz=bsz,
        device=device,
        dtype=dtype,
    )
    if all(instruction.name == "ry" for instruction in instructions):
        packed = torch.stack(tuple(tensors))
        return tuple(apply_ry_bucket(packed, matrices, compiled=compile_ry).unbind(0))
    return tuple(
        apply_one_mps_tensor(tensor, matrix)
        for tensor, matrix in zip(tensors, matrices.unbind(0))
    )


def contract_mps_two_site_bucket(
    instructions: Sequence[Instruction],
    left_tensors: Sequence[torch.Tensor],
    right_tensors: Sequence[torch.Tensor],
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
    compiled: bool,
) -> torch.Tensor:
    """Contract independent equal-shape two-site instructions without splitting."""

    size = len(instructions)
    if size == 0 or len(left_tensors) != size or len(right_tensors) != size:
        raise ValueError(
            "two-site instructions and tensor pairs must be non-empty and equal"
        )
    matrices = _packed_instruction_matrices(
        instructions,
        bsz=bsz,
        device=device,
        dtype=dtype,
    )
    return apply_rxx_contraction_bucket(
        torch.stack(tuple(left_tensors)),
        torch.stack(tuple(right_tensors)),
        matrices,
        compiled=compiled,
    )


def apply_compiled_mps_two_site_bucket(
    instructions: Sequence[Instruction],
    left_tensors: Sequence[torch.Tensor],
    right_tensors: Sequence[torch.Tensor],
    config: MPSConfig,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype,
    isolate_factorizations: bool,
    svd_driver: str | None,
    compiled: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]], ...]:
    """Contract and factorize one equal-shape bucket of two-site instructions."""

    pairs = contract_mps_two_site_bucket(
        instructions,
        left_tensors,
        right_tensors,
        bsz=bsz,
        device=device,
        dtype=dtype,
        compiled=compiled,
    )
    if isolate_factorizations:
        return tuple(
            _split_pair_matrix(
                pair,
                left_dim=left_tensors[0].shape[1],
                right_dim=right_tensors[0].shape[-1],
                config=config,
            )
            for pair in pairs
        )
    return _split_pair_matrix_bucket(
        pairs,
        left_dim=left_tensors[0].shape[1],
        right_dim=right_tensors[0].shape[-1],
        config=config,
        svd_driver=svd_driver,
    )


__all__ = (
    "apply_compiled_mps_one_site_bucket",
    "apply_compiled_mps_two_site_bucket",
    "apply_mps_one_site_bucket",
    "contract_mps_two_site_bucket",
)
