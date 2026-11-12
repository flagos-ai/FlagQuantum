# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Support predicates follow quimb 1.15.0 tensor/array_ops.py.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/array_ops.py

"""Shared finite support masks and lexicographic (anti)diagonal detection."""

from itertools import combinations
from math import isfinite
from numbers import Real

import torch


def _tensor_support(tensor: torch.Tensor, atol: float) -> torch.Tensor:
    if (
        isinstance(atol, bool)
        or not isinstance(atol, Real)
        or not isfinite(atol)
        or atol < 0
    ):
        raise ValueError("structure tolerance must be a finite nonnegative real number")
    if tensor.is_meta:
        raise ValueError("support detection requires tensor values, not meta data")
    if tensor.dtype not in {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise ValueError(
            "support detection requires supported floating or complex data"
        )
    data = tensor.detach()
    if not bool(torch.isfinite(data).all()):
        raise ValueError("support detection requires finite tensor values")
    # Reference Numba compares dtype-sized magnitudes against float64 atol.
    return data.abs().to(torch.float64) > atol


def find_tensor_diagonal(
    tensor: torch.Tensor, *, antidiagonal: bool = False, atol: float = 1e-12
) -> tuple[int, int] | None:
    """Return the first equal-sized axis pair with (anti)diagonal support.

    Match the finite NumPy/Numba reference candidate order and strict > atol
    activity rule. Per-pair support reductions avoid a rank-sized coordinate
    grid. All-zero and empty data admit the first equal-sized pair. Decisions
    read detached data and synchronize device execution; they are not gradients.
    Finite float32/64 or complex64/128 values and finite nonnegative tolerance
    are required. Meta data and unsupported precisions fail explicitly.
    """
    if type(antidiagonal) is not bool:
        raise ValueError("antidiagonal selection requires a built-in boolean")
    active = _tensor_support(tensor, atol)
    for left, right in combinations(range(tensor.ndim), 2):
        if tensor.shape[left] != tensor.shape[right]:
            continue
        if not active.numel():
            return left, right
        others = tuple(axis for axis in range(tensor.ndim) if axis not in (left, right))
        projected = active.any(dim=others) if others else active
        diagonal = (projected.flip((1,)) if antidiagonal else projected).diagonal()
        if bool(projected.sum() == diagonal.sum()):
            return left, right
    return None
