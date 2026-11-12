# Copyright Johnnie Gray and quimb contributors
# SPDX-License-Identifier: Apache-2.0
# Candidate ordering follows quimb 1.15.0 array_ops._numba_find_columns.
# https://github.com/jcmgray/quimb/blob/v1.15.0/quimb/tensor/array_ops.py

"""Identify the first axis supported on a single column at explicit tolerance."""

from math import isfinite
from numbers import Real

import torch


def find_tensor_column(
    tensor: torch.Tensor, *, atol: float = 1e-12
) -> tuple[int, int] | None:
    """Return lexicographically first (axis, column) containing all active data.

    An entry is active iff abs(value) > atol; equality is inactive.
    An all-zero tensor selects the first nonempty axis at column zero. Decisions
    match the reference NumPy/Numba path on admitted finite floating/complex
    data. No dense index grid, NumPy copy or device transfer is made. Reading the
    decision synchronizes device execution; this is not a shape-only planner.
    The detached support mask is not a differentiable sparsity decision.

    Require finite nonnegative real tolerance and finite float32/64 or complex64/128 data.
    Meta tensors, nonfinite data and integer/bool data are explicitly refused.
    This leaves reference backend-specific NaN behavior outside admission.
    """
    if (
        isinstance(atol, bool)
        or not isinstance(atol, Real)
        or not isfinite(atol)
        or atol < 0
    ):
        raise ValueError("column tolerance must be a finite nonnegative real number")
    if tensor.is_meta:
        raise ValueError("column detection requires tensor values, not meta data")
    if tensor.dtype not in {
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }:
        raise ValueError("column detection requires supported floating or complex data")
    data = tensor.detach()
    if not bool(torch.isfinite(data).all()):
        raise ValueError("column detection requires finite tensor values")
    # Reference Numba compares a dtype-sized magnitude against a float64 atol.
    active = data.abs().to(torch.float64) > atol
    for axis, size in enumerate(tensor.shape):
        if not size:
            continue
        other_axes = tuple(i for i in range(tensor.ndim) if i != axis)
        occupied = active.any(dim=other_axes) if other_axes else active
        columns = occupied.nonzero().flatten()
        if columns.numel() < 2:
            return axis, int(columns[0]) if columns.numel() else 0
    return None
