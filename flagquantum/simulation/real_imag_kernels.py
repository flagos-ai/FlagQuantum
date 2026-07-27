"""Shared real/imag fused kernels for complex MPS and TN contractions."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import torch

_COMPILED_PAIR_CACHE: dict[tuple[Any, ...], Callable[..., torch.Tensor]] = {}
_COMPILE_FAILURES: set[tuple[Any, ...]] = set()
_FUSED_WORKING_SET_BYTES = 256 * 1024 * 1024
_FUSED_INFERENCE_MIN_VOLUME = 2**25
_CANONICAL_LAYOUT_CACHE: dict[
    tuple[str, tuple[int, ...], tuple[int, ...]],
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, int, int],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ]
    | None,
] = {}


def _bmm_real_imag_eager(
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
) -> torch.Tensor:
    real = torch.bmm(left_real, right_real) - torch.bmm(left_imag, right_imag)
    imag = torch.bmm(left_real, right_imag) + torch.bmm(left_imag, right_real)
    return torch.stack((real, imag), dim=-1)


def _canonical_bmm_layout(equation: str, left: torch.Tensor, right: torch.Tensor) -> (
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, int, int],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ]
    | None
):
    cache_key = (equation, tuple(left.shape), tuple(right.shape))
    if cache_key in _CANONICAL_LAYOUT_CACHE:
        layout = _CANONICAL_LAYOUT_CACHE[cache_key]
        return layout
    inputs, output = equation.split("->")
    left_labels, right_labels = inputs.split(",")
    if (
        len(left_labels) != left.ndim
        or len(right_labels) != right.ndim
        or len(set(left_labels)) != len(left_labels)
        or len(set(right_labels)) != len(right_labels)
    ):
        _CANONICAL_LAYOUT_CACHE[cache_key] = None
        return None
    batch = tuple(
        label for label in output if label in left_labels and label in right_labels
    )
    contracted = tuple(
        label for label in left_labels if label in right_labels and label not in output
    )
    left_free = tuple(
        label for label in output if label in left_labels and label not in batch
    )
    right_free = tuple(
        label for label in output if label in right_labels and label not in batch
    )
    if set(left_labels) != set((*batch, *left_free, *contracted)) or set(
        right_labels
    ) != set((*batch, *contracted, *right_free)):
        _CANONICAL_LAYOUT_CACHE[cache_key] = None
        return None

    def dimensions(tensor: torch.Tensor, labels: str) -> dict[str, int]:
        return dict(zip(labels, tensor.shape))

    left_dims = dimensions(left, left_labels)
    right_dims = dimensions(right, right_labels)
    if any(left_dims[label] != right_dims[label] for label in (*batch, *contracted)):
        _CANONICAL_LAYOUT_CACHE[cache_key] = None
        return None
    left_order = (*batch, *left_free, *contracted)
    right_order = (*batch, *contracted, *right_free)
    left_permutation = tuple(left_labels.index(label) for label in left_order)
    right_permutation = tuple(right_labels.index(label) for label in right_order)
    batch_shape = tuple(left_dims[label] for label in batch)
    left_shape = tuple(left_dims[label] for label in left_free)
    contracted_shape = tuple(left_dims[label] for label in contracted)
    right_shape = tuple(right_dims[label] for label in right_free)
    b, m, _, n = (
        prod(batch_shape) or 1,
        prod(left_shape) or 1,
        prod(contracted_shape) or 1,
        prod(right_shape) or 1,
    )
    canonical_labels = (*batch, *left_free, *right_free)
    output_permutation = tuple(canonical_labels.index(label) for label in output)
    canonical_shape = (*batch_shape, *left_shape, *right_shape)
    layout = (
        left_permutation,
        right_permutation,
        (b, m, n),
        canonical_shape,
        output_permutation,
        (batch_shape, left_shape, contracted_shape, right_shape),
    )
    _CANONICAL_LAYOUT_CACHE[cache_key] = layout
    return layout


def _canonical_bmm_inputs(
    equation: str, left: torch.Tensor, right: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[int, ...]] | None:
    layout = _canonical_bmm_layout(equation, left, right)
    if layout is None:
        return None
    (
        left_permutation,
        right_permutation,
        (b, m, n),
        canonical_shape,
        output_permutation,
        _,
    ) = layout
    k = left.numel() // (b * m)
    left_matrix = left.permute(left_permutation).reshape(b, m, k)
    right_matrix = right.permute(right_permutation).reshape(b, k, n)
    return (left_matrix, right_matrix), canonical_shape, output_permutation


def _canonical_layout_requires_materialization(
    left: torch.Tensor,
    right: torch.Tensor,
    layout: tuple,
) -> bool:
    left_permutation, right_permutation, (b, m, n), _, _, _ = layout
    k = left.numel() // (b * m)
    try:
        left.permute(left_permutation).view(b, m, k)
        right.permute(right_permutation).view(b, k, n)
    except RuntimeError:
        return True
    return False


def _real_imag_eager(
    equation: str,
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
) -> torch.Tensor:
    real = torch.einsum(equation, left_real, right_real) - torch.einsum(
        equation, left_imag, right_imag
    )
    imag = torch.einsum(equation, left_real, right_imag) + torch.einsum(
        equation, left_imag, right_real
    )
    return torch.stack((real, imag), dim=-1)


def complex_einsum_pair(
    equation: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    compile_cuda: bool = True,
) -> torch.Tensor:
    """Contract two tensors without placing complex operators in the compiled graph."""

    if not (left.is_complex() and right.is_complex()):
        return torch.einsum(equation, left, right)
    if not left.is_cuda:
        # Native CPU complex einsum is faster than four real contractions.
        return torch.einsum(equation, left, right)
    layout = _canonical_bmm_layout(equation, left, right)
    if layout is not None:
        _, _, (b, m, n), _, _, shapes = layout
        output_elements = b * m * n
        working_set_bytes = (
            left.numel() + right.numel() + output_elements
        ) * left.element_size()
        inference_layout_crossover = (
            not (left.requires_grad or right.requires_grad)
            and b * m * (left.numel() // (b * m)) * n >= _FUSED_INFERENCE_MIN_VOLUME
            and _canonical_layout_requires_materialization(left, right, layout)
        )
        if working_set_bytes >= _FUSED_WORKING_SET_BYTES or inference_layout_crossover:
            from .triton_kernels.complex_bmm import fused_complex_layout_bmm

            (
                left_permutation,
                right_permutation,
                _,
                output_shape,
                output_permutation,
                _,
            ) = layout
            result = fused_complex_layout_bmm(
                left, right, left_permutation, right_permutation, shapes
            ).reshape(output_shape)
            if output_permutation != tuple(range(len(output_permutation))):
                result = result.permute(output_permutation)
            return result
        return torch.einsum(equation, left, right)
    if not compile_cuda:
        # Callers can suppress shape-specialized torch.compile kernels without
        # disabling the bounded-specialization fused canonical BMM path above.
        return torch.einsum(equation, left, right)
    left_parts = torch.view_as_real(left.resolve_conj().resolve_neg())
    right_parts = torch.view_as_real(right.resolve_conj().resolve_neg())
    args = (
        left_parts[..., 0],
        left_parts[..., 1],
        right_parts[..., 0],
        right_parts[..., 1],
    )
    key = (
        equation,
        tuple(left.shape),
        tuple(right.shape),
        left.dtype,
        str(left.device),
    )
    kernel = _COMPILED_PAIR_CACHE.get(key)
    if (
        kernel is None
        and compile_cuda
        and left.is_cuda
        and hasattr(torch, "compile")
        and key not in _COMPILE_FAILURES
    ):

        def eager(
            left_real: torch.Tensor,
            left_imag: torch.Tensor,
            right_real: torch.Tensor,
            right_imag: torch.Tensor,
        ) -> torch.Tensor:
            return _real_imag_eager(
                equation, left_real, left_imag, right_real, right_imag
            )

        kernel = torch.compile(
            eager,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        _COMPILED_PAIR_CACHE[key] = kernel
    if kernel is None:
        pair = _real_imag_eager(equation, *args)
    else:
        try:
            from torch._dynamo import config

            limit = max(int(config.recompile_limit), 4 * len(_COMPILED_PAIR_CACHE), 64)
            with config.patch(
                recompile_limit=limit,
                accumulated_recompile_limit=max(
                    int(config.accumulated_recompile_limit), limit
                ),
            ):
                pair = kernel(*args)
        except Exception:  # noqa: BLE001 - optional compiler must be recoverable.
            _COMPILED_PAIR_CACHE.pop(key, None)
            _COMPILE_FAILURES.add(key)
            pair = _real_imag_eager(equation, *args)
    return torch.view_as_complex(pair.contiguous())


def kernel_cache_summary() -> dict[str, int]:
    return {
        "compiled_pair_kernels": len(_COMPILED_PAIR_CACHE),
        "canonical_bmm_kernels": sum(
            int(key[0] == "canonical_bmm") for key in _COMPILED_PAIR_CACHE
        ),
        "equation_kernels": sum(
            int(key[0] != "canonical_bmm") for key in _COMPILED_PAIR_CACHE
        ),
        "compile_failures": len(_COMPILE_FAILURES),
        "canonical_layouts": sum(
            layout is not None for layout in _CANONICAL_LAYOUT_CACHE.values()
        ),
        "noncanonical_layouts": sum(
            layout is None for layout in _CANONICAL_LAYOUT_CACHE.values()
        ),
    }


__all__ = ["complex_einsum_pair", "kernel_cache_summary"]
