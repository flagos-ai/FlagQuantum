# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 collect_like_terms.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/rewrites.py

"""Collect signed terms before binding a block encoding to a circuit."""

from collections import defaultdict
from dataclasses import replace
from typing import cast

from ...compiler._component_trace.symbolic_math import SymbolicInt, smax
from ._block_metadata import _BlockMetadata
from .linear_combination_block_encoding import LinearCombinationBlockEncoding
from .phase_block_encoding import PhaseBlockEncoding
from .product_block_encoding import ProductBlockEncoding
from .tensor_product_block_encoding import TensorProductBlockEncoding


def _uses_default_oracles(block: LinearCombinationBlockEncoding) -> bool:
    return block.prepare_override is None and block.select_override is None


def collect_like_terms(block: _BlockMetadata) -> _BlockMetadata:
    """Flatten nested default LCUs, preserving first occurrence order and precision.

    Native child equality determines which terms can merge. Custom-oracle
    blocks and unrecognized block families pass through as opaque children.
    Bind the rewritten block with factories in its new term order.
    """
    if isinstance(block, PhaseBlockEncoding):
        return replace(block, block_encoding=collect_like_terms(block.block_encoding))
    if isinstance(block, (ProductBlockEncoding, TensorProductBlockEncoding)):
        return replace(
            block,
            block_encodings=tuple(
                collect_like_terms(child) for child in block.block_encodings
            ),
        )
    if not isinstance(
        block, LinearCombinationBlockEncoding
    ) or not _uses_default_oracles(block):
        return block

    coefficients: dict[_BlockMetadata, float] = defaultdict(float)
    precision = block.coefficient_bits
    for child, weight in zip(block.block_encodings, block.coefficients, strict=True):
        term = collect_like_terms(child)
        if isinstance(term, LinearCombinationBlockEncoding) and _uses_default_oracles(
            term
        ):
            precision = cast(
                SymbolicInt,
                smax(precision, block.coefficient_bits + term.coefficient_bits),
            )
            for nested, coefficient in zip(
                term.block_encodings, term.coefficients, strict=True
            ):
                coefficients[nested] += weight * coefficient
        else:
            coefficients[term] += weight
    return LinearCombinationBlockEncoding(
        tuple(coefficients), tuple(coefficients.values()), precision
    )
