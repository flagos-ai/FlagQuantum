# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 collect_like_terms.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/rewrites.py

"""Collect signed terms before binding a block encoding to a circuit."""

from collections.abc import Callable, Hashable, Iterable
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


def collect_like_terms(
    block: _BlockMetadata,
    *,
    rebuild_oracles: bool = False,
    term_key: Callable[[_BlockMetadata], Hashable | None] | None = None,
) -> _BlockMetadata:
    """Collect terms in first occurrence order with the source precision rule.

    By default, supplied oracles remain opaque. Set `rebuild_oracles` to use
    the original rule that recomputes PREPARE and SELECT even for customized
    blocks. Bind the result using its new layouts and term order.

    A supplied key declares interchangeability of complete terms, including
    their metadata and directional binders. A key of None uses native object
    equality, including owned snapshots whose IR mappings are not hashable.
    Explicit keys must be hashable. The first term for each key supplies the binding.
    """

    def rewrite(node: _BlockMetadata) -> _BlockMetadata:
        if isinstance(node, PhaseBlockEncoding):
            return replace(node, block_encoding=rewrite(node.block_encoding))
        if isinstance(node, (ProductBlockEncoding, TensorProductBlockEncoding)):
            return replace(
                node,
                block_encodings=tuple(rewrite(child) for child in node.block_encodings),
            )
        if not isinstance(node, LinearCombinationBlockEncoding) or (
            not rebuild_oracles and not _uses_default_oracles(node)
        ):
            return node

        terms: list[
            tuple[tuple[bool, _BlockMetadata | Hashable], _BlockMetadata, float]
        ] = []
        buckets: dict[int | None, list[int]] = {}

        def accumulate(term: _BlockMetadata, coefficient: float) -> None:
            explicit = term_key(term) if term_key is not None else None
            key = (False, term) if explicit is None else (True, explicit)
            key_hash: int | None
            candidates: Iterable[int]
            try:
                key_hash = hash(key)
            except TypeError:
                if explicit is not None:
                    raise
                key_hash = None
                candidates = range(len(terms))
            else:
                candidates = (*buckets.get(key_hash, ()), *buckets.get(None, ()))
            for index in candidates:
                old_key, first, subtotal = terms[index]
                if old_key == key:
                    terms[index] = old_key, first, subtotal + coefficient
                    return
            buckets.setdefault(key_hash, []).append(len(terms))
            terms.append((key, term, coefficient))

        precision = node.coefficient_bits
        for child, weight in zip(node.block_encodings, node.coefficients, strict=True):
            term = rewrite(child)
            if isinstance(
                term, LinearCombinationBlockEncoding
            ) and _uses_default_oracles(term):
                precision = cast(
                    SymbolicInt,
                    smax(precision, node.coefficient_bits + term.coefficient_bits),
                )
                for nested, coefficient in zip(
                    term.block_encodings, term.coefficients, strict=True
                ):
                    accumulate(nested, weight * coefficient)
            else:
                accumulate(term, weight)
        return LinearCombinationBlockEncoding(
            tuple(term for _, term, _ in terms),
            tuple(coefficient for _, _, coefficient in terms),
            precision,
        )

    return rewrite(block)
