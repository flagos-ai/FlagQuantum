# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OnEach.build_composite_bloq and numerical graph order.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py

"""Evaluate a bound word's single-bit child components in source order."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..core.value_parameters import is_symbolic_value
from .bit_encoding import validate_bit_array, validate_bit_value
from .classical_bit_binding import join_classical_value, split_classical_value
from .classical_registers import collect_integer_register_value
from .classical_updates import commit_phased_classical_outputs

if TYPE_CHECKING:
    from sympy import Expr

Input = TypeVar("Input")
Output = TypeVar("Output")


def _split_bits(
    value: Input,
    encode: Callable[[Input], Sequence[int]],
    shape: tuple[int, ...],
) -> NDArray[Any]:
    bits = split_classical_value(value, encode)
    if bits.shape != shape:
        raise ValueError("Split bit array differs from its bound register shape")
    validate_bit_array(bits)
    return bits


def on_each_classical_value(
    value: Input,
    width: int | np.integer[Any] | Expr,
    encode: Callable[[Input], Sequence[int]],
    decode: Callable[[Sequence[int]], Output],
    child: Callable[[object], tuple[object, complex]],
    validate_word: Callable[[object], None],
    *,
    register_width: int | np.integer[Any] | Expr | None = None,
) -> tuple[Output, complex]:
    """Use bound codecs and a complete single-q THRU numerical child evaluator.

    The metadata owner validates child and dtype signatures. Bit results retain
    their objects until the original uint8 Join collection step. Zero-width graph
    execution reaches the independent Join before Split, retaining its diagnostic.
    """
    validate_word(value)
    if not isinstance(width, (int, np.integer)):
        try:
            from sympy import Expr
        except ModuleNotFoundError as error:
            if error.name != "sympy":
                raise
        else:
            if isinstance(width, Expr):
                raise NotImplementedError(
                    "Cannot numerically decompose OnEach with an expression width"
                )
    bound_width = width if register_width is None else register_width
    if is_symbolic_value(bound_width):
        raise ValueError("Split cannot have a symbolic data type")
    # Match the graph builder's concrete array allocation before numerical Split.
    layout = np.empty((bound_width,), dtype=object)
    for index in range(width):
        if index >= layout.size:
            raise IndexError("OnEach child index is outside the bound split register")
    if layout.size == 0:
        empty = np.empty((0,), dtype=np.uint8)
        output = join_classical_value(empty, decode)
        validate_word(output)
        _split_bits(value, encode, layout.shape)
        return output, 1.0
    bits = _split_bits(value, encode, layout.shape)
    assigned: dict[tuple[int, ...], object] = {
        (index,): bits[index] for index in range(layout.size)
    }

    def assign(index: int, outputs: Mapping[str, object]) -> None:
        validate_bit_value(outputs["q"])
        assigned[(index,)] = outputs["q"]

    phase: complex = 1.0
    for index in range(width):
        result, contribution = child(assigned[(index,)])
        phase = commit_phased_classical_outputs(
            {"q": result},
            contribution,
            partial(assign, index),
            phase=phase,
            outcomes={},
        )
    collected = collect_integer_register_value(assigned, 1, shape=layout.shape)
    # The shared collector returns an ndarray for this non-scalar shape.
    assert isinstance(collected, np.ndarray)
    output = join_classical_value(collected, decode)
    validate_word(output)
    return output, phase
