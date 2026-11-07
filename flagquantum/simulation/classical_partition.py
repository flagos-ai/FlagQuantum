# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _PartitionBase classical methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py

"""Numerical shaped partitions over bound native scalar and array codecs."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .classical_bit_binding import join_classical_value

Input = TypeVar("Input")
Output = TypeVar("Output")


@dataclass(frozen=True)
class PartitionValuePort:
    """One already-bound named numerical port, without a foreign register type.

    Compiler owns layout legality. Scalar/array codecs own numerical domains.
    Shapes are traversed in the original NumPy row-major order.
    """

    name: str
    width: int
    shape: tuple[int, ...]
    decode: Callable[[Sequence[int]], object]
    decode_array: Callable[[NDArray[Any]], NDArray[Any]]
    encode_array: Callable[[NDArray[Any]], NDArray[Any]]


def partition_classical_value(
    value: Input,
    encode: Callable[[Input], Sequence[int]],
    ports: Sequence[PartitionValuePort],
) -> dict[str, object]:
    """Slice the original encoded word before scalar decode or shaped array decode."""
    outputs: dict[str, object] = {}
    bits = encode(value)
    start = 0
    for port in ports:
        size = int(np.prod(port.shape + (port.width,)))
        part = bits[start : start + size]
        if port.shape == ():
            outputs[port.name] = port.decode(part)
        else:
            outputs[port.name] = port.decode_array(
                np.asarray(part).reshape(port.shape + (port.width,))
            )
        start += size
    return outputs


def unpartition_classical_values(
    values: Mapping[str, object],
    ports: Sequence[PartitionValuePort],
    decode: Callable[[Sequence[int]], Output],
) -> Output:
    """Flatten declared port values, encode arrays, concatenate and decode the word.

    Source callback behavior ignores extra keys and does not validate array
    shapes before ravel. Missing fields and codec/NumPy failures propagate.
    """
    bitstrings: list[NDArray[Any]] = []
    for port in ports:
        value = np.asanyarray(values[port.name])
        encoded = port.encode_array(value.ravel())
        bitstrings.append(encoded.ravel())
    return join_classical_value(np.concatenate(bitstrings), decode)
