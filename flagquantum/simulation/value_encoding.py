# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitEncoding default array encode/decode.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Shared scalar-codec vectorization without a dtype or register hierarchy."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def encode_value_array(
    values: NDArray[Any], encode: Callable[[object], Sequence[int]]
) -> NDArray[np.uint8]:
    """Apply the source default scalar-to-bit-vector signature and uint8 output cast."""
    return cast(
        "NDArray[np.uint8]",
        np.vectorize(
            lambda value: np.asarray(encode(value), dtype=np.uint8),
            signature="()->(n)",
        )(values),
    )


def decode_value_array(
    bits: NDArray[Any], decode: Callable[[NDArray[Any]], object]
) -> NDArray[Any]:
    """Keep source vectorized scalar result inference and core-dimension errors."""
    return cast("NDArray[Any]", np.vectorize(decode, signature="(n)->()")(bits))
