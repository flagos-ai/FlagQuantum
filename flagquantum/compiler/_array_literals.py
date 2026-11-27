# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 serialization/args.py.

"""Own pickle-free immutable array literals for compiler data and wire codecs."""

from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..errors import SerializationError


def freeze_array(value: NDArray[Any]) -> bytes:
    """Preserve dtype, shape and scalar precision without object pickle."""
    with BytesIO() as stream:
        try:
            np.save(stream, value, allow_pickle=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise SerializationError("Invalid array argument") from error
        return stream.getvalue()


def thaw_array(payload: bytes) -> NDArray[Any]:
    """Read a fresh array, rejecting pickle and non-array containers."""
    with BytesIO(payload) as stream:
        try:
            result = np.load(stream, allow_pickle=False)
        except (ValueError, EOFError, OSError, OverflowError) as error:
            raise SerializationError("Invalid NumPy array argument payload") from error
        if isinstance(result, np.ndarray):
            return result
        result.close()
    raise SerializationError("Array argument requires a single NumPy array")
