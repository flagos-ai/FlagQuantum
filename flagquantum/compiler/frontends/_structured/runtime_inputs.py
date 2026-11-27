"""Validate hybrid runtime inputs and hash their structural descriptors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from flagquantum.compiler.frontends._structured.diagnostics import SpecializationError
from flagquantum.compiler.frontends._structured.model import IRType


def _normalized_dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _runtime_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError):
        return None


def _runtime_device_class(value: Any) -> str:
    device = getattr(value, "device", None)
    if device is None:
        return "python"
    return str(getattr(device, "type", device))


def _input_descriptor(value: Any, declared: IRType) -> dict[str, Any]:
    shape = _runtime_shape(value)
    dtype = _normalized_dtype(value)
    if declared.kind == "tensor":
        expected_dtype, expected_shape = declared.parameters
        if shape is None:
            raise SpecializationError(
                "input.tensor", "tensor input must expose a concrete shape"
            )
        if len(shape) != len(expected_shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(shape, expected_shape, strict=True)
        ):
            raise SpecializationError(
                "input.shape",
                f"runtime shape {shape} does not match declared {expected_shape}",
            )
        if dtype is not None and dtype != expected_dtype:
            raise SpecializationError(
                "input.dtype",
                f"runtime dtype {dtype!r} does not match declared {expected_dtype!r}",
            )
    elif declared.kind == "scalar":
        if shape not in {None, ()}:
            raise SpecializationError(
                "input.scalar",
                f"scalar input must have shape (), got {shape}",
            )
        expected_dtype = declared.parameters[0]
        if dtype is not None and dtype != expected_dtype:
            raise SpecializationError(
                "input.dtype",
                f"runtime dtype {dtype!r} does not match declared {expected_dtype!r}",
            )
    elif declared.kind == "index":
        if shape not in {None, ()}:
            raise SpecializationError(
                "input.scalar", f"index input must have shape (), got {shape}"
            )
        if dtype is None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise SpecializationError(
                    "input.index", "index input must be an integer"
                )
        elif not dtype.startswith(("int", "uint")):
            raise SpecializationError(
                "input.index", f"index tensor must have integer dtype, got {dtype!r}"
            )
    elif declared.kind == "bool":
        if shape not in {None, ()}:
            raise SpecializationError(
                "input.scalar", f"bool input must have shape (), got {shape}"
            )
        if dtype is None and not isinstance(value, bool):
            raise SpecializationError("input.bool", "bool input must be boolean")
        if dtype is not None and dtype != "bool":
            raise SpecializationError(
                "input.bool", f"bool tensor must have bool dtype, got {dtype!r}"
            )
    else:
        raise SpecializationError(
            "input.type", f"unsupported runtime input type {declared.kind!r}"
        )
    return {
        "declared": declared.to_data(),
        "runtime_shape": None if shape is None else list(shape),
        "runtime_dtype": dtype or type(value).__name__,
        "device_class": _runtime_device_class(value),
    }


def _input_signature(values: Sequence[Any], types: Sequence[IRType]) -> str:
    descriptors = [
        _input_descriptor(value, declared)
        for value, declared in zip(values, types, strict=True)
    ]
    encoded = json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
