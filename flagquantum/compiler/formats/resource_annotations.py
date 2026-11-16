# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Field layout adapted from Qualtran v0.7.0 protos/annotations.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/annotations.proto

"""Interchange for standalone legacy three-count resource annotations."""

from typing import Any, cast

from ...errors import SerializationError
from .._component_trace.resource_model import GateCost
from ._legacy_protobuf import message_factory, parse_message

_FIELDS = ("clifford", "rotations", "t")


def _annotation_factory() -> Any:
    """Build the wire schema in its own lazy descriptor pool."""
    return message_factory(
        "ResourceAnnotation", tuple((name, "INT64") for name in _FIELDS)
    )


def serialize_resource_annotation(cost: GateCost) -> bytes:
    """Serialize legacy counts already reduced to T, rotations, and Clifford.

    Use the existing legacy budget projection for a factored gate model.
    Sign and range are the proto3 signed int64 wire contract.
    """
    if any((cost.and_bloq, cost.toffoli, cost.cswap, cost.measurement)):
        raise SerializationError("Resource annotation requires reduced legacy counts")
    factory = _annotation_factory()
    try:
        message = factory(clifford=cost.clifford, rotations=cost.rotation, t=cost.t)
        return cast(bytes, message.SerializeToString())
    except (TypeError, ValueError, OverflowError) as error:
        raise SerializationError(
            "Resource annotation counts must be signed int64"
        ) from error


def deserialize_resource_annotation(data: bytes) -> GateCost:
    """Decode standalone annotation bytes into the three corresponding counts."""
    annotation = parse_message(_annotation_factory(), data, "Resource annotation")
    return GateCost(
        clifford=annotation.clifford,
        rotation=annotation.rotations,
        t=annotation.t,
    )
