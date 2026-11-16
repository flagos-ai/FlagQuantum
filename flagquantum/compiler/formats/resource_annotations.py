# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Field layout adapted from Qualtran v0.7.0 protos/annotations.proto.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/protos/annotations.proto

"""Interchange for standalone legacy three-count resource annotations."""

from functools import cache
from importlib import import_module
from typing import Any, cast

from ...errors import SerializationError
from .._component_trace.resource_model import GateCost

_FIELDS = ("clifford", "rotations", "t")


@cache
def _annotation_factory() -> Any:
    """Build the wire schema in its own lazy descriptor pool."""
    try:
        descriptors = import_module("google.protobuf.descriptor_pb2")
        pool_module = import_module("google.protobuf.descriptor_pool")
        messages = import_module("google.protobuf.message_factory")
    except ImportError as error:
        raise ImportError(
            "Legacy resource annotation interchange requires protobuf"
        ) from error

    schema = descriptors.FileDescriptorProto(
        name="flagquantum/resource_annotation.proto",
        package="flagquantum",
        syntax="proto3",
    )
    message = schema.message_type.add(name="ResourceAnnotation")
    field_proto = descriptors.FieldDescriptorProto
    for number, name in enumerate(_FIELDS, start=1):
        message.field.add(
            name=name,
            number=number,
            type=field_proto.TYPE_INT64,
            label=field_proto.LABEL_OPTIONAL,
        )
    registered = pool_module.DescriptorPool().Add(schema)
    return messages.GetMessageClass(
        registered.message_types_by_name["ResourceAnnotation"]
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
    if not isinstance(data, bytes):
        raise TypeError("Resource annotation requires bytes")
    factory = _annotation_factory()
    decoding = import_module("google.protobuf.message")
    try:
        annotation = factory.FromString(data)
    except decoding.DecodeError as error:
        raise SerializationError("Invalid resource annotation payload") from error
    return GateCost(
        clifford=annotation.clifford,
        rotation=annotation.rotations,
        t=annotation.t,
    )
