"""Private, lazy message factories for standalone legacy wire formats."""

from functools import cache
from importlib import import_module
from typing import Any

from ...errors import SerializationError


@cache
def message_factory(
    name: str, fields: tuple[tuple[str, str], ...], *, oneof: str | None = None
) -> Any:
    """Build a private proto3 descriptor using protobuf's runtime API."""
    try:
        descriptors = import_module("google.protobuf.descriptor_pb2")
        pools = import_module("google.protobuf.descriptor_pool")
        messages = import_module("google.protobuf.message_factory")
    except ImportError as error:
        raise ImportError("Legacy protobuf interchange requires protobuf") from error

    schema = descriptors.FileDescriptorProto(
        name=f"flagquantum/{name}.proto", package="flagquantum", syntax="proto3"
    )
    message = schema.message_type.add(name=name)
    if oneof is not None:
        message.oneof_decl.add(name=oneof)
    proto = descriptors.FieldDescriptorProto
    for number, (field_name, field_type) in enumerate(fields, start=1):
        field = message.field.add(
            name=field_name,
            number=number,
            type=getattr(proto, f"TYPE_{field_type}"),
            label=proto.LABEL_OPTIONAL,
        )
        if oneof is not None:
            field.oneof_index = 0
    descriptor = pools.DescriptorPool().Add(schema)
    return messages.GetMessageClass(descriptor.message_types_by_name[name])


def parse_message(factory: Any, data: bytes, label: str) -> Any:
    """Parse bytes and preserve the decoding exception as the cause."""
    if not isinstance(data, bytes):
        raise TypeError(f"{label} requires bytes")
    decoding = import_module("google.protobuf.message")
    try:
        return factory.FromString(data)
    except decoding.DecodeError as error:
        raise SerializationError(f"Invalid {label.lower()} payload") from error
