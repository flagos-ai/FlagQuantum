"""Private, lazy message factories for standalone legacy wire formats."""

from functools import cache
from importlib import import_module
from typing import Any

from ...errors import SerializationError


@cache
def _protobuf_runtime() -> tuple[Any, Any, Any]:
    """Load the optional descriptor backend only when a wire adapter is used."""
    try:
        descriptors = import_module("google.protobuf.descriptor_pb2")
        pools = import_module("google.protobuf.descriptor_pool")
        messages = import_module("google.protobuf.message_factory")
    except ImportError as error:
        raise ImportError("Legacy protobuf interchange requires protobuf") from error

    return descriptors, pools, messages


def _register_dependency(pool: Any, file: Any, registered: set[str]) -> None:
    if file.name in registered:
        return
    for dependency in file.dependencies:
        _register_dependency(pool, dependency, registered)
    pool.AddSerializedFile(file.serialized_pb)
    registered.add(file.name)


@cache
def message_factory(
    name: str,
    fields: tuple[tuple[str, str | type[Any]], ...],
    *,
    oneof: str | None = None,
    optional: tuple[str, ...] = (),
    repeated: tuple[str, ...] = (),
    enums: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> Any:
    """Build a private proto3 descriptor using protobuf's runtime API."""
    descriptors, pools, messages = _protobuf_runtime()

    schema = descriptors.FileDescriptorProto(
        name=f"flagquantum/{name}.proto", package="flagquantum", syntax="proto3"
    )
    message = schema.message_type.add(name=name)
    enum_names = set()
    for enum_name, values in enums:
        enum_names.add(enum_name)
        enum = message.enum_type.add(name=enum_name)
        for number, value in enumerate(values):
            enum.value.add(name=value, number=number)
    if oneof is not None:
        message.oneof_decl.add(name=oneof)
    proto = descriptors.FieldDescriptorProto
    pool = pools.DescriptorPool()
    dependencies: set[str] = set()
    for number, (field_name, field_type) in enumerate(fields, start=1):
        field = message.field.add(
            name=field_name,
            number=number,
            label=(
                proto.LABEL_REPEATED if field_name in repeated else proto.LABEL_OPTIONAL
            ),
        )
        if isinstance(field_type, str):
            if field_type in enum_names:
                field.type = proto.TYPE_ENUM
                field.type_name = f".flagquantum.{name}.{field_type}"
            else:
                field.type = getattr(proto, f"TYPE_{field_type}")
        else:
            nested = field_type.DESCRIPTOR
            field.type = proto.TYPE_MESSAGE
            field.type_name = "." + nested.full_name
            dependency = nested.file
            _register_dependency(pool, dependency, dependencies)
            if dependency.name not in schema.dependency:
                schema.dependency.append(dependency.name)
        if oneof is not None:
            field.oneof_index = 0
        if field_name in optional:
            field.oneof_index = len(message.oneof_decl)
            field.proto3_optional = True
            message.oneof_decl.add(name=f"_{field_name}")
    descriptor = pool.Add(schema)
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
