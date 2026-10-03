"""The typed payload a realtime slot carries, and the schema that types it.

The wire format in :mod:`.protocol` is deliberately typeless: the bytes after
the header are opaque. Meaning comes from a handler schema registered in the
dispatcher's function table, which names the type and size of each argument and
each result. This module owns that type system and the encoding it implies, so
one implementation of a handler and one implementation of a caller agree on the
bytes without either reading the other's source.

The type ids, the descriptor layout, and the argument and result limits are
taken from the CUDA-Q realtime type system rather than invented, because an
interoperating implementation -- a vendor firmware image, in the case this
protocol exists for -- has to produce the same bytes. The names are FlagQuantum's
own.

**What is not here.** No handler, no dispatcher, no transport, and no timing.
A schema mismatch is a caller error caught here by name, not a runtime decode
that quietly reinterprets bytes.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .protocol import RealtimeProtocolError

__all__ = [
    "MAX_ARGUMENTS",
    "MAX_RESULTS",
    "SCHEMA_SIZE",
    "TYPE_DESCRIPTOR_SIZE",
    "HandlerSchema",
    "PayloadType",
    "TypeDescriptor",
    "decode_arguments",
    "decode_results",
    "encode_arguments",
    "encode_results",
]


class PayloadType(IntEnum):
    """The payload type identifiers a handler schema may name.

    The identifier values are part of the interchange format. Scalars occupy
    0x10 to 0x14, arrays 0x20 to 0x23, and bit-packed data 0x30.
    """

    UINT8 = 0x10
    INT32 = 0x11
    INT64 = 0x12
    FLOAT32 = 0x13
    FLOAT64 = 0x14
    ARRAY_UINT8 = 0x20
    ARRAY_INT32 = 0x21
    ARRAY_FLOAT32 = 0x22
    ARRAY_FLOAT64 = 0x23
    BIT_PACKED = 0x30


#: Every multi-byte integer is little-endian and every float is IEEE 754, so the
#: element format is one character and the array format is a repeat count plus
#: that character.
_SCALAR_STRUCT: dict[PayloadType, str] = {
    PayloadType.UINT8: "<B",
    PayloadType.INT32: "<i",
    PayloadType.INT64: "<q",
    PayloadType.FLOAT32: "<f",
    PayloadType.FLOAT64: "<d",
}
_ELEMENT_FORMAT: dict[PayloadType, str] = {
    PayloadType.ARRAY_UINT8: "B",
    PayloadType.ARRAY_INT32: "i",
    PayloadType.ARRAY_FLOAT32: "f",
    PayloadType.ARRAY_FLOAT64: "d",
}
_ELEMENT_WIDTH: dict[PayloadType, int] = {
    payload_type: struct.calcsize(f"<{element}")
    for payload_type, element in _ELEMENT_FORMAT.items()
}
_SCALAR_WIDTH: dict[PayloadType, int] = {
    payload_type: struct.calcsize(fmt) for payload_type, fmt in _SCALAR_STRUCT.items()
}

#: A descriptor is one type byte, three reserved bytes, a size, and an element
#: count. A schema is a four-byte header and fixed arrays of eight arguments and
#: four results, so its image is the same size for every handler.
TYPE_DESCRIPTOR_SIZE = 12
_SCHEMA_HEADER_SIZE = 4
MAX_ARGUMENTS = 8
MAX_RESULTS = 4
SCHEMA_SIZE = _SCHEMA_HEADER_SIZE + TYPE_DESCRIPTOR_SIZE * (MAX_ARGUMENTS + MAX_RESULTS)

_TYPE_DESCRIPTOR = struct.Struct("<B3xII")
_SCHEMA_HEADER = struct.Struct("<BBH")


@dataclass(frozen=True, slots=True)
class TypeDescriptor:
    """The type, byte size, and element count of one argument or result.

    `num_elements` is unused and pinned to one for a scalar, the element count
    for an array, and the bit count for bit-packed data. The size and the count
    are cross-checked against each other here, so a descriptor that describes an
    impossible buffer is refused before it is ever used to slice one.
    """

    type_id: PayloadType
    size_bytes: int
    num_elements: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.type_id, PayloadType):
            raise RealtimeProtocolError(
                f"payload type {self.type_id!r} is not a PayloadType"
            )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise RealtimeProtocolError("size_bytes must be a non-negative integer")
        if type(self.num_elements) is not int or self.num_elements < 0:
            raise RealtimeProtocolError("num_elements must be a non-negative integer")
        if self.type_id in _SCALAR_WIDTH:
            if self.num_elements != 1:
                raise RealtimeProtocolError(
                    f"a scalar {self.type_id.name} payload holds one element, not "
                    f"{self.num_elements}"
                )
            expected = _SCALAR_WIDTH[self.type_id]
        elif self.type_id in _ELEMENT_WIDTH:
            expected = self.num_elements * _ELEMENT_WIDTH[self.type_id]
            if self.num_elements == 0 and self.size_bytes != 0:
                raise RealtimeProtocolError(
                    f"an empty {self.type_id.name} payload holds no bytes"
                )
        else:
            expected = math.ceil(self.num_elements / 8)
        if self.size_bytes != expected:
            raise RealtimeProtocolError(
                f"a {self.type_id.name} payload of {self.num_elements} elements "
                f"occupies {expected} bytes, not {self.size_bytes}"
            )

    @property
    def is_bit_packed(self) -> bool:
        """Whether this descriptor counts bits rather than elements."""
        return self.type_id is PayloadType.BIT_PACKED

    def encode(self) -> bytes:
        """Return the fixed-layout descriptor image."""
        return _TYPE_DESCRIPTOR.pack(
            int(self.type_id), self.size_bytes, self.num_elements
        )

    @classmethod
    def decode(cls, image: bytes) -> TypeDescriptor:
        """Read a descriptor, refusing a type id this protocol does not define."""
        if len(image) < TYPE_DESCRIPTOR_SIZE:
            raise RealtimeProtocolError(
                f"a type descriptor is {TYPE_DESCRIPTOR_SIZE} bytes, not {len(image)}"
            )
        raw_type, size_bytes, num_elements = _TYPE_DESCRIPTOR.unpack_from(image)
        try:
            type_id = PayloadType(raw_type)
        except ValueError:
            raise RealtimeProtocolError(
                f"payload type 0x{raw_type:02x} is not a realtime payload type"
            ) from None
        return cls(type_id, size_bytes, num_elements)


@dataclass(frozen=True, slots=True)
class HandlerSchema:
    """What a handler accepts and returns, argument by argument.

    The dispatcher's function table holds one of these per function id and the
    caller's compiled call site must match it. The protocol has no runtime type
    check on the wire, so this record is the only place the agreement is
    written down.
    """

    args: tuple[TypeDescriptor, ...] = ()
    results: tuple[TypeDescriptor, ...] = ()

    def __post_init__(self) -> None:
        for field_name, descriptors, limit in (
            ("argument", self.args, MAX_ARGUMENTS),
            ("result", self.results, MAX_RESULTS),
        ):
            if len(descriptors) > limit:
                raise RealtimeProtocolError(
                    f"a handler declares at most {limit} {field_name} descriptors, "
                    f"not {len(descriptors)}"
                )
            for index, descriptor in enumerate(descriptors):
                if not isinstance(descriptor, TypeDescriptor):
                    raise RealtimeProtocolError(
                        f"{field_name} {index} is not a TypeDescriptor"
                    )

    @property
    def num_args(self) -> int:
        """How many arguments the handler accepts."""
        return len(self.args)

    @property
    def num_results(self) -> int:
        """How many results the handler returns."""
        return len(self.results)

    @property
    def argument_bytes(self) -> int:
        """How many payload bytes a well-formed call occupies."""
        return sum(descriptor.size_bytes for descriptor in self.args)

    @property
    def result_bytes(self) -> int:
        """How many payload bytes a well-formed response occupies."""
        return sum(descriptor.size_bytes for descriptor in self.results)

    def encode(self) -> bytes:
        """Return the fixed-layout schema image.

        The protocol does not put the schema on the wire -- out-of-band
        agreement is the point -- but this image is how a Python-declared
        schema reaches an out-of-process dispatcher that reads the same fixed
        struct. Unused slots are zero, and a reader ignores them.
        """
        unused = bytes(TYPE_DESCRIPTOR_SIZE)
        image = _SCHEMA_HEADER.pack(self.num_args, self.num_results, 0)
        image += b"".join(descriptor.encode() for descriptor in self.args)
        image += unused * (MAX_ARGUMENTS - self.num_args)
        image += b"".join(descriptor.encode() for descriptor in self.results)
        image += unused * (MAX_RESULTS - self.num_results)
        return image

    @classmethod
    def decode(cls, image: bytes) -> HandlerSchema:
        """Read a schema image, refusing one that is not the declared size."""
        if len(image) != SCHEMA_SIZE:
            raise RealtimeProtocolError(
                f"a handler schema image is {SCHEMA_SIZE} bytes, not {len(image)}"
            )
        num_args, num_results, _ = _SCHEMA_HEADER.unpack_from(image)
        if num_args > MAX_ARGUMENTS or num_results > MAX_RESULTS:
            raise RealtimeProtocolError(
                f"handler schema declares {num_args} arguments and {num_results} "
                f"results, beyond the {MAX_ARGUMENTS} and {MAX_RESULTS} this "
                f"protocol defines"
            )
        offset = _SCHEMA_HEADER_SIZE
        arguments = tuple(
            TypeDescriptor.decode(image[offset + index * TYPE_DESCRIPTOR_SIZE :])
            for index in range(num_args)
        )
        offset += MAX_ARGUMENTS * TYPE_DESCRIPTOR_SIZE
        results = tuple(
            TypeDescriptor.decode(image[offset + index * TYPE_DESCRIPTOR_SIZE :])
            for index in range(num_results)
        )
        return cls(args=arguments, results=results)


def encode_arguments(schema: HandlerSchema, values: Sequence[Any]) -> bytes:
    """Lay `values` out in schema order, with no padding or delimiters."""
    return _encode(schema.args, values, owner="argument")


def encode_results(schema: HandlerSchema, values: Sequence[Any]) -> bytes:
    """Lay result `values` out in schema order, with no padding or delimiters."""
    return _encode(schema.results, values, owner="result")


def decode_arguments(schema: HandlerSchema, payload: bytes) -> tuple[Any, ...]:
    """Read one argument per descriptor from `payload`."""
    return _decode(
        schema.args, payload, owner="argument", expected=schema.argument_bytes
    )


def decode_results(schema: HandlerSchema, payload: bytes) -> tuple[Any, ...]:
    """Read one result per descriptor from `payload`."""
    return _decode(
        schema.results, payload, owner="result", expected=schema.result_bytes
    )


def _encode(
    descriptors: tuple[TypeDescriptor, ...], values: Sequence[Any], *, owner: str
) -> bytes:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise RealtimeProtocolError(
            f"{owner} values must be a sequence of one value per descriptor"
        )
    if len(values) != len(descriptors):
        raise RealtimeProtocolError(
            f"the handler declares {len(descriptors)} {owner}s but "
            f"{len(values)} were supplied"
        )
    chunks: list[bytes] = []
    for index, (descriptor, value) in enumerate(zip(descriptors, values, strict=True)):
        chunks.append(_encode_one(descriptor, value, owner=f"{owner} {index}"))
    return b"".join(chunks)


def _decode(
    descriptors: tuple[TypeDescriptor, ...],
    payload: bytes,
    *,
    owner: str,
    expected: int,
) -> tuple[Any, ...]:
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise RealtimeProtocolError(f"a {owner} payload must be bytes")
    if len(payload) != expected:
        raise RealtimeProtocolError(
            f"the handler declares {expected} {owner} bytes but the payload holds "
            f"{len(payload)}"
        )
    values: list[Any] = []
    offset = 0
    for index, descriptor in enumerate(descriptors):
        chunk = bytes(payload[offset : offset + descriptor.size_bytes])
        values.append(_decode_one(descriptor, chunk, owner=f"{owner} {index}"))
        offset += descriptor.size_bytes
    return tuple(values)


def _encode_one(descriptor: TypeDescriptor, value: Any, *, owner: str) -> bytes:
    if descriptor.is_bit_packed:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise RealtimeProtocolError(
                f"{owner} is bit-packed and must be a sequence of bits"
            )
        if len(value) != descriptor.num_elements:
            raise RealtimeProtocolError(
                f"{owner} is bit-packed for {descriptor.num_elements} bits but "
                f"{len(value)} were supplied"
            )
        packed = bytearray(descriptor.size_bytes)
        for position, bit in enumerate(value):
            if type(bit) is not bool and not (isinstance(bit, int) and bit in (0, 1)):
                raise RealtimeProtocolError(
                    f"{owner} bit {position} is neither 0 nor 1"
                )
            if bit:
                packed[position // 8] |= 1 << (position % 8)
        return bytes(packed)
    if descriptor.type_id in _SCALAR_STRUCT:
        return _encode_scalar(descriptor, value, owner=owner)
    return _encode_array(descriptor, value, owner=owner)


def _decode_one(descriptor: TypeDescriptor, chunk: bytes, *, owner: str) -> Any:
    if descriptor.is_bit_packed:
        return tuple(
            bool(chunk[position // 8] >> (position % 8) & 1)
            for position in range(descriptor.num_elements)
        )
    if descriptor.type_id in _SCALAR_STRUCT:
        return struct.unpack(_SCALAR_STRUCT[descriptor.type_id], chunk)[0]
    count = descriptor.num_elements
    return struct.unpack(f"<{count}{_ELEMENT_FORMAT[descriptor.type_id]}", chunk)


def _encode_scalar(descriptor: TypeDescriptor, value: Any, *, owner: str) -> bytes:
    format_string = _SCALAR_STRUCT[descriptor.type_id]
    if (
        descriptor.type_id is PayloadType.FLOAT32
        or descriptor.type_id is PayloadType.FLOAT64
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RealtimeProtocolError(
                f"{owner} is a {descriptor.type_id.name} and must be a number"
            )
    elif isinstance(value, bool) or not isinstance(value, int):
        raise RealtimeProtocolError(
            f"{owner} is a {descriptor.type_id.name} and must be an integer"
        )
    try:
        return struct.pack(format_string, value)
    except struct.error as error:
        raise RealtimeProtocolError(
            f"{owner} does not fit a {descriptor.type_id.name}: {error}"
        ) from None


def _encode_array(descriptor: TypeDescriptor, value: Any, *, owner: str) -> bytes:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise RealtimeProtocolError(
            f"{owner} is a {descriptor.type_id.name} and must be a sequence"
        )
    if len(value) != descriptor.num_elements:
        raise RealtimeProtocolError(
            f"{owner} is a {descriptor.type_id.name} of {descriptor.num_elements} "
            f"elements but {len(value)} were supplied"
        )
    element = _ELEMENT_FORMAT[descriptor.type_id]
    try:
        return struct.pack(f"<{len(value)}{element}", *value)
    except struct.error as error:
        raise RealtimeProtocolError(
            f"{owner} does not fit a {descriptor.type_id.name}: {error}"
        ) from None
