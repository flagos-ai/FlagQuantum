"""Bounded, data-only JSON/tensor members for the FQX 1.0 ZIP container."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import struct
import sys
import zipfile
from collections.abc import Mapping
from typing import BinaryIO, TypeAlias

import torch

from ..errors import SerializationError

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
MAX_MEMBERS = 256
MAX_JSON_BYTES = 16 * 1024**2
MAX_TENSOR_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
_CHUNK_BYTES = 1024**2
_DTYPES = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


def fields(value: object, expected: set[str], owner: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise SerializationError(f"{owner} has missing or unknown fields")
    return value


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise SerializationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise SerializationError(f"non-finite JSON constant: {value}")


def read_json(data: bytes) -> object:
    if len(data) > MAX_JSON_BYTES:
        raise SerializationError("FQX JSON member exceeds 16 MiB")
    try:
        return json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SerializationError(f"invalid FQX JSON: {error}") from error


def json_bytes(value: object) -> bytes:
    try:
        data = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise SerializationError(
            f"FQX metadata is not portable JSON: {error}"
        ) from error
    if len(data) > MAX_JSON_BYTES:
        raise SerializationError("FQX JSON member exceeds 16 MiB")
    return data


def _info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100600 << 16
    return info


def preflight(stream: BinaryIO) -> None:
    """Bound the ZIP directory before ZipFile allocates entry objects."""
    stream.seek(0, 2)
    size = stream.tell()
    if not 22 <= size <= MAX_TOTAL_BYTES + MAX_MEMBERS * 1024:
        raise SerializationError("FQX archive size exceeds its limit or is truncated")
    stream.seek(-22, 2)
    end = struct.unpack("<4s4H2IH", stream.read(22))
    (
        signature,
        disk,
        directory_disk,
        disk_count,
        count,
        directory_size,
        offset,
        comment,
    ) = end
    if (
        signature != b"PK\x05\x06"
        or disk
        or directory_disk
        or comment
        or disk_count != count
        or not 1 <= count <= MAX_MEMBERS
        or directory_size > _CHUNK_BYTES
        or offset + directory_size != size - 22
    ):
        raise SerializationError(
            "FQX requires a bounded single-disk ZIP without ZIP64 or comments"
        )
    stream.seek(0)
    if stream.read(4) != b"PK\x03\x04":
        raise SerializationError("FQX is not a ZIP experiment archive")
    stream.seek(0)


class Writer:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.members: dict[str, dict[str, str | int]] = {}
        self.total = 0
        self.tensors = 0
        self.nodes = 0

    def _reserve(self, name: str, size: int) -> None:
        if name in self.members or len(self.members) >= MAX_MEMBERS - 1:
            raise SerializationError("FQX has duplicate or too many members")
        self.total += size
        if self.total > MAX_TOTAL_BYTES - MAX_JSON_BYTES:
            raise SerializationError("FQX exceeds the 256 MiB total limit")

    def add_json(self, name: str, value: object) -> None:
        data = json_bytes(value)
        self._reserve(name, len(data))
        self.archive.writestr(_info(name), data)
        self.members[name] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def _tensor(self, value: torch.Tensor) -> JSONValue:
        dtype = str(value.dtype).removeprefix("torch.")
        if (
            sys.byteorder != "little"
            or dtype not in _DTYPES
            or value.layout != torch.strided
            or value.device.type == "meta"
        ):
            raise SerializationError(
                "FQX requires supported strided tensors on a little-endian host"
            )
        size = value.numel() * value.element_size()
        if size > MAX_TENSOR_BYTES or value.ndim > 32:
            raise SerializationError("FQX tensor exceeds 64 MiB or 32 dimensions")
        name = f"arrays/{self.tensors:04d}.bin"
        self.tensors += 1
        self._reserve(name, size)
        # Own the snapshot. Never request an absent state or retain an autograd graph.
        tensor = (
            value.detach()
            .resolve_conj()
            .resolve_neg()
            .to(device="cpu", copy=True)
            .contiguous()
        )
        digest = hashlib.sha256()
        with self.archive.open(_info(name), "w") as output:
            for offset in range(0, size, _CHUNK_BYTES):
                block = ctypes.string_at(
                    tensor.data_ptr() + offset, min(size - offset, _CHUNK_BYTES)
                )
                output.write(block)
                digest.update(block)
        self.members[name] = {"size": size, "sha256": digest.hexdigest()}
        return {
            "kind": "tensor",
            "member": name,
            "dtype": dtype,
            "shape": list(tensor.shape),
        }

    def encode(self, value: object, depth: int = 0) -> JSONValue:
        self.nodes += 1
        if depth > 64 or self.nodes > 100_000:
            raise SerializationError(
                "FQX result nesting or item count exceeds its limit"
            )
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SerializationError("FQX JSON metadata must be finite")
            return value
        if isinstance(value, torch.Tensor):
            return self._tensor(value)
        if isinstance(value, Mapping):
            items: list[JSONValue] = []
            for key, item in value.items():
                if type(key) not in (str, int):
                    raise SerializationError(
                        "FQX mapping keys must be strings or integers"
                    )
                items.append([key, self.encode(item, depth + 1)])
            return {"kind": "mapping", "items": items}
        if isinstance(value, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [self.encode(item, depth + 1) for item in value],
            }
        raise SerializationError(
            f"FQX cannot serialize {type(value).__name__}; native handles and executable objects are not supported"
        )


class Reader:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or len(names) > MAX_MEMBERS:
            raise SerializationError("FQX has duplicate or too many members")
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise SerializationError("FQX exceeds the 256 MiB total limit")
        for info in infos:
            if (
                info.file_size > MAX_TENSOR_BYTES
                or info.flag_bits & 1
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            ):
                raise SerializationError(
                    "FQX member size, encryption or compression is unsupported"
                )
        self.used: set[str] = set()
        self.nodes = 0

    def read(self, name: str) -> bytes:
        info = self.archive.getinfo(name)
        if name.endswith(".json") and info.file_size > MAX_JSON_BYTES:
            raise SerializationError("FQX JSON member exceeds 16 MiB")
        with self.archive.open(info) as stream:
            data = stream.read(info.file_size + 1)
        if len(data) != info.file_size:
            raise SerializationError(f"FQX member length differs: {name}")
        return data

    def decode(self, value: object, depth: int = 0) -> object:
        self.nodes += 1
        if depth > 64 or self.nodes > 100_000:
            raise SerializationError(
                "FQX result nesting or item count exceeds its limit"
            )
        if value is None or type(value) in (bool, int, str, float):
            return value
        if not isinstance(value, dict):
            raise SerializationError("invalid FQX typed result value")
        kind = value.get("kind")
        if kind == "tensor":
            node = fields(value, {"kind", "member", "dtype", "shape"}, "tensor")
            name, dtype, shape = node["member"], node["dtype"], node["shape"]
            if (
                not isinstance(name, str)
                or not name.startswith("arrays/")
                or name in self.used
                or not isinstance(dtype, str)
                or dtype not in _DTYPES
            ):
                raise SerializationError("invalid or reused FQX tensor member")
            if (
                not isinstance(shape, list)
                or len(shape) > 32
                or any(type(n) is not int or not 0 <= n < 2**31 for n in shape)
            ):
                raise SerializationError("invalid FQX tensor shape")
            expected = (
                math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
            )
            if (
                expected > MAX_TENSOR_BYTES
                or expected != self.archive.getinfo(name).file_size
                or sys.byteorder != "little"
            ):
                raise SerializationError(
                    "FQX tensor shape/dtype does not match its byte length"
                )
            self.used.add(name)
            if expected == 0:
                try:
                    return torch.empty(shape, dtype=_DTYPES[dtype])
                except RuntimeError as error:
                    raise SerializationError(
                        "FQX tensor shape cannot be represented"
                    ) from error
            return (
                torch.frombuffer(bytearray(self.read(name)), dtype=_DTYPES[dtype])
                .clone()
                .reshape(shape)
            )
        node = fields(value, {"kind", "items"}, "container")
        items = node["items"]
        if not isinstance(items, list):
            raise SerializationError("FQX container items must be an array")
        if kind in ("tuple", "list"):
            decoded = [self.decode(item, depth + 1) for item in items]
            return tuple(decoded) if kind == "tuple" else decoded
        if kind == "mapping":
            mapping: dict[str | int, object] = {}
            for pair in items:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or type(pair[0]) not in (str, int)
                    or pair[0] in mapping
                ):
                    raise SerializationError("invalid or duplicate FQX mapping key")
                mapping[pair[0]] = self.decode(pair[1], depth + 1)
            return mapping
        raise SerializationError(f"unknown FQX value kind: {kind}")
