"""Bounded file admission and pinned record codecs; no decoder policy."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import torch

from ...errors import CapabilityError
from ._measurement_file_validation import _snapshot, _validate_r8

_FORMATS = ("01", "b8", "r8", "ptb64", "hits", "dets")
_MAX_PAYLOAD = 64 * 1024**2


def _dimensions(
    values: tuple[int | None, int | None, int | None],
) -> tuple[int, int, int]:
    if all(value is None for value in values):
        raise ValueError("specify at least one stored column dimension")
    for value in values:
        if value is not None and type(value) is not int:
            raise TypeError("stored column dimensions must be integers or None")
        if value is not None and value < 0:
            raise ValueError("stored column dimensions cannot be negative")
    nm, nd, no = (0 if value is None else value for value in values)
    if nm + nd + no == 0:
        raise CapabilityError("zero stored columns are not supported")
    if nm + nd + no > sys.maxsize:
        raise CapabilityError("stored columns exceed the native integer range")
    return nm, nd, no


def _budget(
    rows: int, dimensions: tuple[int, int, int], packed: bool, separate: bool
) -> None:
    nm, nd, no = dimensions
    columns = nm + nd + no
    stride = (columns + 7) // 8
    output = columns
    if packed:
        output = ((nm + nd + 7) // 8 + (no + 7) // 8) if separate else stride
    needed = (
        4 * rows * (stride + output) + 128 * ((columns + 1023) // 1024) * 128 + 65536
    )
    if rows > sys.maxsize or needed > _MAX_PAYLOAD:
        raise CapabilityError(
            "shot file principal payload exceeds the 64 MiB byte limit"
        )


def _backend(*, streaming: bool) -> ModuleType:
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        try:
            return import_module("flagquantum.simulation.stabilizer._embedded_stim")
        except ImportError as error:
            raise CapabilityError(
                "embedded shot reader extension is unavailable"
            ) from error
    if streaming:
        raise CapabilityError(
            "iter_shot_data_file requires an embedded FlagQuantum build"
        )
    from .engine import _stim

    sdk = _stim()
    if not isinstance(sdk, ModuleType):
        raise CapabilityError("invalid Stim SDK module")
    if sdk.__version__ != "1.16.0":
        raise CapabilityError("shot file reading requires Stim 1.16.0")
    return sdk


def _validate_arguments(
    path: str | os.PathLike[str],
    format: str,
    dimensions: tuple[int | None, int | None, int | None],
    packed: bool,
    separate: bool,
) -> tuple[Path, tuple[int, int, int]]:
    if not isinstance(format, str) or format not in _FORMATS:
        raise ValueError("shot format must be 01, b8, r8, ptb64, hits or dets")
    if type(packed) is not bool or type(separate) is not bool:
        raise TypeError("packed and separate flags must be bool")
    dims = _dimensions(dimensions)
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError("shot path must resolve to a string")
    if "://" in raw:
        raise ValueError("shot reading requires a regular local file")
    _budget(0, dims, packed, separate)
    return Path(raw), dims


def _native_count(
    native: ModuleType, path: Path, format: str, dims: tuple[int, int, int]
) -> int:
    cursor = native.open_shot_reader(str(path), format, *dims)
    try:
        count: object = native.scan_shot_reader(cursor)
        if type(count) is not int or count < 0:
            raise CapabilityError("embedded reader returned an invalid shot count")
        return count
    finally:
        native.close_shot_reader(cursor)


def _tensor(buffer: object, rows: int, columns: int, packed: bool) -> torch.Tensor:
    stride = (columns + 7) // 8 if packed else columns
    if not isinstance(buffer, bytearray) or len(buffer) != rows * stride:
        raise CapabilityError("embedded reader returned an invalid buffer")
    dtype = torch.uint8 if packed else torch.bool
    if not rows or not stride:
        return torch.empty((rows, stride), dtype=dtype)
    result = torch.frombuffer(buffer, dtype=torch.uint8).reshape(rows, stride)
    if packed:
        if columns % 8 and bool((result[:, -1] >> (columns % 8)).any()):
            raise CapabilityError("embedded reader returned nonzero packed padding")
    elif bool((result > 1).any()):
        raise CapabilityError("embedded reader returned nonbinary values")
    return result.to(dtype=dtype, copy=True)


def _native_batch(
    native: ModuleType,
    cursor: object,
    rows: int,
    dims: tuple[int, int, int],
    packed: bool,
    separate: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    nm, nd, no = dims
    result: object = native.read_shot_reader(cursor, rows, packed, separate, nm + nd)
    if (
        not isinstance(result, tuple)
        or len(result) != 3
        or type(result[2]) is not int
        or result[2] != rows
    ):
        raise CapabilityError("embedded shot count changed after validation")
    first = _tensor(result[0], rows, nm + nd if separate else nm + nd + no, packed)
    if separate:
        return first, _tensor(result[1], rows, no, packed)
    return first


def _sdk_shot_upper_bound(path: Path, format: str, dims: tuple[int, int, int]) -> int:
    """Count framing before the eager SDK can allocate its complete table."""
    columns = sum(dims)
    size = path.stat().st_size
    if format in ("b8", "ptb64"):
        unit = (columns + 7) // 8 if format == "b8" else columns * 8
        if size % unit:
            raise ValueError("shot file has an incomplete encoded record")
        if format == "b8" and columns % 8:
            offset = 0
            with path.open("rb") as stream:
                while block := stream.read(65536):
                    for value in block:
                        if offset % unit == unit - 1 and value >> (columns % 8):
                            raise ValueError("nonzero b8 shot padding")
                        offset += 1
        return size // unit * (64 if format == "ptb64" else 1)
    if format == "r8":
        with path.open("rb") as stream:
            return _validate_r8(stream, None, columns)
    # Valid text has no more than one record per line. Count an unterminated
    # final line as well (allowed by the pinned hits/dets readers). The SDK
    # still validates all content before exposing its result.
    count = 0
    last = b""
    with path.open("rb") as stream:
        while block := stream.read(65536):
            count += block.count(b"\n")
            last = block[-1:]
    return count + int(bool(last) and last != b"\n")


def _sdk_read(
    sdk: ModuleType,
    path: Path,
    format: str,
    dims: tuple[int, int, int],
    packed: bool,
    separate: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    bound = _sdk_shot_upper_bound(path, format, dims)
    _budget(bound, dims, packed, separate)
    nm, nd, no = dims
    raw: object = sdk.read_shot_data_file(
        path=str(path),
        format=format,
        num_measurements=nm,
        num_detectors=nd,
        num_observables=no,
        bit_packed=packed,
        separate_observables=separate,
    )
    widths = (nm + nd, no) if separate else (nm + nd + no,)
    parts = raw if isinstance(raw, tuple) else (raw,)
    if len(parts) != len(widths):
        raise CapabilityError("SDK returned invalid shot sections")
    tensors: list[torch.Tensor] = []
    rows: int | None = None
    for part, columns in zip(parts, widths, strict=True):
        value = torch.as_tensor(part)
        stride = (columns + 7) // 8 if packed else columns
        if (
            value.ndim != 2
            or value.shape[1] != stride
            or value.device.type != "cpu"
            or value.dtype != (torch.uint8 if packed else torch.bool)
            or len(value) > bound
        ):
            raise CapabilityError("SDK returned invalid shot data")
        if rows is not None and rows != len(value):
            raise CapabilityError("SDK returned inconsistent shot counts")
        rows = len(value)
        if packed and columns % 8 and bool((value[:, -1] >> (columns % 8)).any()):
            raise CapabilityError("SDK returned nonzero shot padding")
        tensors.append(value.clone())
    return (tensors[0], tensors[1]) if separate else tensors[0]


def _read_file(
    path: str | os.PathLike[str],
    format: str,
    dimensions: tuple[int | None, int | None, int | None],
    packed: bool,
    separate: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    source, dims = _validate_arguments(path, format, dimensions, packed, separate)
    backend = _backend(streaming=False)
    with TemporaryDirectory(prefix="fq-shot-read-") as directory:
        snapshot = Path(directory) / "input"
        _snapshot(source, snapshot)
        if not hasattr(backend, "open_shot_reader"):
            return _sdk_read(backend, snapshot, format, dims, packed, separate)
        rows = _native_count(backend, snapshot, format, dims)
        _budget(rows, dims, packed, separate)
        cursor = backend.open_shot_reader(str(snapshot), format, *dims)
        try:
            return _native_batch(backend, cursor, rows, dims, packed, separate)
        finally:
            backend.close_shot_reader(cursor)


def _iterate_file(
    path: str | os.PathLike[str],
    format: str,
    dimensions: tuple[int | None, int | None, int | None],
    packed: bool,
    separate: bool,
    batch_size: int,
) -> Generator[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None]:
    source, dims = _validate_arguments(path, format, dimensions, packed, separate)
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer")
    if not 1 <= batch_size <= 65536:
        raise ValueError("batch_size must be between 1 and 65536")
    backend = _backend(streaming=True)

    def batches() -> (
        Generator[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None]
    ):
        with TemporaryDirectory(prefix="fq-shot-read-") as directory:
            snapshot = Path(directory) / "input"
            _snapshot(source, snapshot)
            remaining = _native_count(backend, snapshot, format, dims)
            _budget(min(batch_size, remaining), dims, packed, separate)
            cursor = backend.open_shot_reader(str(snapshot), format, *dims)
            try:
                while remaining:
                    rows = min(batch_size, remaining)
                    _budget(rows, dims, packed, separate)
                    yield _native_batch(backend, cursor, rows, dims, packed, separate)
                    remaining -= rows
            finally:
                backend.close_shot_reader(cursor)

    return batches()
