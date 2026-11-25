"""Validate, encode and publish caller-owned shot data."""

from __future__ import annotations

import os
import shutil
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import torch

from ...errors import CapabilityError
from . import engine
from ._measurement_file_validation import _file_size_bound, _validate_record_file
from ._shot_data import _FORMATS, _dimensions

_MAX_PAYLOAD = 64 * 1024**2
_MAX_OUTPUT = 1024**3
_NATIVE_BATCH_SHOTS = 64


def _round_simd(value: int) -> int:
    return ((value + 1023) // 1024) * 1024


def _principal_bytes(shots: int, columns: int, packed: bool) -> int:
    stride = (columns + 7) // 8 if packed else columns
    aligned_columns = _round_simd(columns)
    aligned_shots = _round_simd(max(shots, _NATIVE_BATCH_SHOTS))
    return (
        shots * stride
        + 2 * aligned_columns * aligned_shots // 8
        + 2 * aligned_columns // 8
        + 65536
    )


def _check_budget(shots: int, columns: int, packed: bool) -> None:
    if shots > sys.maxsize or _principal_bytes(shots, columns, packed) > _MAX_PAYLOAD:
        raise CapabilityError(
            "shot data principal payload exceeds the 64 MiB byte limit"
        )


def _backend() -> ModuleType:
    marker = Path(__file__).with_name("_embedded_stim_profile.txt")
    if marker.is_file():
        try:
            backend = import_module("flagquantum.simulation.stabilizer._embedded_stim")
        except ImportError as error:
            raise CapabilityError(
                "embedded shot writer extension is unavailable"
            ) from error
        if not isinstance(backend, ModuleType) or not callable(
            getattr(backend, "write_shot_records", None)
        ):
            raise CapabilityError("invalid embedded shot writer module")
        return backend
    sdk = engine._stim()
    if not isinstance(sdk, ModuleType):
        raise CapabilityError("invalid Stim SDK module")
    if sdk.__version__ != "1.16.0":
        raise CapabilityError("shot file writing requires Stim 1.16.0")
    return sdk


def _validated_tensor(
    data: torch.Tensor,
    dims: tuple[int, int, int],
) -> tuple[int, int, bool]:
    if not isinstance(data, torch.Tensor):
        raise TypeError("shot data must be a torch.Tensor")
    if data.layout != torch.strided:
        raise CapabilityError("shot data must use dense strided storage")
    if data.device.type != "cpu":
        raise CapabilityError("shot data must be on the CPU")
    if data.dtype not in (torch.bool, torch.uint8):
        raise TypeError("shot data dtype must be torch.bool or torch.uint8")
    if data.ndim != 2:
        raise ValueError("shot data must be a two-dimensional table")
    shots, columns = data.shape
    expected_columns = sum(dims)
    packed = data.dtype == torch.uint8
    expected_width = (expected_columns + 7) // 8 if packed else expected_columns
    if columns != expected_width:
        raise ValueError("shot data shape does not match the stored column dimensions")
    _check_budget(shots, expected_columns, packed)
    byte_count = shots * expected_width
    if byte_count > sys.maxsize:
        raise CapabilityError("shot data exceeds the native integer range")
    return shots, expected_columns, packed


def _owned_data(data: torch.Tensor, columns: int, packed: bool) -> bytearray:
    shots, width = data.shape
    byte_count = shots * width
    owned = bytearray(byte_count)
    if byte_count:
        target = torch.frombuffer(owned, dtype=data.dtype).reshape(shots, width)
        target.copy_(data)
        if packed and columns % 8:
            mask = 255 ^ ((1 << (columns % 8)) - 1)
            if bool((target[:, -1] & mask).any()):
                raise ValueError("packed shot data has nonzero padding bits")
        elif not packed and bool(
            (torch.frombuffer(owned, dtype=torch.uint8) > 1).any()
        ):
            raise ValueError("dense shot data contains a nonbinary value")
    return owned


def _write_backend(
    backend: ModuleType,
    owned: bytearray,
    shots: int,
    columns: int,
    packed: bool,
    staging: Path,
    format: str,
    dims: tuple[int, int, int],
) -> None:
    nm, nd, no = dims
    if hasattr(backend, "write_shot_records"):
        try:
            backend.write_shot_records(
                owned, shots, packed, str(staging), format, nm, nd, no
            )
        except OSError:
            raise
        except Exception as error:
            raise OSError("shot-data backend failed while encoding output") from error
        return
    try:
        import numpy as np
    except ImportError as error:
        raise CapabilityError("Stim SDK shot writing requires NumPy") from error
    width = (columns + 7) // 8 if packed else columns
    array = np.frombuffer(owned, dtype=np.uint8 if packed else np.bool_).reshape(
        shots, width
    )
    try:
        backend.write_shot_data_file(
            data=array,
            path=str(staging),
            format=format,
            num_measurements=nm,
            num_detectors=nd,
            num_observables=no,
        )
    except OSError:
        raise
    except Exception as error:
        raise OSError("Stim SDK failed while encoding shot data") from error


def _write_file(
    data: torch.Tensor,
    path: str | os.PathLike[str],
    format: str,
    dimensions: tuple[int | None, int | None, int | None],
) -> None:
    if not isinstance(format, str) or format not in _FORMATS:
        raise ValueError("shot format must be 01, b8, r8, ptb64, hits or dets")
    dims = _dimensions(dimensions)
    if dims[0] and (dims[1] or dims[2]):
        raise ValueError(
            "measurements cannot be mixed with detector or observable columns"
        )
    shots, columns, packed = _validated_tensor(data, dims)
    if format == "ptb64" and shots % 64:
        raise ValueError("ptb64 requires shots to be a multiple of 64")
    output_bound = _file_size_bound(shots, columns, format)
    if output_bound > _MAX_OUTPUT:
        raise CapabilityError("shot data output exceeds the 1 GiB byte limit")
    owned = _owned_data(data, columns, packed)
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError("shot output path must resolve to a string")
    if "://" in raw_path:
        raise ValueError("shot writing requires a regular local path")
    destination = Path(raw_path)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(str(parent))
    if shutil.disk_usage(parent).free < output_bound:
        raise OSError(f"insufficient free space for {output_bound} shot-data bytes")

    backend = _backend()
    with TemporaryDirectory(prefix=".fq-shot-write-", dir=parent) as directory:
        staging = Path(directory) / "records"
        _write_backend(backend, owned, shots, columns, packed, staging, format, dims)
        sections = (("M", dims[0]),) if dims[0] else (("D", dims[1]), ("L", dims[2]))
        _validate_record_file(
            staging,
            shots,
            columns,
            format,
            output_bound,
            sections=sections,
        )
        os.link(staging, parent / destination.name)
