# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix py/circuit/circuit.pybind.cc file boundaries.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Convert Python paths and streams to owned physical circuit values."""

import os
from collections.abc import Callable
from io import TextIOBase
from typing import cast

from ..compiler.formats.kmb import encode_kmb
from ..compiler.formats.kmx_circuit import (
    KmxCircuit,
    read_circuit_file,
    read_kmb_circuit,
    read_kmx_circuit,
)


def read_physical_circuit(source: object, format: str = "auto") -> KmxCircuit:
    """Read a caller-owned stream or path without closing supplied streams."""
    if format not in ("auto", "kmx", "kmb"):
        raise ValueError(f"Unrecognized format {format!r}. Expected auto, kmx or kmb")
    if hasattr(source, "read"):
        content = cast(Callable[[], object], source.read)()
        if isinstance(content, str):
            if format == "kmb":
                raise ValueError("format='kmb' requires a binary file.")
            return read_kmx_circuit(content)
        data = bytes(cast(bytes, content))
    else:
        path = os.fspath(
            cast("str | bytes | os.PathLike[str] | os.PathLike[bytes]", source)
        )
        try:
            with open(path, "rb") as stream:
                data = stream.read()
        except OSError as error:
            raise ValueError(
                f"Failed to open file for reading: {os.fsdecode(path)!r}."
            ) from error
    if format == "kmb":
        return read_kmb_circuit(data)
    if format == "kmx":
        # Explicit text requests must reject binary before automatic detection.
        data.decode()
    return read_circuit_file(data)


def write_physical_circuit(
    circuit: KmxCircuit, destination: object, format: str = "kmx"
) -> None:
    """Write canonical text or binary using existing Compiler serializers."""
    if format not in ("kmx", "kmb"):
        raise ValueError(f"Unrecognized format {format!r}. Expected kmx or kmb")
    text = circuit.to_kmx()
    data = (
        ((text + "\n").encode() if text else b"")
        if format == "kmx"
        else encode_kmb(text)
    )
    if hasattr(destination, "write"):
        payload: str | bytes = (
            data.decode()
            if format == "kmx" and isinstance(destination, TextIOBase)
            else data
        )
        cast(Callable[[str | bytes], object], destination.write)(payload)
        return
    path = os.fspath(
        cast("str | bytes | os.PathLike[str] | os.PathLike[bytes]", destination)
    )
    try:
        with open(path, "wb") as stream:
            stream.write(data)
    except OSError as error:
        raise ValueError(
            f"Failed to open file for writing: {os.fsdecode(path)!r}."
        ) from error
