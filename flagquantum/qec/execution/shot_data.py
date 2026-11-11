"""Read and write measurement, detector and logical records as CPU tensors."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Literal

import torch

from ...simulation.stabilizer._shot_data import _iterate_file, _read_file
from ...simulation.stabilizer._shot_data_write import _write_file

__all__ = ("read_shot_data_file", "iter_shot_data_file", "write_shot_data_file")


def read_shot_data_file(
    *,
    path: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    num_measurements: int | None = None,
    num_detectors: int | None = None,
    num_observables: int | None = None,
    separate_observables: bool = False,
    bit_packed: bool = False,
    bit_pack: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Read stored M/D/L columns in order into owned CPU bool or uint8 tensors.

    Specify at least one dimension; their sum must be positive. With separate
    observables, return (measurement-plus-detector, logical) tensors, packing
    each independently. bit_pack is the upstream alias for bit_packed (OR).
    The 1 GiB snapshot and 64 MiB principal-payload guards are not RSS limits.

    Raises:
        TypeError: Dimensions or flags have invalid types.
        ValueError: Dimensions, framing, padding or format are invalid.
        OSError: Input admission, snapshot or file I/O fails.
        CapabilityError: A resource limit or the selected codec is unavailable.

    Examples:
        >>> from importlib.util import find_spec
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> if find_spec("stim") is not None:
        ...     with TemporaryDirectory() as directory:
        ...         path = Path(directory) / "events.b8"
        ...         _ = path.write_bytes(b"\\x00\\x01")
        ...         rows = read_shot_data_file(path=path, format="b8", num_detectors=1)
        ...         assert isinstance(rows, torch.Tensor)
        ...         assert rows.tolist() == [[False], [True]]
    """
    if type(bit_packed) is not bool or type(bit_pack) is not bool:
        raise TypeError("packed flags must be bool")
    return _read_file(
        path,
        format,
        (num_measurements, num_detectors, num_observables),
        bit_packed or bit_pack,
        separate_observables,
    )


def iter_shot_data_file(
    *,
    path: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    num_measurements: int | None = None,
    num_detectors: int | None = None,
    num_observables: int | None = None,
    separate_observables: bool = False,
    bit_packed: bool = False,
    batch_size: int = 1024,
) -> Generator[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None]:
    """Read owned batches after validating the complete stable input snapshot.

    Requires an embedded build, with 1 <= batch_size <= 65,536. Shapes and
    independent logical packing match read_shot_data_file. Empty files yield
    no batches. Use contextlib.closing when a consumer may stop early.
    Retaining prior batches is the caller's responsibility and memory cost.

    Raises:
        TypeError: Dimensions, flags or batch_size have invalid types.
        ValueError: Framing, dimensions, padding or batch_size are invalid.
        OSError: Snapshot or file I/O fails.
        CapabilityError: The embedded reader is unavailable or a guard is exceeded.

    Examples:
        >>> from contextlib import closing
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import flagquantum.simulation.stabilizer as stabilizer
        >>> if Path(stabilizer.__file__).with_name("_embedded_stim_profile.txt").is_file():
        ...     with TemporaryDirectory() as directory:
        ...         path = Path(directory) / "events.b8"
        ...         _ = path.write_bytes(b"\\x00\\x01")
        ...         with closing(iter_shot_data_file(
        ...             path=path, format="b8", num_detectors=1, batch_size=1,
        ...         )) as batches:
        ...             assert [batch.tolist() for batch in batches] == [[[False]], [[True]]]
    """
    return _iterate_file(
        path,
        format,
        (num_measurements, num_detectors, num_observables),
        bit_packed,
        separate_observables,
        batch_size,
    )


def write_shot_data_file(
    *,
    data: torch.Tensor,
    path: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    num_measurements: int | None = None,
    num_detectors: int | None = None,
    num_observables: int | None = None,
) -> None:
    """Write CPU bool or bit-packed uint8 shot records without replacing a file.

    Specify measurement columns alone, or detector and observable columns.
    Packed uint8 rows use little-endian bit order and must have zero padding.
    The 64 MiB principal-payload and 1 GiB output guards are not RSS limits.

    Raises:
        TypeError: The tensor dtype or dimensions have invalid types.
        ValueError: Shapes, sections, values, padding or format are invalid.
        OSError: File admission, encoding or publication fails.
        CapabilityError: A selected backend is unavailable or a byte limit is exceeded.

    Examples:
        >>> from importlib.util import find_spec
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import torch
        >>> import flagquantum.simulation.stabilizer as stabilizer
        >>> if find_spec("stim") is not None or Path(stabilizer.__file__).with_name("_embedded_stim_profile.txt").is_file():
        ...     with TemporaryDirectory() as directory:
        ...         path = Path(directory) / "detectors.b8"
        ...         rows = torch.tensor([[False, True], [True, False]])
        ...         write_shot_data_file(
        ...             data=rows, path=path, format="b8", num_detectors=2,
        ...         )
        ...         assert path.read_bytes() == b"\\x02\\x01"
    """
    _write_file(
        data,
        path,
        format,
        (num_measurements, num_detectors, num_observables),
    )
