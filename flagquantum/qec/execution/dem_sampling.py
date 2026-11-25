"""Reusable, bounded sampling and replay of independent detector error models."""

from __future__ import annotations

import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread, current_thread
from typing import TYPE_CHECKING, Literal, NoReturn, SupportsIndex

import torch

from ...errors import CapabilityError
from ...simulation.stabilizer import _dem_sampling as _engine
from ...simulation.stabilizer._measurement_file_validation import (
    _file_size_bound,
    _snapshot,
    _validate_record_file,
)
from ...simulation.stabilizer._validation import _validate_seed

if TYPE_CHECKING:
    from ..dem import DetectorErrorModel

__all__ = ("CompiledDemSampler",)

_Format = Literal["01", "b8", "r8", "ptb64", "hits", "dets"]
_MAX_FILE = 1024**3
_MAX_DIMENSION = 65536
_MAX_TARGETS = 262144
_MAX_TEXT = 1024**2


def _shots(shots: int) -> int:
    if type(shots) is not int:
        raise TypeError("shots must be a nonnegative integer")
    if shots < 0:
        raise ValueError("shots must be nonnegative")
    if shots > sys.maxsize:
        raise CapabilityError("shots exceed the native integer range")
    return shots


def _path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or "://" in raw or "\x00" in raw:
        raise ValueError("DEM files require local string paths")
    return Path(raw)


def _replay(data: torch.Tensor | None, shots: int, errors: int) -> bool | None:
    if data is None:
        return None
    if not isinstance(data, torch.Tensor):
        raise TypeError("recorded_errors_to_replay must be a CPU tensor")
    if data.device.type != "cpu" or data.layout != torch.strided:
        raise CapabilityError("DEM replay requires dense CPU storage")
    if data.dtype not in (torch.bool, torch.uint8):
        raise TypeError("DEM replay dtype must be bool or uint8")
    packed = data.dtype == torch.uint8
    columns = (errors + 7) // 8 if packed else errors
    if data.ndim != 2 or data.shape != (shots, columns):
        raise ValueError(
            "DEM replay has the wrong shot or canonical error-column count"
        )
    return packed


class CompiledDemSampler:
    """Sample an owned independent DEM with optional error recording and replay.

    Results use CPU tensors and canonical ``model.errors`` column order. Arrays
    are owned, packed streams are independent, and files never replace targets.
    Handles belong to their creating thread/process and cannot be serialized.
    The 64 MiB principal-data guard does not bound process RSS or model heap.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.qec import DemError, DetectorErrorModel
        >>> if find_spec("stim") is not None:
        ...     model = DetectorErrorModel(1, 1, (DemError(1.0, (0,), (0,)),))
        ...     detectors, logicals, errors = model.compile_sampler().sample(
        ...         2, bit_packed=True, return_errors=True)
        ...     assert detectors.tolist() == logicals.tolist() == [[1], [1]]
    """

    __slots__ = ("_stream", "_widths", "_pid", "_thread")
    _stream: _engine._Stream | None
    _widths: tuple[int, int, int]
    _pid: int
    _thread: Thread

    def __init__(self) -> None:
        raise TypeError("create a sampler with model.compile_sampler()")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("compiled DEM samplers cannot be serialized")

    def _active(self) -> _engine._Stream:
        if self._pid != os.getpid() or self._thread is not current_thread():
            raise RuntimeError("DEM sampler belongs to its creating thread and process")
        if self._stream is None:
            raise RuntimeError("DEM sampler is invalidated")
        return self._stream

    def sample(
        self,
        shots: int,
        *,
        bit_packed: bool = False,
        return_errors: bool = False,
        recorded_errors_to_replay: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return detector, observable and optional error tables from one draw.

        Replay dtype determines its packing independently of output packing.
        Error column ``i`` is ``model.errors[i]``, after model canonicalization.
        Zero shots return empty tables without advancing RNG. Identical seeds
        require the same backend, build, SIMD and call sequence for repetition.

        Examples:
            >>> from importlib.util import find_spec
            >>> from flagquantum.qec import DemError, DetectorErrorModel
            >>> if find_spec("stim") is not None:
            ...     s = DetectorErrorModel(1, 0, (DemError(1.0, (0,)),)).compile_sampler()
            ...     d, o, e = s.sample(1, return_errors=True)
            ...     assert d.tolist() == [[True]] and o.shape == (1, 0)
        """
        stream = self._active()
        shots = _shots(shots)
        if type(bit_packed) is not bool or type(return_errors) is not bool:
            raise TypeError("DEM sampling flags must be bool")
        replay_packed = _replay(recorded_errors_to_replay, shots, self._widths[2])
        _engine._check_budget(
            shots, self._widths, bit_packed, return_errors, replay_packed
        )
        replay = (
            None
            if recorded_errors_to_replay is None
            else recorded_errors_to_replay.clone(memory_format=torch.contiguous_format)
        )
        dtype = torch.uint8 if bit_packed else torch.bool
        results: list[torch.Tensor | None] = [
            torch.empty((shots, (n + 7) // 8 if bit_packed else n), dtype=dtype)
            for n in self._widths[: 3 if return_errors else 2]
        ]
        if not return_errors:
            results.append(None)
        try:
            for start in range(0, shots, 1024):
                count = min(1024, shots - start)
                block = stream.sample(
                    count,
                    packed=bit_packed,
                    errors=None if replay is None else replay[start : start + count],
                    return_errors=return_errors,
                )
                if not isinstance(block, tuple) or len(block) != 3:
                    raise CapabilityError("backend returned an invalid DEM tuple")
                for index, (target, value) in enumerate(
                    zip(results, block, strict=True)
                ):
                    if target is None:
                        if value is not None:
                            raise CapabilityError(
                                "backend returned unrequested error data"
                            )
                    elif (
                        not isinstance(value, torch.Tensor)
                        or value.dtype != dtype
                        or value.device.type != "cpu"
                        or value.shape != target[start : start + count].shape
                    ):
                        raise CapabilityError("backend returned an invalid DEM table")
                    else:
                        if bit_packed and self._widths[index] % 8:
                            padding = value[:, -1] >> (self._widths[index] % 8)
                            if bool(padding.any()):
                                raise CapabilityError(
                                    "backend returned nonzero DEM padding"
                                )
                        target[start : start + count].copy_(value)
        except BaseException:
            self._stream = None
            raise
        det, obs, err = results
        assert det is not None and obs is not None
        return det, obs, err

    def sample_write(
        self,
        shots: int,
        *,
        det_out_file: str | os.PathLike[str] | None,
        det_out_format: _Format = "01",
        obs_out_file: str | os.PathLike[str] | None,
        obs_out_format: _Format = "01",
        err_out_file: str | os.PathLike[str] | None = None,
        err_out_format: _Format = "01",
        replay_err_in_file: str | os.PathLike[str] | None = None,
        replay_err_in_format: _Format = "01",
    ) -> None:
        """Stream detector/observable/error records, optionally replaying a file.

        Six formats are supported; PTB64 requires a multiple of 64 shots. Output
        totals are bounded by 1 GiB and an immutable replay snapshot separately
        by 1 GiB. Outputs publish sequentially, without cross-file atomicity.
        Existing targets are never replaced. Execution/publication errors
        invalidate the handle and remove only this call's owned output inodes.

        Examples:
            >>> from pathlib import Path
            >>> from tempfile import TemporaryDirectory
            >>> from importlib.util import find_spec
            >>> from flagquantum.qec import DemError, DetectorErrorModel
            >>> if find_spec("stim") is not None:
            ...     s = DetectorErrorModel(1, 0, (DemError(1.0, (0,)),)).compile_sampler()
            ...     with TemporaryDirectory() as directory:
            ...         path = Path(directory) / "detectors.b8"
            ...         s.sample_write(2, det_out_file=path, det_out_format="b8", obs_out_file=None)
            ...         assert path.read_bytes() == bytes([1, 1])
        """
        stream = self._active()
        shots = _shots(shots)
        formats = (det_out_format, obs_out_format, err_out_format)
        for encoding in (*formats, replay_err_in_format):
            _file_size_bound(64, 0, encoding)
        _engine._check_budget(0, self._widths, True, False, None)
        targets = (det_out_file, obs_out_file, err_out_file)
        plans: list[tuple[Path, _Format, tuple[tuple[str, int], ...], int]] = []
        source = None if replay_err_in_file is None else _path(replay_err_in_file)
        capacity: dict[int, tuple[Path, int]] = {}
        for target, encoding, label, width in zip(
            targets, formats, ("D", "L", "M"), self._widths, strict=True
        ):
            if target is None:
                continue
            path = _path(target)
            if os.path.lexists(path):
                raise FileExistsError(f"DEM sample destination already exists: {path}")
            parent = path.parent.resolve(strict=True)
            if not parent.is_dir():
                raise NotADirectoryError(str(parent))
            path = parent / path.name
            if any(path == plan[0] for plan in plans) or (
                source is not None and path == source.absolute()
            ):
                raise ValueError("DEM sample input and outputs must be distinct")
            bound = _file_size_bound(shots, width, encoding)
            plans.append((path, encoding, ((label, width),), bound))
            device = parent.stat().st_dev
            capacity[device] = (parent, capacity.get(device, (parent, 0))[1] + bound)
        if sum(plan[3] for plan in plans) > _MAX_FILE:
            raise CapabilityError("DEM outputs exceed the aggregate 1 GiB byte limit")
        for parent, bound in capacity.values():
            if shutil.disk_usage(parent).free < bound:
                raise OSError("insufficient free space for DEM output files")
        published: list[tuple[Path, int, int]] = []
        executing = False
        try:
            with ExitStack() as stack:
                staged: list[Path] = []
                for path, _, _, _ in plans:
                    directory = stack.enter_context(
                        TemporaryDirectory(prefix=".fq-dem-", dir=path.parent)
                    )
                    staged.append(Path(directory) / "records")
                replay_path: Path | None = None
                if source is not None:
                    directory = stack.enter_context(
                        TemporaryDirectory(prefix="fq-dem-replay-")
                    )
                    replay_path = Path(directory) / "errors"
                    _snapshot(source, replay_path)
                    width = self._widths[2]
                    bound = _file_size_bound(shots, width, replay_err_in_format)
                    _validate_record_file(
                        replay_path,
                        shots,
                        width,
                        replay_err_in_format,
                        bound,
                        sections=(("M", width),),
                    )
                outputs: list[str | None] = []
                iterator = iter(staged)
                for target in targets:
                    outputs.append(None if target is None else str(next(iterator)))
                if shots:
                    executing = True
                    stream.sample_write(
                        shots,
                        (outputs[0], outputs[1], outputs[2]),
                        formats,
                        None if replay_path is None else str(replay_path),
                        replay_err_in_format,
                    )
                else:
                    for path in staged:
                        path.touch(exist_ok=False)
                for path, (_, encoding, layout, bound) in zip(
                    staged, plans, strict=True
                ):
                    _validate_record_file(
                        path, shots, layout[0][1], encoding, bound, sections=layout
                    )
                for path, (target, _, _, _) in zip(staged, plans, strict=True):
                    identity = path.stat()
                    executing = True
                    os.link(path, target)
                    published.append((target, identity.st_dev, identity.st_ino))
        except BaseException as error:
            if executing or published:
                self._stream = None
            cleanup_errors = []
            for path, device, inode in reversed(published):
                try:
                    st = path.lstat()
                    if (st.st_dev, st.st_ino) == (device, inode):
                        path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            if cleanup_errors:
                raise OSError(
                    f"DEM output cleanup failed: {cleanup_errors}; original: {error}"
                ) from error
            raise


def _compile(model: DetectorErrorModel, seed: int | None) -> CompiledDemSampler:
    if seed is not None and type(seed) is not int:
        raise TypeError("DEM seed must be an integer or None")
    seed = _validate_seed(seed)
    widths = (model.num_detectors, model.num_observables, model.num_errors)
    if any(n > _MAX_DIMENSION for n in widths):
        raise CapabilityError("DEM dimensions exceed 65536")
    _engine._check_budget(0, widths, True, False, None)
    # Preserve the model's authoritative refusal, including unique stated IDs.
    if model.stated_error_ids():
        model.to_stim_text()
    targets = sum(
        len(error.detectors) + len(error.observables) for error in model.errors
    )
    if targets > _MAX_TARGETS:
        raise CapabilityError("DEM targets exceed 262144")
    estimate = sum(
        len(f"error({e.probability!r}) ")
        + sum(len(str(t)) + 2 for t in (*e.detectors, *e.observables))
        + 1
        for e in model.errors
    )
    estimate += sum(len("detector D") + len(str(i)) + 1 for i in range(widths[0]))
    estimate += sum(
        len("logical_observable L") + len(str(i)) + 1 for i in range(widths[1])
    )
    if estimate > _MAX_TEXT:
        raise CapabilityError("canonical DEM text exceeds the 1 MiB byte limit")
    text = model.to_stim_text()
    assert len(text.encode("utf-8")) <= estimate
    sampler = object.__new__(CompiledDemSampler)
    sampler._widths = widths
    sampler._pid = os.getpid()
    sampler._thread = current_thread()
    sampler._stream = _engine.prepare(text, seed, widths)
    return sampler
