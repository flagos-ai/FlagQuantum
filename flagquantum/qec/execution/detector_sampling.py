"""Reference-normalized detector samples and exact DEMs from native circuits."""

from __future__ import annotations

import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread, current_thread
from typing import TYPE_CHECKING, Literal, NoReturn, SupportsIndex

import torch

from ...core.ir import CircuitIR
from ...errors import CapabilityError, ValidationError
from ...simulation.stabilizer._detector_sampling import (
    _DetectorStream,
    _exact_detector_error_model_text,
    _prepare_detector_sampler,
)
from ...simulation.stabilizer._measurement_file_validation import (
    _file_size_bound,
    _validate_record_file,
)
from ...simulation.stabilizer._validation import _validate_seed
from ...simulation.stabilizer.measurement_sampler import _require_within_limit
from ..dem import DetectorErrorModel
from .context import MeasurementMap

if TYPE_CHECKING:
    from ...circuit import Circuit

__all__ = ("CompiledDetectorSampler",)


def _source(
    program: Circuit, detector_map: MeasurementMap, observable_map: MeasurementMap
) -> CircuitIR:
    if (
        torch.device(program.device).type != "cpu"
        or program.runtime_config.backend != "pytorch"
        or program.bsz != 1
        or program.circuit_param.get("inputs") is not None
    ):
        raise CapabilityError(
            "detector execution requires one default-state CPU PyTorch circuit"
        )
    if not isinstance(detector_map, MeasurementMap) or not isinstance(
        observable_map, MeasurementMap
    ):
        raise TypeError(
            "detector_map and observable_map must be MeasurementMap instances"
        )
    ir = program.to_ir()
    width = sum(g.name == "measure" for g in ir.instructions)
    if (
        detector_map.num_measurements != width
        or observable_map.num_measurements != width
    ):
        raise ValueError("both measurement maps must match the explicit record width")
    if not detector_map.num_rows:
        raise ValueError("detector execution requires at least one detector")
    _require_within_limit(
        "parity rows", detector_map.num_rows + observable_map.num_rows, 65536
    )
    _require_within_limit(
        "parity references",
        detector_map.num_entries + observable_map.num_entries,
        100000,
    )
    return ir


def _packed(bits: torch.Tensor) -> torch.Tensor:
    shots, columns = bits.shape
    result = torch.zeros((shots, (columns + 7) // 8), dtype=torch.uint8)
    for bit in range(8):
        part = bits[:, bit::8].to(torch.uint8)
        result[:, : part.shape[1]] |= part << bit
    return result


class CompiledDetectorSampler:
    """Reusable detector sampling with bounded CPU outputs.

    Create this handle with ``circuit.compile_detector_sampler``. Dense rows are
    bool; packed rows are LSB-first uint8. Separate outputs each pad to a byte;
    appended outputs pack across their boundary. Seeds reproduce an identical
    call schedule within one pinned implementation, not across batching changes.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.dynamic import DynamicCircuit
        >>> if find_spec("stim") is not None:
        ...     sampler = DynamicCircuit(1).x(0).measure(0).compile_detector_sampler(
        ...         detector_map=MeasurementMap(1, ((0,),)),
        ...         observable_map=MeasurementMap(1, ()), seed=7,
        ...     )
        ...     assert sampler.sample(2).tolist() == [[False], [False]]
    """

    __slots__ = ("_stream", "_widths", "_qubits", "_measurements", "_pid", "_thread")
    _stream: _DetectorStream | None
    _widths: tuple[int, int]
    _qubits: int
    _measurements: int
    _pid: int
    _thread: Thread

    def __init__(self) -> None:
        raise TypeError("create a sampler with circuit.compile_detector_sampler()")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("compiled detector samplers cannot be serialized")

    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return detector flips, a separate pair, or detector-then-observable rows.

        Invalid requests do not draw. A backend failure invalidates the handle.
        Zero shots return empty outputs without advancing RNG. Output and padded
        principal-table estimates are individually bounded at 64 MiB; these are
        allocation guards, not a process RSS guarantee.
        """
        if self._pid != os.getpid() or self._thread is not current_thread():
            raise RuntimeError(
                "detector sampler belongs to its creating thread and process"
            )
        stream = self._stream
        if stream is None:
            raise RuntimeError("detector sampler is invalidated")
        if type(shots) is not int or shots < 0:
            raise ValidationError("shots must be a nonnegative integer")
        if any(
            type(flag) is not bool
            for flag in (separate_observables, append_observables, bit_packed)
        ):
            raise TypeError("detector sample flags must be bool")
        if separate_observables and append_observables:
            raise ValueError(
                "separate_observables and append_observables are mutually exclusive"
            )
        dets, obs = self._widths
        total = dets + obs
        _require_within_limit(
            "detector sample copies bytes", shots * total * 3, 64 * 1024**2
        )

        def padded(n: int) -> int:
            return max(1024, ((n + 1023) // 1024) * 1024)

        rows = (
            2 * padded(self._qubits)
            + 2 * padded(2 * self._measurements)
            + padded(dets)
            + padded(obs)
            + 4 * padded(0)
        )
        _require_within_limit(
            "detector frame estimate bytes",
            rows * ((shots + 1023) // 1024) * 128,
            64 * 1024**2,
        )
        if shots == 0:
            pair = (
                torch.empty((0, dets), dtype=torch.bool),
                torch.empty((0, obs), dtype=torch.bool),
            )
        else:
            try:
                pair = stream.sample(shots)
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise CapabilityError("detector engine returned an invalid pair")
                for result, width in zip(pair, self._widths, strict=True):
                    if (
                        not isinstance(result, torch.Tensor)
                        or result.dtype != torch.bool
                        or result.device.type != "cpu"
                        or result.layout != torch.strided
                        or result.shape != (shots, width)
                        or not result.is_contiguous()
                    ):
                        raise CapabilityError(
                            "detector engine returned invalid sample data"
                        )
                pair = (pair[0].clone(), pair[1].clone())
            except BaseException:
                self._stream = None
                raise
        if separate_observables:
            return (_packed(pair[0]), _packed(pair[1])) if bit_packed else pair
        result = torch.cat(pair, dim=1) if append_observables else pair[0]
        return _packed(result) if bit_packed else result

    def sample_write(
        self,
        shots: int,
        *,
        filepath: str | os.PathLike[str],
        format: Literal["01", "b8", "r8", "hits", "dets", "ptb64"] = "01",
        prepend_observables: bool = False,
        append_observables: bool = False,
        obs_out_filepath: str | os.PathLike[str] | None = None,
        obs_out_format: Literal["01", "b8", "r8", "hits", "dets", "ptb64"] = "01",
    ) -> None:
        """Write bounded detector flips and optional same-shot observable flips.

        Files never replace existing paths. Select at most one observable layout:
        prepend, append, or a separate file. Two files publish sequentially after
        both validate; cross-file atomic visibility is not guaranteed. Backend or
        publication failure invalidates this handle. Limits bound allocations and
        final-file sizes, not process RSS or all temporary disk usage.

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from pathlib import Path
            >>> from importlib.util import find_spec
            >>> from flagquantum.dynamic import DynamicCircuit
            >>> if find_spec("stim") is not None:
            ...     circuit = DynamicCircuit(1).x(0).bit_flip(0, 1.0).measure(0)
            ...     writer = circuit.compile_detector_sampler(
            ...         detector_map=MeasurementMap(1, ((0,),)),
            ...         observable_map=MeasurementMap(1, ((0,),)), seed=7,
            ...     )
            ...     with TemporaryDirectory() as directory:
            ...         path = Path(directory) / "events.b8"
            ...         writer.sample_write(64, filepath=path, format="b8",
            ...                             append_observables=True)
            ...         assert path.read_bytes() == b"\\x03" * 64
        """
        if self._pid != os.getpid() or self._thread is not current_thread():
            raise RuntimeError(
                "detector sampler belongs to its creating thread and process"
            )
        stream = self._stream
        if stream is None:
            raise RuntimeError("detector sampler is invalidated")
        if type(shots) is not int or shots <= 0:
            raise ValidationError("shots must be a positive integer")
        if (
            type(prepend_observables) is not bool
            or type(append_observables) is not bool
        ):
            raise TypeError("detector file flags must be bool")
        if (
            prepend_observables + append_observables + (obs_out_filepath is not None)
            > 1
        ):
            raise ValueError("observable file layouts are mutually exclusive")
        # Validate the unused format too, without applying its shot grouping.
        _file_size_bound(64, 0, obs_out_format)
        dets, obs = self._widths
        sections: tuple[tuple[str, int], ...] = (("D", dets),)
        if prepend_observables:
            sections = (("L", obs), ("D", dets))
        elif append_observables:
            sections += (("L", obs),)
        requests: list[
            tuple[
                Path,
                Literal["01", "b8", "r8", "hits", "dets", "ptb64"],
                tuple[tuple[str, int], ...],
            ]
        ] = [(Path(filepath), format, sections)]
        if obs_out_filepath is not None:
            requests.append((Path(obs_out_filepath), obs_out_format, (("L", obs),)))
        plans: list[
            tuple[
                Path,
                Literal["01", "b8", "r8", "hits", "dets", "ptb64"],
                tuple[tuple[str, int], ...],
                int,
            ]
        ] = []
        capacity: dict[int, tuple[Path, int]] = {}
        for target, encoding, layout in requests:
            columns = sum(n for _, n in layout)
            bound = _file_size_bound(shots, columns, encoding)
            if encoding == "dets":
                bound = shots * (
                    5 + sum(n * (len(str(max(0, n - 1))) + 2) for _, n in layout)
                )
            if os.path.lexists(target):
                raise FileExistsError(f"sample destination already exists: {target}")
            parent = target.parent.resolve(strict=True)
            if not parent.is_dir():
                raise NotADirectoryError(str(parent))
            resolved = parent / target.name
            if any(resolved == prior[0] for prior in plans):
                raise ValueError("sample destinations must be distinct")
            device = parent.stat().st_dev
            previous = capacity.get(device, (parent, 0))[1]
            capacity[device] = (parent, previous + bound)
            plans.append((resolved, encoding, layout, bound))
        _require_within_limit("detector file bytes", sum(x[3] for x in plans), 1024**3)
        for parent, bound in capacity.values():
            if shutil.disk_usage(parent).free < bound:
                raise OSError(f"insufficient free space for {bound} sample file bytes")

        def padded(n: int) -> int:
            return max(1024, ((n + 1023) // 1024) * 1024)

        rows = (
            2 * padded(self._qubits)
            + 2 * padded(2 * self._measurements)
            + padded(dets)
            + padded(obs)
            + 4 * padded(0)
        )
        if prepend_observables or append_observables:
            rows += padded(dets + obs)
        _require_within_limit(
            "detector file frame estimate bytes", rows * 128, 64 * 1024**2
        )
        staged: list[Path] = []
        published: list[tuple[Path, int, int]] = []
        try:
            with ExitStack() as stack:
                for target, encoding, _, _ in plans:
                    directory = stack.enter_context(
                        TemporaryDirectory(prefix=".fq-detectors-", dir=target.parent)
                    )
                    staged.append(Path(directory) / ("records." + encoding))
                stream.sample_write(
                    shots,
                    filepath=str(staged[0]),
                    format=format,
                    prepend_observables=prepend_observables,
                    append_observables=append_observables,
                    obs_out_filepath=str(staged[1]) if len(staged) == 2 else None,
                    obs_out_format=obs_out_format,
                )
                for path, (_, encoding, layout, bound) in zip(
                    staged, plans, strict=True
                ):
                    _validate_record_file(
                        path,
                        shots,
                        sum(n for _, n in layout),
                        encoding,
                        bound,
                        sections=layout,
                    )
                for path, (target, _, _, _) in zip(staged, plans, strict=True):
                    identity = path.stat()
                    os.link(path, target)
                    published.append((target, identity.st_dev, identity.st_ino))
        except BaseException as error:
            self._stream = None
            cleanup_errors = []
            for target, device, inode in reversed(published):
                try:
                    current = target.lstat()
                    if (current.st_dev, current.st_ino) == (device, inode):
                        target.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    cleanup_errors.append(f"{target}: {cleanup_error}")
            remaining = [str(path.parent) for path in staged if path.parent.exists()]
            if cleanup_errors or remaining:
                raise OSError(
                    f"detector output cleanup failed: {cleanup_errors}; staging: {remaining}; original: {error}"
                ) from error
            raise


def _compile_detector_sampler(
    program: Circuit,
    *,
    detector_map: MeasurementMap,
    observable_map: MeasurementMap,
    seed: int | None,
) -> CompiledDetectorSampler:
    ir = _source(program, detector_map, observable_map)
    seed = _validate_seed(seed)
    stream = _prepare_detector_sampler(
        ir,
        detector_columns=detector_map.rows,
        observable_columns=observable_map.rows,
        seed=seed,
    )
    handle = object.__new__(CompiledDetectorSampler)
    handle._stream = stream
    handle._widths = (detector_map.num_rows, observable_map.num_rows)
    handle._qubits = ir.n_wires
    handle._measurements = detector_map.num_measurements
    handle._pid = os.getpid()
    handle._thread = current_thread()
    return handle


def _detector_error_model(
    program: Circuit, *, detector_map: MeasurementMap, observable_map: MeasurementMap
) -> DetectorErrorModel:
    ir = _source(program, detector_map, observable_map)
    text = _exact_detector_error_model_text(
        ir, detector_columns=detector_map.rows, observable_columns=observable_map.rows
    )
    return DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=False)
