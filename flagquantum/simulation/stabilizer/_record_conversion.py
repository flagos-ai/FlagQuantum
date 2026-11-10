"""Private Stim preparation and conversion of externally supplied records."""

from __future__ import annotations

import math
import os
import shutil
import stat
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from ...core.classical_control import is_classical_control_instruction
from ...core.ir import CircuitIR, Instruction
from ...errors import CapabilityError
from ._measurement_file_validation import (
    _READ_CHUNK_BYTES,
    _file_size_bound,
    _inspect_measurement_file,
    _validate_record_file,
)
from .engine import (
    CLIFFORD_GATE_NAMES,
    _annotated_stim_source,
    _parse_stim_experiment,
    _validate_stim_reference,
)

_FileWriter = Callable[[str, str, str, str, bool, str | None, str], None]
_MAX_SNAPSHOT_BYTES = 1024**3
_MAX_FILE_OUTPUT_BYTES = 1024**3
_MAX_FILE_PRINCIPAL_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class _RecordBackend:
    dense: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    file: _FileWriter

    def __call__(self, records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dense(records)


@dataclass(frozen=True)
class _PreparedConversion:
    backend: _RecordBackend
    widths: tuple[int, int, int]
    qubits: int

    def __call__(self, records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend(records)

    def convert_file(
        self,
        source: Path,
        input_format: str,
        target: Path,
        output_format: str,
        append: bool,
        observable: Path | None,
        obs_format: str,
    ) -> None:
        _convert_record_file(
            self,
            source,
            input_format,
            target,
            output_format,
            append,
            observable,
            obs_format,
        )


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _snapshot(source: Path, target: Path) -> int:
    # O_NONBLOCK prevents an input replaced by a FIFO from blocking admission.
    source_metadata = source.lstat()
    if not stat.S_ISREG(source_metadata.st_mode):
        raise ValueError("measurement input must be a regular file, without symlinks")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(source, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("measurement input must be a regular file")
        if _identity(before) != _identity(source_metadata) or _identity(
            source.lstat()
        ) != _identity(before):
            raise OSError("measurement input changed before snapshot admission")
        if before.st_size > _MAX_SNAPSHOT_BYTES:
            raise CapabilityError("measurement snapshot exceeds the 1 GiB byte limit")
        if shutil.disk_usage(target.parent).free < before.st_size:
            raise OSError("insufficient free space for the measurement snapshot")
        copied = 0
        with target.open("xb") as output:
            while block := stream.read(_READ_CHUNK_BYTES):
                copied += len(block)
                if copied > before.st_size or copied > _MAX_SNAPSHOT_BYTES:
                    raise OSError("measurement input changed while copying")
                output.write(block)
        if (
            copied != before.st_size
            or _identity(os.fstat(stream.fileno())) != _identity(before)
            or _identity(source.lstat()) != _identity(before)
        ):
            raise OSError("measurement input changed while copying")
    return copied


def _file_principal_bytes(widths: tuple[int, int, int], qubits: int) -> int:
    def padded(n: int) -> int:
        return max(1024, ((n + 1023) // 1024) * 1024)

    measurements, detectors, observables = widths
    # Fixed 1024-shot upstream input, both output orientations, frame and scratch.
    rows = (
        padded(measurements)
        + 2 * padded(detectors + observables)
        + 2 * padded(qubits)
        + 2 * padded(2 * measurements)
        + padded(observables)
        + 4 * padded(0)
        + 8
    )
    return rows * 128


def _convert_record_file(
    prepared: _PreparedConversion,
    source: Path,
    input_format: str,
    target: Path,
    output_format: str,
    append: bool,
    observable: Path | None,
    obs_format: str,
) -> None:
    measurements, detectors, observables = prepared.widths
    if (
        _file_principal_bytes(prepared.widths, prepared.qubits)
        > _MAX_FILE_PRINCIPAL_BYTES
    ):
        raise CapabilityError(
            "file conversion principal tables exceed the 64 MiB byte limit"
        )
    sections = (("D", detectors),) + ((("L", observables),) if append else ())
    requests = [(target, output_format, sections)]
    if observable is not None:
        requests.append((observable, obs_format, (("L", observables),)))
    plans: list[tuple[Path, str, tuple[tuple[str, int], ...], int]] = []
    source_identity = source.lstat()
    if not stat.S_ISREG(source_identity.st_mode):
        raise ValueError("measurement input must be a regular file, without symlinks")
    for destination, encoding, layout in requests:
        parent = destination.parent.resolve(strict=True)
        if not parent.is_dir():
            raise NotADirectoryError(str(parent))
        resolved = parent / destination.name
        if os.path.lexists(resolved):
            raise FileExistsError(f"conversion destination already exists: {resolved}")
        if resolved == source.absolute() or any(resolved == plan[0] for plan in plans):
            raise ValueError(
                "conversion destinations must be distinct from input and each other"
            )
        plans.append((resolved, encoding, layout, 0))
    staged: list[Path] = []
    published: list[tuple[Path, int, int]] = []
    snapshot_directory: Path | None = None
    try:
        with ExitStack() as stack:
            snapshot_directory = Path(
                stack.enter_context(TemporaryDirectory(prefix="fq-record-input-"))
            )
            snapshot = snapshot_directory / "measurements"
            snapshot_size = _snapshot(source, snapshot)
            if _identity(source_identity) != _identity(source.lstat()):
                raise OSError("measurement input changed before snapshot admission")
            shots = _inspect_measurement_file(snapshot, measurements, input_format)
            capacity: dict[int, tuple[Path, int]] = {}
            for i, (destination, encoding, layout, _) in enumerate(plans):
                bound = _file_size_bound(shots, sum(n for _, n in layout), encoding)
                if encoding == "dets":
                    bound = shots * (
                        5 + sum(n * (len(str(max(0, n - 1))) + 2) for _, n in layout)
                    )
                plans[i] = (destination, encoding, layout, bound)
                device = destination.parent.stat().st_dev
                capacity[device] = (
                    destination.parent,
                    capacity.get(device, (destination.parent, 0))[1] + bound,
                )
            output_bound = sum(plan[3] for plan in plans)
            if (
                output_bound > _MAX_FILE_OUTPUT_BYTES
                or snapshot_size + output_bound > 2 * 1024**3
            ):
                raise CapabilityError(
                    "conversion files exceed the aggregate byte limit"
                )
            # The snapshot is already allocated; remaining reservations share free space per device.
            for parent, bound in capacity.values():
                if shutil.disk_usage(parent).free < bound:
                    raise OSError(
                        f"insufficient free space for {bound} conversion output bytes"
                    )
            for destination, encoding, _, _ in plans:
                directory = stack.enter_context(
                    TemporaryDirectory(prefix=".fq-converted-", dir=destination.parent)
                )
                staged.append(Path(directory) / ("records." + encoding))
            if shots:
                prepared.backend.file(
                    str(snapshot),
                    input_format,
                    str(staged[0]),
                    output_format,
                    append,
                    str(staged[1]) if len(staged) == 2 else None,
                    obs_format,
                )
            else:
                for path in staged:
                    path.touch(exist_ok=False)
            for path, (_, encoding, layout, bound) in zip(staged, plans, strict=True):
                _validate_record_file(
                    path,
                    shots,
                    sum(n for _, n in layout),
                    encoding,
                    bound,
                    sections=layout,
                )
            for path, (destination, _, _, _) in zip(staged, plans, strict=True):
                metadata = path.stat()
                os.link(path, destination)
                published.append((destination, metadata.st_dev, metadata.st_ino))
    except BaseException as error:
        # Roll back only this request's published inodes, never a caller replacement.
        cleanup_errors = []
        for destination, device, inode in reversed(published):
            try:
                current = destination.lstat()
                if (current.st_dev, current.st_ino) == (device, inode):
                    destination.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(f"{destination}: {cleanup_error}")
        remaining = [str(path.parent) for path in staged if path.parent.exists()]
        if snapshot_directory is not None and snapshot_directory.exists():
            remaining.append(str(snapshot_directory))
        if cleanup_errors or remaining:
            raise OSError(
                f"conversion cleanup failed: {cleanup_errors}; staging: {remaining}; original: {error}"
            ) from error
        raise


def _terminal_clifford_view(program: CircuitIR) -> CircuitIR:
    instructions: list[Instruction] = []
    for index, gate in enumerate(program.instructions):
        if gate.name in {"measure", "reset"} or is_classical_control_instruction(gate):
            raise CapabilityError("terminal conversion requires a static circuit")
        if gate.matrix is not None or gate.metadata.get("is_channel"):
            raise CapabilityError(
                f"instruction {index} is outside the exact Clifford subset"
            )
        if gate.name in CLIFFORD_GATE_NAMES and not gate.params:
            instructions.append(gate)
            continue
        theta = gate.params.get("theta")
        if (
            gate.name == "rx"
            and gate.params.keys() == {"theta"}
            and type(theta) is float
            and theta == math.pi
        ):
            # RX(pi) differs from X by a global phase, which cannot affect Z records.
            instructions.append(Instruction("x", gate.wires))
            continue
        raise CapabilityError(
            f"instruction {index} {gate.name!r} is outside the bound exact Clifford subset"
        )
    return replace(program, instructions=tuple(instructions))


def _stim_record_array(
    record: torch.Tensor, width: int, packed: bool, name: str
) -> Any:
    expected = (width + 7) // 8 if packed else width
    if record.shape[1] != expected:
        raise ValueError(f"{name} has the wrong declared width")
    if packed and width % 8 and bool((record[:, -1] >> (width % 8)).any()):
        raise ValueError(f"{name} has nonzero packed padding")
    return record.to(dtype=torch.uint8 if packed else torch.bool).contiguous().numpy()


def _owned_stim_conversion_bits(buffer: Any, shape: tuple[int, int]) -> torch.Tensor:
    try:
        bits = torch.from_numpy(buffer)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CapabilityError("Stim returned an invalid conversion buffer") from error
    if bits.dtype != torch.bool or bits.shape != shape:
        raise CapabilityError("Stim returned an invalid conversion dtype or shape")
    return bits.clone()


def _compile_stim_converter(circuit: Any) -> Any:
    try:
        return circuit.compile_m2d_converter(skip_reference_sample=False)
    except (ValueError, RuntimeError) as error:
        raise CapabilityError("Stim measurement conversion failed") from error


def _convert_stim_arrays(
    converter: Any,
    raw: Any,
    sweeps: Any,
    shapes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if shapes[0][0] == 0:
        return torch.empty(shapes[0], dtype=torch.bool), torch.empty(
            shapes[1], dtype=torch.bool
        )
    try:
        result = converter.convert(
            measurements=raw,
            sweep_bits=sweeps,
            separate_observables=True,
            bit_packed=False,
        )
    except (ValueError, RuntimeError) as error:
        raise CapabilityError("Stim measurement conversion failed") from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise CapabilityError("Stim returned an invalid conversion pair")
    return (
        _owned_stim_conversion_bits(result[0], shapes[0]),
        _owned_stim_conversion_bits(result[1], shapes[1]),
    )


def _convert_stim_measurements(
    circuit_text: str,
    measurements: torch.Tensor,
    *,
    sweep_bits: torch.Tensor | None,
    bit_packed_measurements: bool,
    bit_packed_sweep_bits: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve one-shot validation order and zero-shot behavior."""
    circuit = _parse_stim_experiment(circuit_text)
    raw = _stim_record_array(
        measurements, circuit.num_measurements, bit_packed_measurements, "measurements"
    )
    sweeps = None
    if sweep_bits is not None:
        sweeps = _stim_record_array(
            sweep_bits, circuit.num_sweep_bits, bit_packed_sweep_bits, "sweep_bits"
        )
    _validate_stim_reference(circuit)
    shots = measurements.shape[0]
    converter = _compile_stim_converter(circuit) if shots else None
    return _convert_stim_arrays(
        converter,
        raw,
        sweeps,
        ((shots, circuit.num_detectors), (shots, circuit.num_observables)),
    )


def _prepare_terminal_record_conversion(
    program: CircuitIR,
    *,
    terminal_qubits: tuple[int, ...],
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> _PreparedConversion:
    text = _annotated_stim_source(
        _terminal_clifford_view(program),
        terminal_qubits=terminal_qubits,
        detector_columns=detector_columns,
        observable_columns=observable_columns,
    )
    circuit = _parse_stim_experiment(text)
    _validate_stim_reference(circuit)
    converter = _compile_stim_converter(circuit)
    width, detectors, observables = (
        circuit.num_measurements,
        circuit.num_detectors,
        circuit.num_observables,
    )

    def convert(measurements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = _stim_record_array(measurements, width, False, "measurements")
        shots = measurements.shape[0]
        return _convert_stim_arrays(
            converter, raw, None, ((shots, detectors), (shots, observables))
        )

    return _PreparedConversion(
        _RecordBackend(convert, _sdk_file_writer(converter)),
        (width, detectors, observables),
        program.n_wires,
    )


_MAX_EXPLICIT_CONVERSION_BYTES = 64 * 1024**2


def _sdk_file_writer(converter: Any) -> _FileWriter:
    def write(
        source: str,
        input_format: str,
        target: str,
        output_format: str,
        append: bool,
        observable: str | None,
        obs_format: str,
    ) -> None:
        converter.convert_file(
            measurements_filepath=source,
            measurements_format=input_format,
            detection_events_filepath=target,
            detection_events_format=output_format,
            append_observables=append,
            obs_out_filepath=observable,
            obs_out_format=obs_format,
        )

    return write


def _prepare_explicit_backend(
    text: str, detectors: int, observables: int
) -> _RecordBackend:
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        from ._embedded_sampler import _EmbeddedRecordConverter

        try:
            native = _EmbeddedRecordConverter(text, detectors, observables)
        except (ValueError, RuntimeError) as error:
            raise CapabilityError(
                "explicit conversion has nondeterministic declared parities"
            ) from error

        def embedded_convert(
            records: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            try:
                return native.convert(records)
            except (TypeError, ValueError, RuntimeError) as error:
                raise CapabilityError(
                    "embedded measurement conversion failed"
                ) from error

        return _RecordBackend(embedded_convert, native.convert_file)

    from .engine import _stim

    if _stim().__version__ != "1.16.0":
        raise CapabilityError("explicit measurement conversion requires Stim 1.16.0")
    circuit = _parse_stim_experiment(text)
    _validate_stim_reference(circuit)
    converter = _compile_stim_converter(circuit)

    def sdk_convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _convert_stim_arrays(
            converter,
            records.contiguous().numpy(),
            None,
            ((len(records), detectors), (len(records), observables)),
        )

    return _RecordBackend(sdk_convert, _sdk_file_writer(converter))


def _prepare_explicit_record_conversion(
    program: CircuitIR,
    *,
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], _PreparedConversion]:
    from ._embedded_sampler import _operations_text
    from .measurement_sampler import _lower_program

    operations, width = _lower_program(program)
    qubits = tuple(
        gate.wires[0] for gate in program.instructions if gate.name == "measure"
    )
    lines = [_operations_text(operations)]
    for row in detector_columns:
        targets = " ".join(f"rec[{column - width}]" for column in row)
        lines.append(f"DETECTOR {targets}")
    for index, row in enumerate(observable_columns):
        targets = " ".join(f"rec[{column - width}]" for column in row)
        lines.append(f"OBSERVABLE_INCLUDE({index}) {targets}")
    backend = _prepare_explicit_backend(
        "\n".join(lines), len(detector_columns), len(observable_columns)
    )
    outputs = len(detector_columns) + len(observable_columns)

    def convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shots = len(records)
        copied_input = shots * width * (1 + int(not records.is_contiguous()))
        output_bytes = shots * outputs
        aligned_shots = ((shots + 1023) // 1024) * 1024

        # Round both table dimensions conservatively above the pinned SIMD width.
        # Include frame tables, growing measurement storage (with the old
        # allocation live), observable storage, zero-width tables and scratch.
        def padded(rows: int) -> int:
            return max(1024, ((rows + 1023) // 1024) * 1024)

        table_rows = (
            padded(width)
            + padded(outputs)
            + 2 * padded(program.n_wires)
            + 2 * padded(2 * width)
            + padded(len(observable_columns))
            + 4 * padded(0)
            + 8
        )
        principal_bytes = table_rows * aligned_shots // 8
        for label, needed in (
            ("input copies", copied_input),
            ("output payload", output_bytes),
            ("principal tables", principal_bytes),
        ):
            if needed > _MAX_EXPLICIT_CONVERSION_BYTES:
                raise CapabilityError(
                    f"explicit conversion {label} exceeds the byte limit"
                )
        if not shots:
            return (
                torch.empty((0, len(detector_columns)), dtype=torch.bool),
                torch.empty((0, len(observable_columns)), dtype=torch.bool),
            )
        return backend(records)

    def write(
        source: str,
        input_format: str,
        target: str,
        output_format: str,
        append: bool,
        observable: str | None,
        obs_format: str,
    ) -> None:
        backend.file(
            source, input_format, target, output_format, append, observable, obs_format
        )

    return qubits, _PreparedConversion(
        _RecordBackend(convert, write),
        (width, len(detector_columns), len(observable_columns)),
        program.n_wires,
    )
