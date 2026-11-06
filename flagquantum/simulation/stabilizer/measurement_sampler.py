"""Prepared sampling of the explicit records in a bounded Clifford circuit."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread, current_thread
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, SupportsIndex, cast

import torch

from ...core.classical_control import (
    instruction_condition_clauses,
    is_classical_control_instruction,
    validate_classical_control,
)
from ...core.ir import CircuitIR
from ...errors import CapabilityError, ValidationError
from ._validation import _validate_seed, _validate_shots
from .engine import (
    CLIFFORD_GATE_NAMES,
    _EngineOperation,
    _instruction_operations,
    _noisy_engine_circuit,
    _stim,
)

if TYPE_CHECKING:
    from ...circuit import Circuit

__all__ = ("CompiledMeasurementSampler",)

_MAX_PAYLOAD_BYTES = 64 * 1024**2
_MAX_FRAME_ESTIMATE_BYTES = 64 * 1024**2
_MAX_FILE_BYTES = 1024**3
_MAX_QUBITS = 4096
_MAX_MEASUREMENTS = 65536
_MAX_LOWERED_OPERATIONS = 100000
_FRAME_SHOT_ALIGNMENT = 1024


class _SdkSampler(Protocol):
    def sample(self, shots: int, *, bit_packed: bool) -> object: ...

    def sample_write(self, shots: int, *, filepath: str, format: str) -> None: ...


def _require_within_limit(label: str, requested: int, maximum: int) -> None:
    if requested > maximum:
        raise CapabilityError(f"{label} requires {requested}; limit is {maximum}")


def _check_sample_budget(shots: int, columns: int, qubits: int, packed: bool) -> int:
    width = (columns + 7) // 8 if packed else columns
    _require_within_limit("sample payload bytes", shots * width, _MAX_PAYLOAD_BYTES)
    blocks = (shots + _FRAME_SHOT_ALIGNMENT - 1) // _FRAME_SHOT_ALIGNMENT
    frame_bits = blocks * _FRAME_SHOT_ALIGNMENT * (2 * qubits + columns)
    _require_within_limit(
        "sample frame estimate bytes", (frame_bits + 7) // 8, _MAX_FRAME_ESTIMATE_BYTES
    )
    return width


def _lower_program(ir: CircuitIR) -> tuple[tuple[_EngineOperation, ...], int]:
    _require_within_limit("source qubits", ir.n_wires, _MAX_QUBITS)
    if ir.measurements or {"layout", "routing"} & ir.metadata.keys():
        raise CapabilityError(
            "measurement sampling requires an unplaced source circuit"
        )
    # Every admitted source operation emits at least one engine operation.
    _require_within_limit(
        "source operations", len(ir.instructions), _MAX_LOWERED_OPERATIONS
    )
    validate_classical_control(ir.instructions, n_qubits=ir.n_wires)
    operations: list[_EngineOperation] = []
    latest: dict[int, int] = {}
    records = 0
    for index, instruction in enumerate(ir.instructions):
        opcode = instruction.name
        metadata = instruction.metadata
        if (
            instruction.matrix is not None
            or instruction.params
            or metadata.get("is_channel")
            or "routing_phase" in metadata
        ):
            raise CapabilityError(
                f"instruction {index} is outside the exact Clifford slice"
            )
        clauses = instruction_condition_clauses(instruction)
        if clauses:
            if (
                opcode != "x"
                or len(instruction.wires) != 1
                or len(clauses) != 1
                or len(clauses[0]) != 1
            ):
                raise CapabilityError(
                    "prepared sampling supports one classical test on X"
                )
            ((slot, expected),) = clauses[0]
            if expected == 0:
                operations.append(("X", instruction.wires, None))
            operations.append(
                ("CX", (latest[slot] - records, *instruction.wires), None)
            )
        else:
            if opcode == "measure":
                latest[int(metadata["classical_bit"])] = records
                records += 1
                _require_within_limit(
                    "explicit measurements", records, _MAX_MEASUREMENTS
                )
            elif opcode != "reset" and (
                opcode not in CLIFFORD_GATE_NAMES
                or is_classical_control_instruction(instruction)
            ):
                raise CapabilityError(
                    f"instruction {index} is outside the exact Clifford slice"
                )
            operations.extend(_instruction_operations(instruction, index))
        _require_within_limit(
            "lowered operations", len(operations), _MAX_LOWERED_OPERATIONS
        )
    if not records:
        raise CapabilityError(
            "prepared sampling requires at least one explicit measurement"
        )
    return tuple(operations), records


def _prepare_sdk_sampler(
    operations: tuple[_EngineOperation, ...], seed: int | None
) -> _SdkSampler:
    stim = _stim()
    if stim.__version__ != "1.16.0":
        raise CapabilityError("prepared measurement sampling requires Stim 1.16.0")
    engine_operations = tuple(
        (name, tuple(stim.target_rec(w) if w < 0 else w for w in wires), arguments)
        for name, wires, arguments in operations
    )
    circuit = _noisy_engine_circuit(engine_operations, ())
    return cast(_SdkSampler, circuit.compile_sampler(seed=seed))


def _prepare_embedded_sampler(
    operations: tuple[_EngineOperation, ...], seed: int | None, columns: int
) -> _SdkSampler:
    from ._embedded_sampler import _EmbeddedSampler

    return _EmbeddedSampler(operations, seed, columns)


def _as_torch_sample(data: object) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    # The SDK returns a new array per call. Torch keeps its storage alive.
    return torch.from_numpy(data)


class CompiledMeasurementSampler:
    """Reusable CPU sampler for the explicit measurements of a Clifford circuit."""

    __slots__ = ("_sdk", "_qubits", "_columns", "_pid", "_thread")
    _sdk: _SdkSampler | None
    _qubits: int
    _columns: int
    _pid: int
    _thread: Thread

    def __init__(self) -> None:
        raise TypeError("create a sampler with circuit.compile_sampler()")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("compiled measurement samplers cannot be serialized")

    def _engine(self) -> _SdkSampler:
        if self._pid != os.getpid() or self._thread is not current_thread():
            raise RuntimeError(
                "compiled measurement sampler belongs to its creating thread and process"
            )
        if self._sdk is None:
            raise RuntimeError("compiled measurement sampler is invalidated")
        return self._sdk

    def sample(self, shots: int, *, bit_packed: bool = False) -> torch.Tensor:
        """Return owned CPU bool rows or LSB-first packed uint8 rows."""
        sdk = self._engine()
        shots = _validate_shots(shots)
        if type(bit_packed) is not bool:
            raise TypeError("bit_packed must be a bool")
        width = _check_sample_budget(shots, self._columns, self._qubits, bit_packed)
        try:
            result = _as_torch_sample(sdk.sample(shots, bit_packed=bit_packed))
            expected_dtype = torch.uint8 if bit_packed else torch.bool
            if (
                result.dtype != expected_dtype
                or result.shape != (shots, width)
                or not result.is_contiguous()
                or result.device.type != "cpu"
            ):
                raise ValidationError("Stim returned an invalid measurement sample")
        except BaseException:
            self._sdk = None
            raise
        return result

    def sample_write(
        self, shots: int, *, filepath: str | os.PathLike[str], format: Literal["b8"]
    ) -> None:
        """Write a complete local b8 file without replacing an existing path."""
        sdk = self._engine()
        shots = _validate_shots(shots)
        if format != "b8":
            raise ValueError("prepared measurement sampling writes only format='b8'")
        expected = shots * ((self._columns + 7) // 8)
        _require_within_limit("sample file bytes", expected, _MAX_FILE_BYTES)
        target = Path(filepath)
        if os.path.lexists(target):
            raise FileExistsError(f"sample destination already exists: {target}")
        parent = target.parent.resolve(strict=True)
        if not parent.is_dir():
            raise NotADirectoryError(str(parent))
        if shutil.disk_usage(parent).free < expected:
            raise OSError(f"insufficient free space for {expected} sample file bytes")
        with TemporaryDirectory(prefix=".fq-sample-", dir=parent) as directory:
            staging = Path(directory) / "records.b8"
            try:
                sdk.sample_write(shots, filepath=str(staging), format="b8")
                if staging.stat().st_size != expected:
                    raise OSError(
                        f"sample file size disagrees with expected {expected} bytes"
                    )
                # A same-filesystem hard link publishes the complete file without replacing it.
                os.link(staging, parent / target.name)
            except BaseException:
                self._sdk = None
                raise


def _compile_measurement_sampler(
    program: Circuit, *, seed: int | None
) -> CompiledMeasurementSampler:
    if (
        torch.device(program.device).type != "cpu"
        or program.runtime_config.backend != "pytorch"
        or program.bsz != 1
        or program.circuit_param.get("inputs") is not None
    ):
        raise CapabilityError(
            "prepared sampling requires one CPU PyTorch circuit with its default initial state"
        )
    validated_seed = _validate_seed(seed)
    ir = program.to_ir()
    operations, columns = _lower_program(ir)
    sampler = object.__new__(CompiledMeasurementSampler)
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        sampler._sdk = _prepare_embedded_sampler(operations, validated_seed, columns)
    else:
        sampler._sdk = _prepare_sdk_sampler(operations, validated_seed)
    sampler._qubits = ir.n_wires
    sampler._columns = columns
    sampler._pid = os.getpid()
    sampler._thread = current_thread()
    return sampler
