"""Portable, data-only FlagQuantum Experiment (.fqx) files.

This module owns archive I/O, not compilation, execution, provider sessions or
training checkpoints. Existing IR and execution-plan schemas remain authoritative.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..circuit import Circuit
from ..core.contracts import AccuracyContract
from ..core.ir import CircuitIR, ensure_circuit_ir
from ..errors import SerializationError
from ..version import __version__
from ._experiment_codec import (
    MAX_JSON_BYTES,
    Reader,
    Writer,
    _info,
    fields,
    json_bytes,
    preflight,
    read_json,
)
from .execution_plan import ExecutionPlan
from .execution_plan_contract import _compiler_identity, _validate_plan_payload
from .result import ExecutionResult, MeasurementResult

_SCHEMA = "flagquantum.experiment"
_VERSION = "1.0"
_ARRAY = re.compile(r"arrays/[0-9]{4}\.bin")
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Experiment:
    """A saved program and optional numeric result, without live execution state.

    Loaded tensors are detached owned CPU data. ``result.plan`` is deliberately
    absent: ``execution_plan`` exposes the recorded data, not an executable plan.
    Save this object directly to preserve its historical execution record.
    """

    program: CircuitIR
    result: ExecutionResult | None
    _execution_json: str | None = field(default=None, repr=False)
    _compiler_json: str | None = field(default=None, repr=False)

    @property
    def execution_plan(self) -> dict[str, object] | None:
        """Return a fresh copy of the recorded plan, without replanning."""
        return (
            None
            if self._execution_json is None
            else _mapping(read_json(self._execution_json.encode()), "execution plan")
        )


def _mapping(value: object, owner: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SerializationError(f"{owner} must be a string-keyed mapping")
    return value


def _static_plan(value: object, compiler: Mapping[str, str]) -> dict[str, object]:
    data = _mapping(value, "execution plan")
    if data.get("version") != "1.0":
        raise SerializationError(
            "FQX 1.0 supports static CircuitIR ExecutionPlans only"
        )
    try:
        return _validate_plan_payload(data, compiler_identity=compiler)
    except (TypeError, ValueError, KeyError) as error:
        raise SerializationError(f"invalid FQX execution plan: {error}") from error


def _binding(program: CircuitIR, plan: Mapping[str, object]) -> None:
    """Compare supplied IR with recorded IR, never compile a replacement."""
    given = program.to_dict()
    executed = CircuitIR.from_dict(
        _mapping(plan["program"], "executed program")
    ).to_dict()
    for value in (given, executed):
        metadata = dict(value["metadata"])
        metadata.pop("runtime_config", None)
        value["metadata"] = metadata
    if not given["measurements"]:
        executed["measurements"] = []
    if json_bytes(given) != json_bytes(executed):
        raise SerializationError(
            "program differs from the result's executed IR; use result.plan to save the exact executed program"
        )


def _prepare(
    program: Circuit | CircuitIR | ExecutionPlan | Experiment,
    result: ExecutionResult | None,
) -> tuple[
    CircuitIR, ExecutionResult | None, dict[str, object] | None, dict[str, str] | None
]:
    if isinstance(program, Experiment):
        if not isinstance(program.program, CircuitIR) or (
            program.result is not None
            and not isinstance(program.result, ExecutionResult)
        ):
            raise TypeError(
                "Experiment must contain CircuitIR and optional ExecutionResult"
            )
        if result is not None:
            raise TypeError("result must be omitted when saving an Experiment")
        plan = program.execution_plan
        compiler = (
            None
            if program._compiler_json is None
            else _compiler(read_json(program._compiler_json.encode()))
        )
        if plan is not None and compiler is not None:
            plan = _static_plan(plan, compiler)
            _binding(program.program, plan)
        if (program.result is None) != (plan is None) or (plan is None) != (
            compiler is None
        ):
            raise SerializationError(
                "Experiment result, execution plan and compiler record disagree"
            )
        return program.program, program.result, plan, compiler
    if not isinstance(program, (Circuit, CircuitIR, ExecutionPlan)):
        raise TypeError(
            "program must be a Circuit, CircuitIR, ExecutionPlan or loaded Experiment"
        )
    if result is not None and not isinstance(result, ExecutionResult):
        raise TypeError("result must be an ExecutionResult or None")
    compiler = _compiler_identity()
    supplied_plan = (
        _static_plan(program.to_dict(), compiler)
        if isinstance(program, ExecutionPlan)
        else None
    )
    ir = (
        CircuitIR.from_dict(_mapping(supplied_plan["program"], "program"))
        if supplied_plan
        else ensure_circuit_ir(program)
    )
    if result is None:
        return ir, None, None, None
    if not isinstance(result.plan, ExecutionPlan):
        raise SerializationError(
            "saving a result requires its retained static ExecutionPlan; planless/provider-native results are unsupported"
        )
    plan = _static_plan(result.plan.to_dict(), compiler)
    if supplied_plan is not None and supplied_plan["identity"] != plan["identity"]:
        raise SerializationError("program and result ExecutionPlan identities differ")
    _binding(ir, plan)
    return ir, result, plan, compiler


def _result_data(result: ExecutionResult) -> dict[str, object]:
    return {
        "value": result.value,
        "state": result.state,
        "samples": result.samples,
        "measurements": [
            {
                "kind": item.kind,
                "qubits": item.qubits,
                "value": item.value,
                "shots": item.shots,
                "metadata": dict(item.metadata),
                "statistics": dict(item.statistics),
            }
            for item in result.measurements
        ],
        "accuracy": result.accuracy.to_dict(),
        "metrics": dict(result.metrics),
        "provenance": dict(result.provenance),
        "runtime": dict(result.runtime),
        "compatibility": dict(result.compatibility),
    }


def _path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise TypeError("path must be a local str or PathLike[str]")
    return Path(raw)


def save(
    path: str | os.PathLike[str],
    program: Circuit | CircuitIR | ExecutionPlan | Experiment,
    *,
    result: ExecutionResult | None = None,
    overwrite: bool = False,
) -> None:
    """Atomically save a program and its existing result as a .fqx experiment.

    Args:
        path: Local destination ending in .fqx. Parent directories must exist.
        program: Circuit, compiled IR, static plan, or an Experiment from load.
        result: Existing result whose retained plan must match the program.
        overwrite: Explicitly allow replacing an existing destination.

    Raises:
        SerializationError: Unsupported, mismatched or oversized data.
        FileExistsError: Destination exists and overwrite is false.
        OSError: Local filesystem operation fails.

    Examples:
        >>> import flagquantum as fq
        >>> program = fq.compile(fq.Circuit(2).h(0).cx(0, 1))
        >>> result = fq.run(program, outputs=fq.counts(), shots=32)
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "bell.fqx"
        ...     fq.save(path, program, result=result)
        ...     print(path.exists())
        True
    """
    destination = _path(path)
    if destination.suffix.lower() != ".fqx":
        raise SerializationError("FlagQuantum Experiment files use the .fqx extension")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    if not overwrite and destination.exists():
        raise FileExistsError(destination)
    ir, recorded, plan, compiler = _prepare(program, result)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".fqx-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
                writer = Writer(archive)
                writer.add_json("circuit.json", ir.to_dict())
                if recorded is not None:
                    writer.add_json(
                        "result.json",
                        {
                            "schema": _SCHEMA + ".result",
                            "version": _VERSION,
                            "execution_plan": plan,
                            "compiler": compiler,
                            "data": writer.encode(_result_data(recorded)),
                        },
                    )
                manifest = {
                    "schema": _SCHEMA,
                    "version": _VERSION,
                    "producer": {"name": "flagquantum", "version": __version__},
                    "program_content_hash": ir.content_hash,
                    "members": writer.members,
                }
                # The manifest describes payload members, never itself.
                archive.writestr(_info("manifest.json"), json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _compiler(value: object) -> dict[str, str]:
    data = fields(value, {"pipeline", "flagquantum_version", "ir_version"}, "compiler")
    if any(not isinstance(item, str) or not item for item in data.values()):
        raise SerializationError("invalid compiler identity")
    return {key: str(item) for key, item in data.items()}


def _verify(reader: Reader, manifest: dict[str, object]) -> None:
    table = _mapping(manifest["members"], "member table")
    if "circuit.json" not in table or set(reader.archive.namelist()) != set(table) | {
        "manifest.json"
    }:
        raise SerializationError("FQX has missing or unlisted members")
    for name, record in table.items():
        if (
            name not in ("circuit.json", "result.json")
            and _ARRAY.fullmatch(name) is None
        ):
            raise SerializationError(f"unknown FQX member: {name}")
        member = fields(record, {"size", "sha256"}, "member")
        size, digest = member["size"], member["sha256"]
        if (
            type(size) is not int
            or not isinstance(digest, str)
            or _HASH.fullmatch(digest) is None
            or size != reader.archive.getinfo(name).file_size
        ):
            raise SerializationError(f"invalid FQX member descriptor: {name}")
        if name.endswith(".json") and size > MAX_JSON_BYTES:
            raise SerializationError("FQX JSON member exceeds 16 MiB")
        actual = hashlib.sha256()
        with reader.archive.open(name) as stream:
            while block := stream.read(1024**2):
                actual.update(block)
        if actual.hexdigest() != digest:
            raise SerializationError(f"FQX checksum differs: {name}")


def _optional_tensor(value: object) -> torch.Tensor | None:
    if value is not None and not isinstance(value, torch.Tensor):
        raise SerializationError("result array must be a tensor or null")
    return value


def _measurement(value: object, n_qubits: int) -> MeasurementResult:
    record = fields(
        value,
        {"kind", "qubits", "value", "shots", "metadata", "statistics"},
        "measurement",
    )
    kind, qubits, numeric, shots = (
        record["kind"],
        record["qubits"],
        record["value"],
        record["shots"],
    )
    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(qubits, tuple)
        or any(type(q) is not int or not 0 <= q < n_qubits for q in qubits)
        or len(set(qubits)) != len(qubits)
    ):
        raise SerializationError("invalid measurement kind or qubit order")
    if shots is not None and (type(shots) is not int or shots <= 0):
        raise SerializationError("measurement shots must be a positive integer or null")
    if (
        isinstance(numeric, torch.Tensor)
        or isinstance(numeric, list)
        and all(
            isinstance(batch, dict)
            and all(
                type(k) in (str, int) and type(v) is int and v >= 0
                for k, v in batch.items()
            )
            for batch in numeric
        )
    ):
        measured = numeric
    else:
        raise SerializationError("measurement value is not a tensor or counts batches")
    return MeasurementResult(
        kind=kind,
        qubits=qubits,
        value=measured,
        shots=shots,
        metadata=_mapping(record["metadata"], "measurement metadata"),
        statistics=_mapping(record["statistics"], "measurement statistics"),
    )


def _accuracy(value: object) -> AccuracyContract:
    data = fields(
        value, {"version", "kind", "metric", "value", "tolerance", "passed"}, "accuracy"
    )
    if (
        data["version"] != "1.0"
        or data["kind"] != "accuracy"
        or not isinstance(data["metric"], str)
    ):
        raise SerializationError("unsupported accuracy record")
    scalars: list[float | None] = []
    for key in ("value", "tolerance"):
        scalar = data[key]
        if scalar is not None and (
            not isinstance(scalar, (int, float)) or isinstance(scalar, bool)
        ):
            raise SerializationError("accuracy values must be numeric or null")
        scalars.append(None if scalar is None else float(scalar))
    passed = data["passed"]
    if passed is not None and not isinstance(passed, bool):
        raise SerializationError("accuracy passed must be a bool or null")
    return AccuracyContract(
        metric=data["metric"], value=scalars[0], tolerance=scalars[1], passed=passed
    )


def _restore_result(
    reader: Reader, payload: object, program: CircuitIR
) -> tuple[ExecutionResult, str, str]:
    record = fields(
        payload, {"schema", "version", "execution_plan", "compiler", "data"}, "result"
    )
    if record["schema"] != _SCHEMA + ".result" or record["version"] != _VERSION:
        raise SerializationError("unsupported FQX result schema or version")
    compiler = _compiler(record["compiler"])
    plan = _static_plan(record["execution_plan"], compiler)
    _binding(program, plan)
    data = fields(
        reader.decode(record["data"]),
        {
            "value",
            "state",
            "samples",
            "measurements",
            "accuracy",
            "metrics",
            "provenance",
            "runtime",
            "compatibility",
        },
        "result data",
    )
    measurements = data["measurements"]
    if not isinstance(measurements, list):
        raise SerializationError("measurements must be an array")
    result = ExecutionResult(
        value=_optional_tensor(data["value"]),
        state=_optional_tensor(data["state"]),
        samples=_optional_tensor(data["samples"]),
        measurements=tuple(
            _measurement(item, program.n_wires) for item in measurements
        ),
        accuracy=_accuracy(data["accuracy"]),
        metrics=_mapping(data["metrics"], "metrics"),
        provenance=_mapping(data["provenance"], "provenance"),
        runtime=_mapping(data["runtime"], "runtime"),
        compatibility=_mapping(data["compatibility"], "compatibility"),
    )
    return result, json_bytes(plan).decode(), json_bytes(compiler).decode()


def load(path: str | os.PathLike[str]) -> Experiment:
    """Read a validated experiment without executing code or restoring a runtime.

    File contents, not the filename suffix, determine the format. Tensors are
    restored as owned CPU data; a historical plan is data, never execution authority.

    Raises:
        SerializationError: Malformed, unsupported, oversized or corrupt archive.
        OSError: The local file cannot be opened.

    Examples:
        >>> import flagquantum as fq
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "program.fqx"
        ...     fq.save(path, fq.Circuit(1).h(0))
        ...     experiment = fq.load(path)
        >>> experiment.result is None
        True
    """
    try:
        with _path(path).open("rb") as stream:
            preflight(stream)
            with zipfile.ZipFile(stream) as archive:
                reader = Reader(archive)
                manifest = fields(
                    read_json(reader.read("manifest.json")),
                    {
                        "schema",
                        "version",
                        "producer",
                        "program_content_hash",
                        "members",
                    },
                    "manifest",
                )
                if manifest["schema"] != _SCHEMA or manifest["version"] != _VERSION:
                    raise SerializationError("unsupported FQX schema or version")
                producer = fields(manifest["producer"], {"name", "version"}, "producer")
                if (
                    producer["name"] != "flagquantum"
                    or not isinstance(producer["version"], str)
                    or not producer["version"]
                ):
                    raise SerializationError("invalid FQX producer")
                _verify(reader, manifest)
                program = CircuitIR.from_dict(
                    _mapping(read_json(reader.read("circuit.json")), "circuit")
                )
                if program.content_hash != manifest["program_content_hash"]:
                    raise SerializationError("FQX circuit content hash differs")
                result = None
                plan_json = compiler_json = None
                if "result.json" in archive.namelist():
                    result, plan_json, compiler_json = _restore_result(
                        reader, read_json(reader.read("result.json")), program
                    )
                if reader.used != {
                    name for name in archive.namelist() if name.startswith("arrays/")
                }:
                    raise SerializationError("FQX contains unreferenced tensor data")
                return Experiment(program, result, plan_json, compiler_json)
    except (
        zipfile.BadZipFile,
        zlib.error,
        KeyError,
        EOFError,
        RecursionError,
    ) as error:
        raise SerializationError(f"invalid FQX archive: {error}") from error
    except (TypeError, ValueError) as error:
        if isinstance(error, SerializationError):
            raise
        raise SerializationError(f"invalid FQX data: {error}") from error
