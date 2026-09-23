"""Explicit Qiskit Aer execution for FlagQuantum programs.

The package is an optional backend plugin rather than an interoperability
adapter: Qiskit objects stay inside this module and only FlagQuantum-owned IR
and execution results cross its boundary.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
from typing import Any, Literal

import torch

from flagquantum.core import CircuitIR, ensure_circuit_ir, parameter_names_in_value
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)
from flagquantum.runtime import ExecutionResult, MeasurementResult

OutputKind = Literal["statevector", "samples", "counts"]


class QiskitAerDependencyError(ImportError):
    """Raised when explicit Aer execution is requested without its dependencies."""


class QiskitAerExecutionError(RuntimeError):
    """Raised when a program is outside the explicit Aer bridge contract."""


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        qiskit = import_module("qiskit")
        aer = import_module("qiskit_aer")
        conversion = import_module("flagquantum.ecosystem.qiskit")
    except ImportError as exc:
        raise QiskitAerDependencyError(
            "Qiskit Aer execution requires the optional dependency; install it "
            "with `pip install 'flagquantum[qiskit]'`."
        ) from exc
    return (
        qiskit,
        aer.AerSimulator,
        conversion.export_qiskit,
        conversion.qiskit_statevector_to_flagquantum,
    )


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a bool")
    try:
        normalized = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return int(normalized)


def _optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("seed must be an integer or None, not a bool")
    try:
        return int(operator.index(value))
    except TypeError:
        raise TypeError("seed must be an integer or None") from None


def _selected_wires(wires: Sequence[int] | None, n_wires: int) -> tuple[int, ...]:
    if wires is None:
        return tuple(range(n_wires))
    normalized: list[int] = []
    for wire in wires:
        if isinstance(wire, bool):
            raise TypeError("wires must contain integers, not bools")
        try:
            normalized.append(int(operator.index(wire)))
        except TypeError:
            raise TypeError("wires must contain integers") from None
    selected = tuple(normalized)
    if not selected:
        raise ValueError("wires must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("wires must be unique")
    if any(wire < 0 or wire >= n_wires for wire in selected):
        raise ValueError(f"wires must be in range [0, {n_wires})")
    return selected


def _parameter_names(ir: CircuitIR) -> tuple[str, ...]:
    names: set[str] = set()
    for instruction in ir.instructions:
        names.update(parameter_names_in_value(instruction.params))
        names.update(parameter_names_in_value(instruction.matrix))
    return tuple(sorted(names))


def _requires_grad(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, Mapping):
        return any(_requires_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_requires_grad(item) for item in value)
    return False


def _validate_program(program: Any) -> CircuitIR:
    ir = ensure_circuit_ir(program)
    if ir.observables or ir.measurements:
        raise QiskitAerExecutionError(
            "the Aer bridge accepts execution output through its output argument; "
            "CircuitIR observables and measurements are unsupported"
        )
    names = _parameter_names(ir)
    if names:
        raise QiskitAerExecutionError(
            "bind symbolic parameters before Aer execution: " + ", ".join(names)
        )
    for instruction in ir.instructions:
        if instruction.name in {"measure", "reset"} or any(
            key in instruction.metadata for key in ("condition", "conditions")
        ):
            raise QiskitAerExecutionError(
                "the Aer bridge does not support dynamic circuit instructions"
            )
        if _requires_grad(instruction.params) or _requires_grad(instruction.matrix):
            raise QiskitAerExecutionError(
                "the Aer bridge does not support autograd; detach trainable values "
                "before execution"
            )
    dimension = 2**ir.n_wires
    supported_shapes = {(dimension,), (1, dimension)}
    if ir.shape not in supported_shapes:
        raise QiskitAerExecutionError(
            "the Aer bridge currently accepts one unbatched state; "
            f"expected IR shape ({dimension},) or (1, {dimension}), got {ir.shape}"
        )
    return ir


def _provenance(
    ir: CircuitIR,
    *,
    qiskit: Any,
    report: Any,
) -> dict[str, Any]:
    return {
        "backend": "qiskit_aer",
        "qiskit_version": str(qiskit.__version__),
        "qiskit_aer_version": metadata.version("qiskit-aer"),
        "flagquantum_ir_hash": ir.content_hash,
        "conversion_report": report.to_dict(),
    }


def run(
    program: Any,
    *,
    output: OutputKind = "statevector",
    wires: Sequence[int] | None = None,
    shots: int | None = None,
    seed: int | None = None,
    threads: int = 1,
) -> ExecutionResult:
    """Execute one FlagQuantum program explicitly on local Qiskit Aer.

    This path never replaces or falls back to FlagQuantum's native runtime.
    Statevector execution is exact and shot execution supports computational-
    basis samples or counts on an ordered wire selection.

    Args:
        program: A FlagQuantum circuit or unbatched, fully bound ``CircuitIR``.
        output: ``"statevector"``, ``"samples"``, or ``"counts"``.
        wires: Ordered sampled wires. Omit to sample every wire.
        shots: Positive shot count for samples or counts.
        seed: Optional Aer simulator and transpiler seed.
        threads: Positive Aer CPU thread limit.

    Returns:
        A FlagQuantum ``ExecutionResult`` with explicit backend provenance.

    Raises:
        QiskitAerDependencyError: If Qiskit Aer is not installed.
        QiskitAerExecutionError: If the program or output is unsupported.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum_qiskit_aer import run
        >>> result = run(fq.Circuit(2).h(0).cx(0, 1))
        >>> result.state.shape
        torch.Size([1, 4])
    """

    if output not in {"statevector", "samples", "counts"}:
        raise ValueError("output must be 'statevector', 'samples', or 'counts'")
    thread_count = _positive_integer(threads, name="threads")
    normalized_seed = _optional_seed(seed)
    ir = _validate_program(program)
    selected: tuple[int, ...] | None = None
    shot_count: int | None = None
    if output == "statevector":
        if shots is not None:
            raise ValueError("shots is only valid for samples or counts")
        if wires is not None:
            raise ValueError("wires is only valid for samples or counts")
    else:
        if shots is None:
            raise ValueError("samples and counts require shots")
        shot_count = _positive_integer(shots, name="shots")
        selected = _selected_wires(wires, ir.n_wires)

    qiskit, simulator_type, export_qiskit, convert_statevector = _dependencies()
    exported = export_qiskit(ir)
    circuit = exported.circuit.copy()
    precision = "single" if ir.dtype == "complex64" else "double"
    simulator = simulator_type(
        method="statevector",
        device="CPU",
        precision=precision,
        max_parallel_threads=thread_count,
    )
    runtime = {
        "backend": "qiskit_aer",
        "device": "cpu",
        "threads": thread_count,
        "output": output,
        "fallback_used": False,
    }
    compatibility = {
        "autograd_supported": False,
        "fallback_used": False,
        "external_backend_explicit": True,
    }
    provenance = _provenance(
        ir,
        qiskit=qiskit,
        report=exported.report,
    )

    if output == "statevector":
        circuit.save_statevector()
        compiled = qiskit.transpile(
            circuit,
            simulator,
            optimization_level=0,
            seed_transpiler=normalized_seed,
        )
        native = simulator.run(
            compiled,
            seed_simulator=normalized_seed,
        ).result()
        state = convert_statevector(
            native.get_statevector(compiled).data,
            ir.n_wires,
        ).to(dtype=getattr(torch, ir.dtype))
        return ExecutionResult(
            state=state.unsqueeze(0),
            metrics={"n_wires": ir.n_wires},
            provenance=provenance,
            runtime=runtime,
            compatibility=compatibility,
        )

    assert shot_count is not None and selected is not None
    classical_register = qiskit.ClassicalRegister(len(selected), "fq_readout")
    circuit.add_register(classical_register)
    for index, wire in enumerate(selected):
        circuit.measure(wire, classical_register[index])
    compiled = qiskit.transpile(
        circuit,
        simulator,
        optimization_level=0,
        seed_transpiler=normalized_seed,
    )
    native = simulator.run(
        compiled,
        shots=shot_count,
        memory=True,
        seed_simulator=normalized_seed,
    ).result()
    memory = native.get_memory(compiled)
    rows = [[int(bit) for bit in item.replace(" ", "")[::-1]] for item in memory]
    samples = torch.tensor([rows], dtype=torch.int64)
    measurement_metadata = {
        "backend": "qiskit_aer",
        "wire_order": "flagquantum",
    }
    if output == "samples":
        measurement = MeasurementResult(
            kind="samples",
            wires=selected,
            value=samples,
            shots=shot_count,
            metadata=measurement_metadata,
        )
        return ExecutionResult(
            samples=samples,
            measurements=(measurement,),
            metrics={"n_wires": ir.n_wires, "shots": shot_count},
            provenance=provenance,
            runtime=runtime,
            compatibility=compatibility,
        )
    encoded = ["".join(str(bit) for bit in row) for row in rows]
    counts = dict(Counter(encoded))
    measurement = MeasurementResult(
        kind="counts",
        wires=selected,
        value=[counts],
        shots=shot_count,
        metadata=measurement_metadata,
    )
    return ExecutionResult(
        measurements=(measurement,),
        metrics={"n_wires": ir.n_wires, "shots": shot_count},
        provenance=provenance,
        runtime=runtime,
        compatibility=compatibility,
    )


class QiskitAerBackend:
    """Execution-backend extension for explicit local Qiskit Aer use."""

    manifest = ExtensionManifest(
        name="qiskit_aer",
        version="0.1.0",
        kind="backend",
        capabilities=frozenset(
            {
                "complex64",
                "complex128",
                "counts",
                "cpu",
                "samples",
                "statevector",
            }
        ),
    )

    def __init__(self) -> None:
        self._active = False
        self._defaults: dict[str, Any] = {}

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        blockers: list[str] = []
        if request.dtype not in {None, "complex64", "complex128"}:
            blockers.append(f"dtype {request.dtype!r} is unsupported")
        if request.device_type not in {None, "cpu"}:
            blockers.append(f"device {request.device_type!r} is unsupported")
        if request.require_gradients:
            blockers.append("Qiskit Aer bridge does not support gradients")
        missing = request.required - self.manifest.capabilities
        if missing:
            blockers.append("missing capabilities: " + ", ".join(sorted(missing)))
        return CapabilityResponse(
            accepted=not blockers,
            supported=self.manifest.capabilities,
            blockers=tuple(blockers),
        )

    def start(self, config: ExtensionConfig) -> None:
        unknown = set(config.values) - {"seed", "threads"}
        if unknown:
            raise ValueError(
                "unsupported Qiskit Aer configuration: " + ", ".join(sorted(unknown))
            )
        self._defaults = dict(config.values)
        if "threads" in self._defaults:
            self._defaults["threads"] = _positive_integer(
                self._defaults["threads"], name="threads"
            )
        if "seed" in self._defaults:
            self._defaults["seed"] = _optional_seed(self._defaults["seed"])
        self._active = True

    def execute(self, program: Any, parameters: Any = None) -> ExecutionResult:
        if not self._active:
            raise RuntimeError("Qiskit Aer backend is not active")
        if parameters is None:
            options: dict[str, Any] = {}
        elif isinstance(parameters, Mapping):
            options = dict(parameters)
        else:
            raise TypeError("execution parameters must be a mapping or None")
        return run(program, **{**self._defaults, **options})

    def close(self) -> None:
        self._defaults.clear()
        self._active = False


def create_extension() -> QiskitAerBackend:
    """Create the extension instance used by entry-point discovery."""

    return QiskitAerBackend()


__all__ = (
    "OutputKind",
    "QiskitAerBackend",
    "QiskitAerDependencyError",
    "QiskitAerExecutionError",
    "create_extension",
    "run",
)
