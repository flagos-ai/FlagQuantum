"""Explicit Qiskit Aer execution for FlagQuantum programs.

The package is an optional backend plugin rather than an interoperability
adapter: Qiskit objects stay inside this module and only FlagQuantum-owned IR
and execution results cross its boundary.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib import import_module, metadata
from typing import Any, Literal, cast

import torch

from flagquantum.core import CircuitIR, ensure_circuit_ir, parameter_names_in_value
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)
from flagquantum.observables import OutputRequest
from flagquantum.runtime import ExecutionOptions, ExecutionResult, MeasurementResult

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
    options: ExecutionOptions | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    shots: int | None = None,
) -> ExecutionResult:
    """Execute one FlagQuantum program explicitly on local Qiskit Aer.

    The expression mirrors ``fq.run``: result selection uses ``fq.samples`` or
    ``fq.counts``, while shots and seed use ``fq.ExecutionOptions``. This path
    never replaces or falls back to FlagQuantum's native runtime.

    Args:
        program: A FlagQuantum circuit or unbatched, fully bound ``CircuitIR``.
        options: Backend-neutral FlagQuantum execution options.
        outputs: One ``fq.samples`` or ``fq.counts`` request. Omit for an exact
            statevector.
        shots: Optional native-style shorthand for the shot count.

    Returns:
        A FlagQuantum ``ExecutionResult`` with explicit backend provenance.

    Raises:
        QiskitAerDependencyError: If Qiskit Aer is not installed.
        QiskitAerExecutionError: If the program or output is unsupported.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.ecosystem.qiskit import run
        >>> result = run(fq.Circuit(2).h(0).cx(0, 1))  # doctest: +SKIP
        >>> result.state.shape  # doctest: +SKIP
        torch.Size([1, 4])
    """

    return _run_with_resources(
        program,
        options=options,
        outputs=outputs,
        shots=shots,
        threads=None,
    )


def _normalize_request(
    *,
    options: ExecutionOptions | None,
    outputs: OutputRequest | Sequence[OutputRequest] | None,
    shots: int | None,
) -> tuple[OutputKind, Sequence[int] | None, int | None, int | None]:
    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an ExecutionOptions instance or None")
    if shots is not None and options is not None and options.shots is not None:
        raise TypeError("shots was specified both directly and in ExecutionOptions")
    if outputs is None:
        if shots is not None:
            raise TypeError(
                "local shots requires fq.samples(...) or fq.counts(...) output"
            )
        seed = options.seed if options is not None else None
        return "statevector", None, None, seed
    requests: tuple[OutputRequest, ...]
    if isinstance(outputs, OutputRequest):
        requests = (outputs,)
    elif isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        requests = tuple(outputs)
    else:
        raise TypeError("outputs must be an OutputRequest, a sequence, or None")
    if len(requests) != 1:
        raise QiskitAerExecutionError(
            "the Qiskit Aer bridge currently accepts exactly one output request"
        )
    request = requests[0]
    if not isinstance(request, OutputRequest):
        raise TypeError("outputs must contain only OutputRequest instances")
    if request.kind not in {"samples", "counts"}:
        raise QiskitAerExecutionError(
            "the Qiskit Aer bridge currently supports fq.samples or fq.counts"
        )
    if request.observable is not None:
        raise QiskitAerExecutionError(
            "the Qiskit Aer bridge currently supports computational-basis outputs"
        )
    selected_shots = (
        shots if shots is not None else options.shots if options is not None else None
    )
    if selected_shots is None:
        raise ValueError("samples and counts require shots")
    return (
        cast(OutputKind, request.kind),
        request.wires or None,
        selected_shots,
        options.seed if options is not None else None,
    )


def _validate_options(options: ExecutionOptions | None, ir: CircuitIR) -> None:
    if options is None:
        return
    if options.backend not in {None, "qiskit_aer"}:
        raise QiskitAerExecutionError("options.backend must be 'qiskit_aer'")
    if options.device not in {None, "cpu"}:
        raise QiskitAerExecutionError("Qiskit Aer execution requires device='cpu'")
    if options.mode not in {None, "auto", "statevector"}:
        raise QiskitAerExecutionError(
            "Qiskit Aer execution supports mode='auto' or 'statevector'"
        )
    if options.precision is not None and options.precision != ir.dtype:
        raise QiskitAerExecutionError(
            f"options.precision {options.precision!r} does not match {ir.dtype!r}"
        )
    if options.batch_size not in {None, 1}:
        raise QiskitAerExecutionError("Qiskit Aer execution accepts one batch item")
    if options.target is not None or options.memory_limit_bytes is not None:
        raise QiskitAerExecutionError(
            "options.target and memory_limit_bytes are unsupported"
        )
    if options.require_gradients:
        raise QiskitAerExecutionError("Qiskit Aer bridge does not support gradients")
    if options.allow_approximate:
        raise QiskitAerExecutionError(
            "Qiskit Aer bridge does not support approximate execution"
        )
    if options.allow_backend_fallback:
        raise QiskitAerExecutionError("Qiskit Aer bridge does not support fallback")


def _run_with_resources(
    program: Any,
    *,
    options: ExecutionOptions | None,
    outputs: OutputRequest | Sequence[OutputRequest] | None,
    shots: int | None,
    threads: int | None,
) -> ExecutionResult:
    output, wires, requested_shots, seed = _normalize_request(
        options=options,
        outputs=outputs,
        shots=shots,
    )
    thread_count = (
        None if threads is None else _positive_integer(threads, name="threads")
    )
    normalized_seed = _optional_seed(seed)
    ir = _validate_program(program)
    _validate_options(options, ir)
    selected: tuple[int, ...] | None = None
    shot_count: int | None = None
    if output != "statevector":
        assert requested_shots is not None
        shot_count = _positive_integer(requested_shots, name="shots")
        selected = _selected_wires(wires, ir.n_wires)

    qiskit, simulator_type, export_qiskit, convert_statevector = _dependencies()
    exported = export_qiskit(ir)
    circuit = exported.circuit.copy()
    precision = "single" if ir.dtype == "complex64" else "double"
    simulator_options: dict[str, Any] = {
        "method": "statevector",
        "device": "CPU",
        "precision": precision,
    }
    if thread_count is not None:
        simulator_options["max_parallel_threads"] = thread_count
    simulator = simulator_type(**simulator_options)
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
            request: dict[str, Any] = {}
        elif isinstance(parameters, Mapping):
            request = dict(parameters)
        else:
            raise TypeError("execution parameters must be a mapping or None")
        unknown = set(request) - {"options", "outputs", "shots"}
        if unknown:
            raise ValueError(
                "unsupported Qiskit Aer execution parameter(s): "
                + ", ".join(sorted(unknown))
            )
        execution_options = request.get("options")
        if execution_options is not None and not isinstance(
            execution_options, ExecutionOptions
        ):
            raise TypeError("options must be an ExecutionOptions instance or None")
        configured_seed = self._defaults.get("seed")
        if configured_seed is not None and (
            execution_options is None or execution_options.seed is None
        ):
            execution_options = replace(
                execution_options or ExecutionOptions(), seed=configured_seed
            )
        return _run_with_resources(
            program,
            options=execution_options,
            outputs=request.get("outputs"),
            shots=request.get("shots"),
            threads=self._defaults.get("threads"),
        )

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
