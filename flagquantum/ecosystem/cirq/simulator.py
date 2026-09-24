"""Explicit Cirq Simulator execution for FlagQuantum programs."""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal, cast

import torch

from flagquantum.core import CircuitIR, ensure_circuit_ir, parameter_names_in_value
from flagquantum.observables import OutputRequest
from flagquantum.runtime import ExecutionOptions, ExecutionResult, MeasurementResult

OutputKind = Literal["statevector", "samples", "counts"]


class CirqSimulatorDependencyError(ImportError):
    """Raised when explicit Cirq execution is requested without Cirq."""


class CirqSimulatorExecutionError(RuntimeError):
    """Raised when a program is outside the explicit Cirq bridge contract."""


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        cirq = import_module("cirq")
        numpy = import_module("numpy")
        conversion = import_module("flagquantum.ecosystem.cirq.conversion")
    except ImportError as exc:
        raise CirqSimulatorDependencyError(
            "Cirq Simulator execution requires the optional dependency; install it "
            "with `pip install 'flagquantum[cirq]'`."
        ) from exc
    return cirq, numpy, conversion.export_cirq


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
        raise CirqSimulatorExecutionError(
            "the Cirq bridge accepts execution output through its output argument; "
            "CircuitIR observables and measurements are unsupported"
        )
    names = _parameter_names(ir)
    if names:
        raise CirqSimulatorExecutionError(
            "bind symbolic parameters before Cirq execution: " + ", ".join(names)
        )
    for instruction in ir.instructions:
        if instruction.name in {"measure", "reset"} or any(
            key in instruction.metadata for key in ("condition", "conditions")
        ):
            raise CirqSimulatorExecutionError(
                "the Cirq bridge does not support dynamic circuit instructions"
            )
        if _requires_grad(instruction.params) or _requires_grad(instruction.matrix):
            raise CirqSimulatorExecutionError(
                "the Cirq bridge does not support autograd; detach trainable values "
                "before execution"
            )
    dimension = 2**ir.n_wires
    if ir.shape not in {(dimension,), (1, dimension)}:
        raise CirqSimulatorExecutionError(
            "the Cirq bridge currently accepts one unbatched state; "
            f"expected IR shape ({dimension},) or (1, {dimension}), got {ir.shape}"
        )
    return ir


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
        raise CirqSimulatorExecutionError(
            "the Cirq bridge currently accepts exactly one output request"
        )
    request = requests[0]
    if not isinstance(request, OutputRequest):
        raise TypeError("outputs must contain only OutputRequest instances")
    if request.kind not in {"samples", "counts"}:
        raise CirqSimulatorExecutionError(
            "the Cirq bridge currently supports fq.samples or fq.counts"
        )
    if request.observable is not None:
        raise CirqSimulatorExecutionError(
            "the Cirq bridge currently supports computational-basis outputs"
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
    if options.backend not in {None, "cirq_simulator"}:
        raise CirqSimulatorExecutionError("options.backend must be 'cirq_simulator'")
    if options.device not in {None, "cpu"}:
        raise CirqSimulatorExecutionError("Cirq execution requires device='cpu'")
    if options.mode not in {None, "auto", "statevector"}:
        raise CirqSimulatorExecutionError(
            "Cirq execution supports mode='auto' or 'statevector'"
        )
    if options.precision is not None and options.precision != ir.dtype:
        raise CirqSimulatorExecutionError(
            f"options.precision {options.precision!r} does not match {ir.dtype!r}"
        )
    if options.batch_size not in {None, 1}:
        raise CirqSimulatorExecutionError("Cirq execution accepts one batch item")
    if options.target is not None or options.memory_limit_bytes is not None:
        raise CirqSimulatorExecutionError(
            "options.target and memory_limit_bytes are unsupported"
        )
    if options.require_gradients:
        raise CirqSimulatorExecutionError("Cirq bridge does not support gradients")
    if options.allow_approximate:
        raise CirqSimulatorExecutionError(
            "Cirq bridge does not support approximate execution"
        )
    if options.allow_backend_fallback:
        raise CirqSimulatorExecutionError("Cirq bridge does not support fallback")


def _export_for_execution(ir: CircuitIR, export_cirq: Any) -> Any:
    exported = export_cirq(ir, allow_lossy=True)
    unsupported = tuple(
        issue
        for issue in exported.report.issues
        if issue.code != "idle_wire_extent_not_represented"
    )
    if unsupported:
        codes = ", ".join(issue.code for issue in unsupported)
        raise CirqSimulatorExecutionError(
            "the Cirq bridge cannot execute this program losslessly: " + codes
        )
    return exported


def run(
    program: Any,
    *,
    options: ExecutionOptions | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    shots: int | None = None,
) -> ExecutionResult:
    """Execute one FlagQuantum program explicitly on local Cirq Simulator.

    Result selection uses the same ``fq.samples``, ``fq.counts``, and
    ``fq.ExecutionOptions`` expressions as ``fq.run``. This path never replaces
    or falls back to FlagQuantum's native runtime.

    Args:
        program: A FlagQuantum circuit or unbatched, fully bound ``CircuitIR``.
        options: Backend-neutral FlagQuantum execution options.
        outputs: One ``fq.samples`` or ``fq.counts`` request. Omit for an exact
            statevector.
        shots: Optional native-style shorthand for the shot count.

    Returns:
        A FlagQuantum ``ExecutionResult`` with explicit backend provenance.

    Raises:
        CirqSimulatorDependencyError: If Cirq is not installed.
        CirqSimulatorExecutionError: If the program or output is unsupported.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.ecosystem.cirq import run
        >>> result = run(fq.Circuit(2).h(0).cx(0, 1))  # doctest: +SKIP
        >>> result.state.shape  # doctest: +SKIP
        torch.Size([1, 4])
    """

    output, wires, requested_shots, seed = _normalize_request(
        options=options,
        outputs=outputs,
        shots=shots,
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

    cirq, numpy, export_cirq = _dependencies()
    exported = _export_for_execution(ir, export_cirq)
    circuit = exported.circuit.copy()
    qubits = tuple(cirq.LineQubit.range(ir.n_wires))
    simulator = cirq.Simulator(
        dtype=numpy.complex64 if ir.dtype == "complex64" else numpy.complex128,
        seed=normalized_seed,
    )
    idle_extent_preserved = any(
        issue.code == "idle_wire_extent_not_represented"
        for issue in exported.report.issues
    )
    runtime = {
        "backend": "cirq_simulator",
        "device": "cpu",
        "output": output,
        "fallback_used": False,
    }
    compatibility = {
        "autograd_supported": False,
        "fallback_used": False,
        "external_backend_explicit": True,
    }
    provenance = {
        "backend": "cirq_simulator",
        "cirq_version": str(cirq.__version__),
        "flagquantum_ir_hash": ir.content_hash,
        "conversion_report": exported.report.to_dict(),
        "idle_wire_extent_preserved": idle_extent_preserved,
    }

    if output == "statevector":
        native = simulator.simulate(circuit, qubit_order=qubits)
        state = torch.from_numpy(native.final_state_vector.copy()).to(
            dtype=getattr(torch, ir.dtype)
        )
        return ExecutionResult(
            state=state.unsqueeze(0),
            metrics={"n_wires": ir.n_wires},
            provenance=provenance,
            runtime=runtime,
            compatibility=compatibility,
        )

    assert shot_count is not None and selected is not None
    circuit.append(cirq.measure(*(qubits[wire] for wire in selected), key="fq_readout"))
    native = simulator.run(circuit, repetitions=shot_count)
    rows = native.measurements["fq_readout"]
    samples = torch.from_numpy(rows.copy()).to(dtype=torch.int64).unsqueeze(0)
    measurement_metadata = {
        "backend": "cirq_simulator",
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
    encoded = ["".join(str(int(bit)) for bit in row) for row in rows]
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


__all__ = (
    "CirqSimulatorDependencyError",
    "CirqSimulatorExecutionError",
    "OutputKind",
    "run",
)
