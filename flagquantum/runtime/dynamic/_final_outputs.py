"""Admission and result projection for local dynamic final readout."""

from __future__ import annotations

from dataclasses import replace

from ...core.ir import CircuitIR, MeasurementNode
from ...errors import CapabilityError, ExecutionError, ValidationError
from ..options import ExecutionOptions
from ..result import ExecutionResult, MeasurementResult

_FINAL_OUTPUT = "_fq_dynamic_final_output"
_REGISTER_KIND = "mid_circuit_measurement"
_OUTPUT_METADATA = {
    "fq_output_index",
    "fq_output_kind",
    "fq_output_name",
    "name",
    "seed",
}


def _validate_requests(
    source: CircuitIR, requests: tuple[MeasurementNode, ...]
) -> None:
    last_operation = next(
        (
            op
            for op in reversed(source.instructions)
            if op.name not in {"barrier", "tick"}
        ),
        None,
    )
    if last_operation is not None and last_operation.name == "measure":
        raise CapabilityError(
            "dynamic final outputs use the existing final readout and do not support "
            "an explicit terminal measurement"
        )
    kinds = {request.kind for request in requests}
    if (
        not kinds
        or not kinds <= {"sample", "counts"}
        or len(kinds) != len(requests)
        or any(
            request.wires != tuple(range(source.n_wires))
            or request.metadata.keys() - _OUTPUT_METADATA - {_FINAL_OUTPUT}
            for request in requests
        )
    ):
        raise CapabilityError(
            "dynamic planning does not accept outputs other than one full-register "
            "Z-basis samples request and one counts request in logical qubit order"
        )
    names = [request.metadata.get("name") for request in requests]
    if any(
        name is not None and (not isinstance(name, str) or not name) for name in names
    ):
        raise ValidationError("dynamic final output names must be non-empty strings")
    names = [name for name in names if name is not None]
    if _REGISTER_KIND in names or len(set(names)) != len(names):
        raise ValidationError(
            "dynamic final output names must be unique and unreserved"
        )
    shots = requests[0].shots
    if (
        shots is None
        or shots < 1
        or any(request.shots != shots for request in requests)
    ):
        raise ValidationError(
            "dynamic final outputs require the same positive shot count"
        )


def mark_dynamic_final_outputs(
    source: CircuitIR, requests: tuple[MeasurementNode, ...]
) -> tuple[MeasurementNode, ...]:
    _validate_requests(source, requests)
    return tuple(
        replace(request, metadata={**request.metadata, _FINAL_OUTPUT: True})
        for request in requests
    )


def _explicit_requests(source: CircuitIR) -> tuple[MeasurementNode, ...]:
    # Older dynamic plans can carry IR measurements without requesting projection.
    requests = source.measurements
    if not any(_FINAL_OUTPUT in request.metadata for request in requests):
        return ()
    if any(request.metadata.get(_FINAL_OUTPUT) is not True for request in requests):
        raise ValidationError(
            "dynamic final output requests must all be explicitly marked"
        )
    return requests


def resolve_dynamic_final_output_options(
    source: CircuitIR, options: ExecutionOptions | None
) -> ExecutionOptions | None:
    requests = _explicit_requests(source)
    if not requests:
        return options
    options = options or ExecutionOptions()
    if options.shots is not None and options.shots != requests[0].shots:
        raise ValidationError(
            "dynamic final output shots disagree with execution options"
        )
    return replace(options, shots=requests[0].shots)


def validate_dynamic_final_outputs(
    source: CircuitIR, *, shots: int | None, batch_size: int
) -> tuple[MeasurementNode, ...]:
    requests = _explicit_requests(source)
    if requests:
        _validate_requests(source, requests)
        if batch_size != 1:
            raise CapabilityError(
                "dynamic final outputs require a single circuit batch"
            )
        if requests[0].shots != shots:
            raise ValidationError("dynamic final output shots disagree with the plan")
    return requests


def project_dynamic_final_outputs(
    result: ExecutionResult, requests: tuple[MeasurementNode, ...]
) -> ExecutionResult:
    if not requests:
        return result

    from ..measurements import _counts_from_samples

    samples = result.samples
    if samples is None or samples.shape != (requests[0].shots, len(requests[0].wires)):
        raise ExecutionError("dynamic final sample shape disagrees with the plan")
    outputs = tuple(
        MeasurementResult(
            kind=request.kind,
            qubits=request.wires,
            value=(
                samples
                if request.kind == "sample"
                else _counts_from_samples(samples.unsqueeze(0), format="bin")
            ),
            shots=request.shots,
            metadata={
                **{
                    key: value
                    for key, value in request.metadata.items()
                    if key != _FINAL_OUTPUT
                },
                "stage": "final",
            },
        )
        for request in requests
    )
    register = tuple(
        item for item in result.measurements if item.kind == _REGISTER_KIND
    )
    return replace(result, measurements=outputs + register)
