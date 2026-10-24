"""Native Ising admission behind the existing executable-plan boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from ..core.ir import _decode_value, _encode_value
from ..core.ir.program.physical import IsingContext
from ..core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from ..core.ising import IsingBraidProgram
from ..errors import CapabilityError, ValidationError
from ..observables import OutputRequest
from ..simulation.lindblad_request import _normalize_observables
from .execution_plan import ExecutionPlan
from .execution_plan_contract import (
    EXECUTION_PLAN_SCHEMA,
    ExecutionPlanContractError,
    _compiler_identity,
    _environment_requirements,
    _identity,
    _require_mapping,
    _resolved_mapping,
    _runtime_config_manifest,
    _validate_plan_options,
    canonical_hash,
    canonical_json,
)
from .options import ExecutionOptions
from .options_resolver import resolve_execution_options
from .result import ExecutionResult, MeasurementResult


def _outputs(
    outputs: OutputRequest | Sequence[OutputRequest] | None,
) -> tuple[OutputRequest, ...]:
    if outputs is None:
        return ()
    requests = (outputs,) if isinstance(outputs, OutputRequest) else tuple(outputs)
    if not requests or any(not isinstance(item, OutputRequest) for item in requests):
        raise TypeError("outputs must be an OutputRequest or a non-empty sequence")
    names = tuple(item.name for item in requests if item.name is not None)
    if len(set(names)) != len(names):
        raise ValidationError("output names must be unique")
    captured = []
    for item in requests:
        matrix = item.observable
        if item.kind != "expectation" or not isinstance(matrix, torch.Tensor):
            raise CapabilityError(
                "Ising outputs require full-space matrix expectations"
            )
        if (
            matrix.dtype != torch.complex128
            or matrix.device.type != "cpu"
            or matrix.requires_grad
        ):
            raise CapabilityError("Ising output matrices require static CPU complex128")
        checked = _normalize_observables(
            {"output": matrix},
            dim=2,
            dtype=torch.complex128,
            device=torch.device("cpu"),
            qubit_space=False,
        )["output"]
        captured.append(
            OutputRequest("expectation", observable=checked.clone(), name=item.name)
        )
    return tuple(captured)


def plan_ising(
    program: IsingBraidProgram,
    *,
    options: ExecutionOptions | None = None,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
) -> ExecutionPlan:
    """Seal a validated native source using the shared options resolver."""
    record = program.to_ir()
    source = native_program_to_dict(record)
    if source.get("version") not in ("1.1", "1.2"):
        raise CapabilityError("Ising execution supports only native source 1.1 or 1.2")
    requests = _outputs(outputs)
    try:
        resolved = resolve_execution_options(
            options, program_constraints=ExecutionOptions(precision="complex128")
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if (
        resolved.mode not in {"auto", "density_matrix"}
        or resolved.backend not in {"auto", "pytorch"}
        or resolved.device not in {"auto", "cpu"}
        or resolved.target not in {"auto", "state", "expectation"}
        or resolved.batch_size != 1
        or resolved.shots is not None
        or resolved.require_gradients
        or resolved.allow_approximate
        or resolved.allow_backend_fallback
    ):
        raise CapabilityError(
            "Ising execution requires exact single-batch CPU density actions"
        )
    if resolved.target == "expectation" and not requests:
        raise ValidationError("expectation target requires output requests")
    if resolved.memory_limit_bytes is not None and resolved.memory_limit_bytes < 64:
        raise CapabilityError("Ising density state requires 64 bytes")
    from .distributed.backend_policy import resolve_distributed_backend_policy

    if resolve_distributed_backend_policy().effective_world_size != 1:
        raise CapabilityError("Ising execution requires world_size=1")
    resolved = replace(
        resolved,
        mode="density_matrix",
        backend="pytorch",
        device="cpu",
        target=(
            ("expectation" if requests else "state")
            if resolved.target == "auto"
            else resolved.target
        ),
    )
    decision = {
        name: getattr(resolved, name)
        for name in (
            "mode",
            "backend",
            "device",
            "target",
            "batch_size",
            "precision",
            "require_gradients",
            "allow_approximate",
            "allow_backend_fallback",
            "memory_limit_bytes",
        )
    }
    decision.update(world_size=1, state_bytes=64, recommended_mode="density_matrix")
    concrete = _resolved_mapping(resolved)
    environment = _environment_requirements(decision)
    payload = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "version": source["version"],
        "identity": "",
        "program": source,
        "requested_options": (options or ExecutionOptions()).to_dict(),
        "resolved_options": concrete,
        "fingerprints": {
            "program": canonical_hash(source),
            "options": canonical_hash(concrete),
            "environment": canonical_hash(environment),
            "compiler": canonical_hash(_compiler_identity()),
        },
        "environment_requirements": environment,
        "decision": decision,
        "extensions": [],
        "outputs": [
            {
                "kind": "expectation",
                "name": item.name,
                "operator": _encode_value(item.observable),
            }
            for item in requests
        ],
    }
    payload["identity"] = _identity(payload, compiler=_compiler_identity())
    from .execution_plan_contract import plan_from_dict

    return plan_from_dict(payload)


def restore_ising_plan(payload: Mapping[str, Any]) -> ExecutionPlan:
    """Restore an already validated native payload without planning."""
    decision = payload["decision"]
    return ExecutionPlan(
        analysis=None,
        layers=(),
        state_bytes=64,
        recommended_mode="density_matrix",
        world_size=1,
        shardable_wires=(),
        state_mode="density_matrix",
        runtime_config=_runtime_config_manifest(decision),
        _contract_payload_json=canonical_json(payload),
    )


def _decode_operator(encoded: object) -> torch.Tensor:
    """Bound the reviewed tensor layout before the shared Core decoder allocates."""
    if not isinstance(encoded, Mapping) or set(encoded) != {"$tensor"}:
        raise ValueError("native output requires a tensor encoding")
    tensor = encoded["$tensor"]
    if not isinstance(tensor, Mapping) or set(tensor) != {"dtype", "shape", "data"}:
        raise ValueError("native tensor fields must be dtype, shape and data")
    shape = tensor["shape"]
    if (
        tensor["dtype"] != "complex128"
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(size) is not int or size != 2 for size in shape)
    ):
        raise ValueError("native output encoding requires shape [2, 2] and complex128")
    data = tensor["data"]
    if not isinstance(data, list) or len(data) != 2:
        raise ValueError("native output encoding requires two matrix rows")
    for row in data:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("native output encoding requires two matrix columns")
        for cell in row:
            if not isinstance(cell, Mapping) or set(cell) != {"$complex"}:
                raise ValueError(
                    "native output entries require the Core complex encoding"
                )
            pair = cell["$complex"]
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(part) not in (int, float) for part in pair)
            ):
                raise ValueError("native complex entries require two real numbers")
    with torch.device("cpu"):
        decoded = _decode_value(encoded)
    if not isinstance(decoded, torch.Tensor):
        raise ValueError("native output did not decode to a tensor")
    return decoded


def _read_outputs(payload: Mapping[str, Any]) -> tuple[OutputRequest, ...]:
    raw = payload["outputs"]
    if not isinstance(raw, list):
        raise ValueError("native outputs must be an ordered JSON array")
    requests = []
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"kind", "name", "operator"}
            or item["kind"] != "expectation"
        ):
            raise ValueError("native output fields must be kind, name and operator")
        requests.append(
            OutputRequest(
                "expectation",
                name=item["name"],
                observable=_decode_operator(item["operator"]),
            )
        )
    return _outputs(requests) if requests else ()


def validate_ising_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    """Apply native semantics inside the existing versioned plan reader."""
    from .execution_plan_contract import (
        _DECISION_FIELDS,
        _FINGERPRINT_NAMES,
        _require_mapping,
        _validate_decision,
        _validate_plan_identity,
    )

    try:
        source = _require_mapping("program", values["program"])
        if source.get("version") != values["version"]:
            raise ValueError(
                f"plan {values['version']} requires native Ising source {values['version']}"
            )
        record = native_program_from_dict(source)
        if not isinstance(record.physical_context, IsingContext):
            raise ValueError("native plan requires the native Ising source")
    except (TypeError, ValueError) as exc:
        raise ExecutionPlanContractError(
            "program_fingerprint_mismatch", str(exc)
        ) from exc
    try:
        requests = _read_outputs(values)
    except (TypeError, ValueError, OverflowError, CapabilityError) as exc:
        raise ExecutionPlanContractError("identity_mismatch", str(exc)) from exc
    resolved = _validate_plan_options(values)
    expected = ExecutionOptions(
        mode="density_matrix",
        backend="pytorch",
        device="cpu",
        target=resolved["target"],
        batch_size=1,
        precision="complex128",
        shots=None,
        seed=resolved["seed"],
        memory_limit_bytes=resolved["memory_limit_bytes"],
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    ).to_dict()
    expected.pop("schema")
    expected.pop("version")
    if (
        resolved != expected
        or resolved["target"] not in {"state", "expectation"}
        or (resolved["target"] == "expectation" and not requests)
        or (
            resolved["memory_limit_bytes"] is not None
            and resolved["memory_limit_bytes"] < 64
        )
    ):
        raise ExecutionPlanContractError(
            "options_fingerprint_mismatch", "unsupported native resolved options"
        )
    requested = _require_mapping("requested_options", values["requested_options"])
    for name, concrete in resolved.items():
        value = requested[name]
        if value is None or (
            name in {"mode", "backend", "device", "target"} and value == "auto"
        ):
            continue
        if value != concrete:
            raise ExecutionPlanContractError(
                "options_fingerprint_mismatch",
                f"requested {name} contradicts resolved options",
            )
    decision = _require_mapping("decision", values["decision"])
    if set(decision) != _DECISION_FIELDS:
        raise ExecutionPlanContractError(
            "identity_mismatch", "native decision fields differ"
        )
    _validate_decision(decision)
    expected_decision = {
        name: resolved[name]
        for name in _DECISION_FIELDS - {"world_size", "state_bytes", "recommended_mode"}
    }
    expected_decision.update(
        world_size=1, state_bytes=64, recommended_mode="density_matrix"
    )
    if decision != expected_decision:
        raise ExecutionPlanContractError(
            "identity_mismatch", "native decision contradicts fixed source semantics"
        )
    if values["extensions"] != []:
        raise ExecutionPlanContractError(
            "extension_incompatible", "native Ising plans have no extensions"
        )
    fingerprints = _require_mapping("fingerprints", values["fingerprints"])
    if set(fingerprints) != set(_FINGERPRINT_NAMES):
        raise ExecutionPlanContractError(
            "identity_mismatch", "native fingerprint fields differ"
        )
    environment = _require_mapping(
        "environment_requirements", values["environment_requirements"]
    )
    _validate_plan_identity(
        values,
        program=record,
        resolved=resolved,
        fingerprints=fingerprints,
        environment=environment,
        decision=decision,
    )
    import json

    normalized: dict[str, Any] = json.loads(canonical_json(values))
    return normalized


def braid_depth(plan: ExecutionPlan) -> int:
    """Read source length without substituting a circuit analysis."""
    record = native_program_from_dict(
        _require_mapping("program", plan.to_dict()["program"])
    )
    return len(record.functions[0].blocks[0].operations)


def execute_ising_plan(plan: ExecutionPlan) -> ExecutionResult:
    """Assemble the shared result from Simulation's exact finite action."""
    from ..simulation.ising_braid import evolve_ising_braid

    payload = plan.to_dict()
    record = native_program_from_dict(_require_mapping("program", payload["program"]))
    state = evolve_ising_braid(record).unsqueeze(0)
    measurements = []
    values = []
    for index, request in enumerate(_read_outputs(payload)):
        assert isinstance(request.observable, torch.Tensor)
        value = torch.einsum("bij,ji->b", state, request.observable).real
        values.append(value)
        metadata = {
            "fq_output_index": index,
            "fq_output_kind": "expectation",
            "fq_output_name": request.name,
        }
        if request.name is not None:
            metadata["name"] = request.name
        measurements.append(
            MeasurementResult("expectation", (), value, metadata=metadata)
        )
    return ExecutionResult(
        state=state,
        plan=plan,
        value=torch.stack(values, dim=1) if values else None,
        measurements=tuple(measurements),
        runtime={
            "mode": "density_matrix",
            "backend": "pytorch",
            "device": "cpu",
            "precision": "complex128",
            "execution_path": "native_ising_braid",
            "world_size": 1,
            "local_world_size": 1,
            "node_count": 1,
            "distribution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "exact_plan_execution": True,
        },
    )
