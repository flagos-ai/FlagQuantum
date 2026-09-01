"""Versioned identity, serialization, and preflight for executable plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..core.ir import IR_VERSION, CircuitIR, IRSerializationError
from ..errors import PlanningError
from ..version import __version__

if TYPE_CHECKING:
    from ..runtime.options import ExecutionOptions
    from ..runtime.options_resolver import ResolvedExecutionOptions
    from .models import ExecutionPlan

EXECUTION_PLAN_SCHEMA = "flagquantum.execution_plan"
EXECUTION_PLAN_VERSION = "1.0"
NOISE_EXTENSION_NAMESPACE = "flagquantum.noise"
NOISE_EXTENSION_KIND = "noise_model"
NOISE_EXTENSION_VERSION = "1.0"
_COMPILER_PIPELINE = "flagquantum.compiler.stable.v1"
_FINGERPRINT_NAMES = ("program", "options", "environment", "compiler")
_DECISION_FIELDS = {
    "mode",
    "backend",
    "device",
    "target",
    "batch_size",
    "precision",
    "world_size",
    "state_bytes",
    "recommended_mode",
    "require_gradients",
    "allow_approximate",
    "allow_backend_fallback",
    "memory_limit_bytes",
}
_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "identity",
    "program",
    "requested_options",
    "resolved_options",
    "fingerprints",
    "environment_requirements",
    "decision",
    "extensions",
}


class ExecutionPlanContractError(PlanningError):
    """Internal typed failure carrying the Proposal 003 reason code."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


def canonical_json(payload: object, *, indent: int | None = None) -> str:
    """Serialize a contract payload deterministically."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def attach_execution_contract(
    plan: ExecutionPlan,
    *,
    program: CircuitIR,
    requested_options: ExecutionOptions,
    resolved_options: ResolvedExecutionOptions,
    noise_model: Any | None = None,
    preserve_program_instructions: bool = False,
) -> ExecutionPlan:
    """Attach one immutable canonical payload to an internally built plan."""

    executable_program = (
        program if preserve_program_instructions else _canonical_program(program, plan)
    )
    resolved = _resolved_mapping(resolved_options)
    backend = (
        "pytorch" if resolved_options.backend == "auto" else resolved_options.backend
    )
    device = _resolve_planned_device(resolved_options.device, backend=backend)
    decision = {
        "mode": plan.state_mode,
        "backend": backend,
        "device": device,
        "target": resolved_options.target,
        "batch_size": resolved_options.batch_size,
        "precision": resolved_options.precision,
        "world_size": plan.world_size,
        "state_bytes": plan.state_bytes,
        "recommended_mode": plan.recommended_mode,
        "require_gradients": resolved_options.require_gradients,
        "allow_approximate": resolved_options.allow_approximate,
        "allow_backend_fallback": resolved_options.allow_backend_fallback,
        "memory_limit_bytes": resolved_options.memory_limit_bytes,
    }
    environment = _environment_requirements(decision)
    compiler = _compiler_identity()
    fingerprints = {
        "program": executable_program.content_hash,
        "options": canonical_hash(resolved),
        "environment": canonical_hash(environment),
        "compiler": canonical_hash(compiler),
    }
    extensions = [] if noise_model is None else [_noise_extension(noise_model)]
    payload: dict[str, object] = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "version": EXECUTION_PLAN_VERSION,
        "identity": "",
        "program": executable_program.to_dict(),
        "requested_options": requested_options.to_dict(),
        "resolved_options": resolved,
        "fingerprints": fingerprints,
        "environment_requirements": environment,
        "decision": decision,
        "extensions": extensions,
    }
    payload["identity"] = _identity(payload, compiler=compiler)
    return replace(plan, _contract_payload_json=canonical_json(payload))


def plan_to_dict(plan: ExecutionPlan) -> dict[str, object]:
    payload = _payload(plan)
    validate_plan_payload(payload)
    return json.loads(canonical_json(payload))


def plan_to_json(plan: ExecutionPlan, *, indent: int | None = None) -> str:
    return canonical_json(plan_to_dict(plan), indent=indent)


def plan_from_dict(payload: Mapping[str, Any]) -> ExecutionPlan:
    """Restore a plan without replanning and verify every identity component."""

    normalized = validate_plan_payload(payload)
    program = CircuitIR.from_dict(normalized["program"])
    decision = normalized["decision"]

    from .execution_plan_builder import build_layer_plans
    from .models import ExecutionPlan
    from .planner import analyze

    planned_program = program
    if normalized["extensions"]:
        from ..noise import NoiseModel
        from .noise import lower_noise_model

        noise_model = NoiseModel.from_dict(normalized["extensions"][0]["payload"])
        planned_program = lower_noise_model(program, noise_model)
    analysis = analyze(planned_program)
    plan = ExecutionPlan(
        analysis=analysis,
        layers=build_layer_plans(planned_program),
        state_bytes=int(decision["state_bytes"]),
        recommended_mode=str(decision["recommended_mode"]),
        world_size=int(decision["world_size"]),
        shardable_wires=tuple(range(max(0, program.n_wires - 1))),
        state_mode=str(decision["mode"]),
        user_tier=(
            "production_distributed"
            if int(decision["world_size"]) > 1
            else "single_device"
        ),
        usability_contract=(
            "single_api_distributed_scale_out"
            if int(decision["world_size"]) > 1
            else "single_api_fast_path"
        ),
        runtime_config=_runtime_config_manifest(decision),
        _contract_payload_json=canonical_json(normalized),
    )
    if normalized["extensions"]:
        from .noise import build_noisy_execution_plan

        extension = normalized["extensions"][0]
        plan = replace(
            plan,
            noisy_execution_plan=build_noisy_execution_plan(
                plan,
                representation="density_matrix",
                evolution="exact_channel",
                memory_limit_bytes=decision["memory_limit_bytes"],
                noise_model_identity=str(extension["identity"]),
            ),
        )
    return plan


def plan_from_json(text: str) -> ExecutionPlan:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExecutionPlanContractError(
            "unsupported_schema", f"invalid ExecutionPlan JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExecutionPlanContractError(
            "unsupported_schema", "serialized ExecutionPlan must be a JSON object"
        )
    return plan_from_dict(payload)


def validate_plan_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("execution plan payload must be a mapping")
    values = dict(payload)
    unknown = set(values) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(values)
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        raise ExecutionPlanContractError("unsupported_schema", "; ".join(details))
    if values["schema"] != EXECUTION_PLAN_SCHEMA:
        raise ExecutionPlanContractError(
            "unsupported_schema", "invalid ExecutionPlan schema"
        )
    if values["version"] != EXECUTION_PLAN_VERSION:
        raise ExecutionPlanContractError(
            "unsupported_schema",
            f"unsupported ExecutionPlan version {values['version']!r}",
        )

    program_payload = _require_mapping("program", values["program"])
    try:
        program = CircuitIR.from_dict(program_payload)
    except (IRSerializationError, KeyError, TypeError, ValueError) as exc:
        raise ExecutionPlanContractError(
            "program_fingerprint_mismatch", str(exc)
        ) from exc

    from ..runtime.options import ExecutionOptions

    requested_payload = _require_mapping(
        "requested_options", values["requested_options"]
    )
    try:
        ExecutionOptions.from_dict(requested_payload)
    except (TypeError, ValueError) as exc:
        raise ExecutionPlanContractError(
            "options_fingerprint_mismatch", str(exc)
        ) from exc
    resolved = _require_mapping("resolved_options", values["resolved_options"])
    expected_option_fields = {field.name for field in fields(ExecutionOptions)}
    if set(resolved) != expected_option_fields:
        raise ExecutionPlanContractError(
            "options_fingerprint_mismatch",
            "resolved options fields do not match ExecutionOptions v1",
        )
    try:
        ExecutionOptions(**resolved)
    except (TypeError, ValueError) as exc:
        raise ExecutionPlanContractError(
            "options_fingerprint_mismatch", str(exc)
        ) from exc

    fingerprints = _require_mapping("fingerprints", values["fingerprints"])
    if set(fingerprints) != set(_FINGERPRINT_NAMES):
        raise ExecutionPlanContractError(
            "identity_mismatch", "fingerprint fields do not match schema v1"
        )
    environment = _require_mapping(
        "environment_requirements", values["environment_requirements"]
    )
    decision = _require_mapping("decision", values["decision"])
    if set(decision) != _DECISION_FIELDS:
        raise ExecutionPlanContractError(
            "identity_mismatch", "decision fields do not match schema v1"
        )
    _validate_decision(decision)
    extensions = values["extensions"]
    if not isinstance(extensions, Sequence) or isinstance(extensions, (str, bytes)):
        raise ExecutionPlanContractError(
            "extension_incompatible", "extensions must be a JSON array"
        )
    _validate_extensions(extensions)

    expected_fingerprints = {
        "program": program.content_hash,
        "options": canonical_hash(resolved),
        "environment": canonical_hash(environment),
        "compiler": canonical_hash(_compiler_identity()),
    }
    for name in _FINGERPRINT_NAMES:
        if fingerprints[name] != expected_fingerprints[name]:
            reason = {
                "program": "program_fingerprint_mismatch",
                "options": "options_fingerprint_mismatch",
                "environment": "environment_incompatible",
                "compiler": "compiler_incompatible",
            }[name]
            raise ExecutionPlanContractError(reason, f"{name} fingerprint mismatch")
    if environment != _environment_requirements(decision):
        raise ExecutionPlanContractError(
            "environment_incompatible",
            "environment requirements disagree with the execution decision",
        )
    expected_identity = _identity(values, compiler=_compiler_identity())
    if values["identity"] != expected_identity:
        raise ExecutionPlanContractError(
            "identity_mismatch", "ExecutionPlan identity does not match its payload"
        )
    return json.loads(canonical_json(values))


def validate_plan_environment(plan: ExecutionPlan) -> None:
    """Fail before kernel launch when the current environment cannot run a plan."""

    payload = validate_plan_payload(_payload(plan))
    decision = payload["decision"]
    backend = str(decision["backend"])

    from ..runtime.backend_registry import get_backend_capabilities

    try:
        capabilities = get_backend_capabilities(backend, refresh=True)
    except KeyError as exc:
        raise ExecutionPlanContractError(
            "backend_unavailable", f"backend {backend!r} is not registered"
        ) from exc
    device_type = str(decision["device"]).split(":", 1)[0]
    if device_type not in capabilities.devices and device_type != "flagos":
        raise ExecutionPlanContractError(
            "environment_incompatible",
            f"backend {backend!r} does not expose device {device_type!r}",
        )
    if str(decision["precision"]) not in capabilities.dtypes:
        raise ExecutionPlanContractError(
            "environment_incompatible",
            f"backend {backend!r} does not support precision {decision['precision']!r}",
        )
    from ..runtime.distributed.backend_policy import (
        resolve_distributed_backend_policy,
    )

    actual_world_size = resolve_distributed_backend_policy().effective_world_size
    if actual_world_size != int(decision["world_size"]):
        raise ExecutionPlanContractError(
            "world_size_mismatch",
            f"plan requires world_size={decision['world_size']}, "
            f"current environment provides {actual_world_size}",
        )


def plan_program(plan: ExecutionPlan) -> CircuitIR:
    return CircuitIR.from_dict(plan_to_dict(plan)["program"])


def plan_decision(plan: ExecutionPlan) -> dict[str, object]:
    return dict(plan_to_dict(plan)["decision"])


def plan_noise_model(plan: ExecutionPlan) -> Any | None:
    """Restore the optional stable noise extension carried by a plan."""

    extensions = plan_to_dict(plan)["extensions"]
    if not extensions:
        return None
    from ..noise import NoiseModel

    return NoiseModel.from_dict(extensions[0]["payload"])


def _payload(plan: ExecutionPlan) -> dict[str, Any]:
    text = getattr(plan, "_contract_payload_json", None)
    if text is None:
        raise ExecutionPlanContractError(
            "unsupported_schema",
            "this internal analysis plan is not an executable ExecutionPlan",
        )
    payload = json.loads(text)
    if not isinstance(payload, dict):  # pragma: no cover - internal invariant
        raise ExecutionPlanContractError(
            "unsupported_schema", "invalid internal ExecutionPlan payload"
        )
    return payload


def _canonical_program(program: CircuitIR, plan: ExecutionPlan) -> CircuitIR:
    metadata = dict(program.metadata)
    metadata.pop("runtime_config", None)
    instructions = tuple(
        instruction
        for layer in sorted(plan.layers, key=lambda item: item.index)
        for instruction in layer.instructions
    )
    return CircuitIR(
        n_wires=program.n_wires,
        instructions=instructions,
        version=program.version,
        dtype=program.dtype,
        shape=program.shape,
        observables=program.observables,
        measurements=program.measurements,
        metadata=metadata,
    )


def _noise_extension(noise_model: Any) -> dict[str, object]:
    from ..noise import NoiseModel

    if not isinstance(noise_model, NoiseModel):
        raise TypeError("noise_model must be a flagquantum.noise.NoiseModel or None")
    return {
        "namespace": NOISE_EXTENSION_NAMESPACE,
        "kind": NOISE_EXTENSION_KIND,
        "version": NOISE_EXTENSION_VERSION,
        "identity": noise_model.identity,
        "payload": noise_model.to_dict(),
    }


def _validate_extensions(extensions: Sequence[object]) -> None:
    if len(extensions) > 1:
        raise ExecutionPlanContractError(
            "extension_incompatible", "ExecutionPlan v1 accepts at most one extension"
        )
    if not extensions:
        return
    extension = _require_mapping("noise extension", extensions[0])
    expected = {"namespace", "kind", "version", "identity", "payload"}
    if set(extension) != expected:
        raise ExecutionPlanContractError(
            "extension_incompatible", "noise extension fields do not match schema v1"
        )
    if (
        extension["namespace"] != NOISE_EXTENSION_NAMESPACE
        or extension["kind"] != NOISE_EXTENSION_KIND
        or extension["version"] != NOISE_EXTENSION_VERSION
    ):
        raise ExecutionPlanContractError(
            "extension_incompatible", "unsupported executable plan extension"
        )
    from ..noise import NoiseModel

    try:
        noise_model = NoiseModel.from_dict(
            _require_mapping("noise extension payload", extension["payload"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionPlanContractError("extension_incompatible", str(exc)) from exc
    if extension["identity"] != noise_model.identity:
        raise ExecutionPlanContractError(
            "extension_incompatible", "noise model identity mismatch"
        )


def _resolved_mapping(resolved: ResolvedExecutionOptions) -> dict[str, object]:
    return {
        field.name: getattr(resolved, field.name)
        for field in fields(resolved)
        if field.name != "sources"
    }


def _resolve_planned_device(device: str, *, backend: str) -> str:
    from ..runtime.backend_registry import resolve_device

    try:
        return str(resolve_device(device, backend=backend))
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ExecutionPlanContractError(
            "backend_unavailable",
            f"cannot resolve backend={backend!r}, device={device!r}: {exc}",
        ) from exc


def _environment_requirements(decision: Mapping[str, object]) -> dict[str, object]:
    world_size = int(decision["world_size"])
    return {
        "protocol": "flagquantum.execution_environment",
        "version": "1.0",
        "backend": decision["backend"],
        "device_kind": str(decision["device"]).split(":", 1)[0],
        "precision": decision["precision"],
        "world_size": world_size,
        "distribution_semantics": (
            "sharded_across_ranks" if world_size > 1 else "single_device_fast_path"
        ),
        "require_gradients": decision["require_gradients"],
        "allow_approximate": decision["allow_approximate"],
    }


def _compiler_identity() -> dict[str, str]:
    return {
        "pipeline": _COMPILER_PIPELINE,
        "flagquantum_version": __version__,
        "ir_version": IR_VERSION,
    }


def _identity(payload: Mapping[str, object], *, compiler: Mapping[str, str]) -> str:
    fingerprints = _require_mapping("fingerprints", payload["fingerprints"])
    identity_payload = {
        "schema": payload["schema"],
        "version": payload["version"],
        "program_fingerprint": fingerprints["program"],
        "resolved_execution_semantics": payload["resolved_options"],
        "compiler": compiler,
        "environment_requirements": payload["environment_requirements"],
        "decision": payload["decision"],
        "extension_identities": payload["extensions"],
    }
    return canonical_hash(identity_payload)


def _require_mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionPlanContractError(
            "unsupported_schema", f"{name} must be a JSON object"
        )
    return dict(value)


def _validate_decision(decision: Mapping[str, object]) -> None:
    if decision["mode"] not in {
        "statevector",
        "mps",
        "tensor_network",
        "density_matrix",
    }:
        raise ExecutionPlanContractError(
            "identity_mismatch", f"invalid planned mode {decision['mode']!r}"
        )
    for name in ("backend", "device", "target", "precision", "recommended_mode"):
        if not isinstance(decision[name], str) or not decision[name]:
            raise ExecutionPlanContractError(
                "identity_mismatch", f"decision {name} must be a non-empty string"
            )
    for name in ("batch_size", "world_size"):
        if type(decision[name]) is not int or int(decision[name]) < 1:
            raise ExecutionPlanContractError(
                "identity_mismatch", f"decision {name} must be a positive integer"
            )
    if type(decision["state_bytes"]) is not int or int(decision["state_bytes"]) < 0:
        raise ExecutionPlanContractError(
            "identity_mismatch", "decision state_bytes must be a non-negative integer"
        )
    for name in (
        "require_gradients",
        "allow_approximate",
        "allow_backend_fallback",
    ):
        if type(decision[name]) is not bool:
            raise ExecutionPlanContractError(
                "identity_mismatch", f"decision {name} must be a bool"
            )
    memory = decision["memory_limit_bytes"]
    if memory is not None and (type(memory) is not int or memory < 1):
        raise ExecutionPlanContractError(
            "identity_mismatch",
            "decision memory_limit_bytes must be null or a positive integer",
        )


def _runtime_config_manifest(decision: Mapping[str, object]) -> dict[str, object]:
    from ..core.runtime_config import RuntimeConfig

    return RuntimeConfig(
        backend=str(decision["backend"]),
        device=str(decision["device"]),
        complex_dtype=str(decision["precision"]),
        real_dtype=("float64" if decision["precision"] == "complex128" else "float32"),
        jax_enable_x64=decision["precision"] == "complex128",
    ).to_manifest()


__all__ = (
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_PLAN_VERSION",
    "ExecutionPlanContractError",
    "attach_execution_contract",
    "canonical_hash",
    "canonical_json",
    "plan_decision",
    "plan_from_dict",
    "plan_from_json",
    "plan_program",
    "plan_to_dict",
    "plan_to_json",
    "validate_plan_environment",
    "validate_plan_payload",
)
