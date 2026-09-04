"""Deterministic application service for local and remote northbound APIs."""

from __future__ import annotations

from typing import Any, Mapping

from ..agent import (
    AgentExecutionPlan,
    ValidationIssue,
    preflight_execution,
    validate,
)
from ..agent import (
    capabilities as _agent_capabilities,
)
from ..core._artifacts import ArtifactKind, ProgramArtifact
from ..core.ir import CircuitIR


def _decode_program(program: Mapping[str, Any]) -> tuple[CircuitIR, tuple[str, ...]]:
    schema = program.get("schema")
    if schema == "flagquantum.program_artifact":
        artifact = ProgramArtifact.from_dict(program)
        if artifact.kind is not ArtifactKind.CIRCUIT:
            raise ValueError(
                "the current agent service accepts only circuit program artifacts"
            )
        return CircuitIR.from_dict(artifact.payload), artifact.required_capabilities
    if schema is not None:
        raise ValueError(f"unsupported program artifact schema {schema!r}")
    return CircuitIR.from_dict(program), ()


def _available_capabilities(manifest: Mapping[str, Any]) -> frozenset[str]:
    available = {
        str(name)
        for name, supported in manifest.get("contracts", {}).items()
        if supported is True
    }
    for backend_name, raw_summary in manifest.get("backends", {}).items():
        if (
            not isinstance(raw_summary, Mapping)
            or raw_summary.get("available") is False
        ):
            continue
        available.add(str(backend_name))
        if raw_summary.get("name"):
            available.add(str(raw_summary["name"]))
        available.update(
            str(name).removeprefix("supports_")
            for name, supported in raw_summary.items()
            if str(name).startswith("supports_") and supported is True
        )
        for field in ("accelerators", "devices", "dtypes"):
            values = raw_summary.get(field, ())
            if isinstance(values, (list, tuple)):
                available.update(str(value) for value in values)
    return frozenset(available)


class AgentApplicationService:
    """Expose safe discovery, validation, and planning without protocol types."""

    def capabilities(self, *, refresh: bool = False) -> dict[str, Any]:
        return _agent_capabilities(refresh=refresh)

    def validate_program(self, program: Mapping[str, Any]) -> dict[str, Any]:
        circuit_ir, _ = _decode_program(program)
        return validate(circuit_ir).to_dict()

    def plan_execution(
        self,
        program: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        circuit_ir, required = _decode_program(program)
        if required:
            available = _available_capabilities(self.capabilities())
            missing = tuple(sorted(set(required) - available))
            if missing:
                validation = validate(circuit_ir)
                blocker = ValidationIssue(
                    code="REQUIRED_CAPABILITY_UNAVAILABLE",
                    message=(
                        "program requires unavailable capabilities: "
                        + ", ".join(missing)
                    ),
                    context={
                        "required_capabilities": list(required),
                        "available_capabilities": sorted(available),
                        "missing_capabilities": list(missing),
                    },
                    suggestions=({"action": "select_compatible_runtime_or_provider"},),
                )
                return AgentExecutionPlan(
                    executable=False,
                    selected_backend=None,
                    validation=validation,
                    blockers=(blocker,),
                ).to_dict()
        report = preflight_execution(circuit_ir, **dict(options or {}))
        return report.to_dict()


__all__ = ("AgentApplicationService",)
