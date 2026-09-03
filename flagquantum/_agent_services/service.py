"""Deterministic application service used by MCP and future northbound APIs."""

from __future__ import annotations

from typing import Any, Mapping

from .. import agent
from ..core._artifacts import ArtifactKind, ProgramArtifact
from ..core.ir import CircuitIR


def _decode_program(program: Mapping[str, Any]) -> CircuitIR:
    schema = program.get("schema")
    if schema == "flagquantum.program_artifact":
        artifact = ProgramArtifact.from_dict(program)
        if artifact.kind is not ArtifactKind.CIRCUIT:
            raise ValueError(
                "the current agent service accepts only circuit program artifacts"
            )
        return CircuitIR.from_dict(artifact.payload)
    return CircuitIR.from_dict(program)


class AgentApplicationService:
    """Expose safe discovery, validation, and planning without protocol types."""

    def capabilities(self, *, refresh: bool = False) -> dict[str, Any]:
        return agent.capabilities(refresh=refresh)

    def validate_program(self, program: Mapping[str, Any]) -> dict[str, Any]:
        circuit_ir = _decode_program(program)
        return agent.validate(circuit_ir).to_dict()

    def plan_execution(
        self,
        program: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        circuit_ir = _decode_program(program)
        report = agent.preflight_execution(circuit_ir, **dict(options or {}))
        return report.to_dict()


__all__ = ("AgentApplicationService",)
