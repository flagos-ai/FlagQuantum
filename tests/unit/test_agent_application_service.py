from __future__ import annotations

from flagquantum.agent_services import AgentApplicationService
from flagquantum.core._artifacts import ArtifactKind, ProgramArtifact
from flagquantum.core.ir import CircuitIR, Instruction


def _program() -> dict[str, object]:
    return CircuitIR(
        n_wires=1,
        instructions=(Instruction("h", (0,)),),
    ).to_dict()


def test_agent_service_validates_serialized_ir_without_protocol_types() -> None:
    report = AgentApplicationService().validate_program(_program())
    assert report["valid"] is True
    assert report["metadata"] == {"n_wires": 1, "n_instructions": 1}


def test_agent_service_capabilities_remain_machine_readable() -> None:
    report = AgentApplicationService().capabilities()
    assert report["schema"] == "flagquantum_agent_capabilities_v1"
    assert isinstance(report["backends"], dict)


def test_agent_service_plans_without_executing_or_using_protocol_types() -> None:
    report = AgentApplicationService().plan_execution(
        _program(), options={"require_gradients": False, "target": "full_state"}
    )
    assert report["executable"] is True
    assert report["selected_backend"]
    assert report["validation"]["valid"] is True


def test_artifact_to_agent_service_to_planner_compatibility_chain() -> None:
    artifact = ProgramArtifact(
        kind=ArtifactKind.CIRCUIT,
        payload=_program(),
        producer="test-sdk",
        required_capabilities=("statevector",),
    )
    report = AgentApplicationService().plan_execution(
        artifact.to_dict(),
        options={"require_gradients": False, "target": "full_state"},
    )
    assert report["executable"] is True
    assert report["plan"]["n_wires"] == 1
