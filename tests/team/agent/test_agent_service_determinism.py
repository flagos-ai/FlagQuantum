from __future__ import annotations

from copy import deepcopy

import pytest

import flagquantum._agent_services.service as service_module
from flagquantum._agent_services import AgentApplicationService
from flagquantum.core._artifacts import ArtifactKind, ProgramArtifact
from flagquantum.core.ir import CircuitIR, Instruction


def _program() -> dict[str, object]:
    return CircuitIR(
        n_wires=1,
        instructions=(Instruction("h", (0,)),),
    ).to_dict()


def _artifact(*, required_capabilities: tuple[str, ...] = ()) -> dict[str, object]:
    return ProgramArtifact(
        kind=ArtifactKind.CIRCUIT,
        payload=_program(),
        producer="agent-boundary-test",
        required_capabilities=required_capabilities,
    ).to_dict()


def test_program_artifact_planning_is_repeatable_and_does_not_mutate_input() -> None:
    service = AgentApplicationService()
    artifact = _artifact(required_capabilities=("statevector",))
    original = deepcopy(artifact)
    options = {"require_gradients": False, "target": "full_state"}

    first = service.plan_execution(artifact, options=options)
    second = service.plan_execution(artifact, options=options)

    assert first == second
    assert first["executable"] is True
    assert first["plan"]["n_wires"] == 1
    assert artifact == original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"kind": "future_artifact"}, "future_artifact"),
        (
            {"schema": "flagquantum.program_artifact.v2"},
            "unsupported program artifact schema",
        ),
    ],
)
def test_unknown_artifacts_are_rejected_repeatably(
    mutation: dict[str, str], message: str
) -> None:
    payload = _artifact()
    payload.update(mutation)
    service = AgentApplicationService()

    observed = []
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            service.validate_program(payload)
        observed.append(str(caught.value))

    assert observed[0] == observed[1]
    assert message in observed[0]


def test_known_non_circuit_artifact_is_rejected_before_planning() -> None:
    payload = ProgramArtifact(
        kind=ArtifactKind.EXECUTABLE,
        payload={"opaque": "executor-owned"},
        producer="agent-boundary-test",
    ).to_dict()

    with pytest.raises(
        ValueError,
        match="current agent service accepts only circuit program artifacts",
    ):
        AgentApplicationService().plan_execution(payload)


def test_missing_artifact_capability_fails_closed_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema": "flagquantum_agent_capabilities_v1",
        "version": "test",
        "contracts": {"ir_validation": True, "execution_planning": True},
        "backends": {
            "cpu": {
                "name": "cpu",
                "available": True,
                "devices": ["cpu"],
                "supports_statevector": True,
            }
        },
    }
    monkeypatch.setattr(service_module, "_agent_capabilities", lambda **_: manifest)

    def unexpected_planning(*args: object, **kwargs: object) -> None:
        raise AssertionError("planner must not run after capability rejection")

    monkeypatch.setattr(service_module, "preflight_execution", unexpected_planning)
    artifact = _artifact(required_capabilities=("qpu", "statevector"))

    first = AgentApplicationService().plan_execution(artifact)
    second = AgentApplicationService().plan_execution(artifact)

    assert first == second
    assert first["executable"] is False
    assert first["selected_backend"] is None
    assert first["blockers"][0]["code"] == "REQUIRED_CAPABILITY_UNAVAILABLE"
    assert first["blockers"][0]["context"]["missing_capabilities"] == ["qpu"]


def test_planning_failure_has_a_repeatable_structured_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compilation import planner

    def fail_planning(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deterministic planner fixture failure")

    monkeypatch.setattr(planner, "plan_runtime_selection", fail_planning)
    service = AgentApplicationService()

    first = service.plan_execution(_program())
    second = service.plan_execution(_program())

    assert first == second
    assert first["executable"] is False
    assert first["validation"]["valid"] is True
    assert first["blockers"][0]["code"] == "EXECUTION_PLANNING_FAILED"
    assert first["blockers"][0]["retryable"] is True
    assert first["blockers"][0]["message"] == "deterministic planner fixture failure"
