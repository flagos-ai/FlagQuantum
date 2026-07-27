from copy import deepcopy

import pytest

import flagquantum as fq
from flagquantum.deployment.routing_evidence import (
    DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA,
    DeploymentRoutingEvidenceError,
    deployment_artifact_sha256,
    stable_payload_sha256,
    validate_deployment_routing_plan,
)

pytestmark = pytest.mark.unit


def _routing_plan() -> tuple[dict, fq.CouplingMap]:
    circuit = fq.Circuit(5)
    circuit.cx(0, 4).h(4).cx(0, 4)
    coupling = fq.CouplingMap.line(5)
    package = fq.create_deployment_package(
        circuit,
        backend=fq.CloudBackendProfile(
            provider="local",
            name="line5",
            n_wires=5,
            coupling_map=coupling,
            is_simulator=True,
        ),
        routing_strategy="auto",
    )
    assert package.metadata["routing_evidence"]["schema"] == (
        DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA
    )
    return dict(package.metadata["routing_plan"]), coupling


def test_valid_deployment_routing_plan_round_trips() -> None:
    plan, coupling = _routing_plan()

    assert (
        validate_deployment_routing_plan(
            plan,
            n_wires=5,
            coupling_map=coupling,
        )
        == plan
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda plan: plan.update(final_logical_to_physical=(1, 0, 2, 3, 4)),
            "restore the final",
        ),
        (
            lambda plan: plan.update(coupling_edges=((0, 4),)),
            "do not match",
        ),
        (
            lambda plan: plan.update(planned_inserted_swap_count=-1),
            "non-negative",
        ),
        (
            lambda plan: plan["strategy_selection"].update(
                selected_strategy="restore_after_each_gate"
            ),
            "disagrees",
        ),
    ),
)
def test_deployment_routing_plan_rejects_tampering(mutation, message: str) -> None:
    plan, coupling = _routing_plan()
    tampered = deepcopy(plan)
    mutation(tampered)

    with pytest.raises(DeploymentRoutingEvidenceError, match=message):
        validate_deployment_routing_plan(
            tampered,
            n_wires=5,
            coupling_map=coupling,
        )


def test_routing_and_artifact_hashes_are_deterministic_and_tamper_sensitive() -> None:
    plan, _ = _routing_plan()
    evidence = {
        "schema": DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA,
        "status": "validated",
        "routing_reused": False,
        "routing_plan": plan,
    }
    first = stable_payload_sha256(evidence)
    second = stable_payload_sha256(deepcopy(evidence))
    tampered = deepcopy(evidence)
    tampered["routing_plan"]["planned_inserted_swap_count"] += 1

    assert first == second
    assert len(first) == 64
    assert stable_payload_sha256(tampered) != first
    artifact = deployment_artifact_sha256(
        name="job",
        backend_provider="local",
        backend_name="line5",
        shots=100,
        program_format="openqasm-2",
        program="OPENQASM 2.0;",
        routing_evidence_sha256=first,
    )
    changed = deployment_artifact_sha256(
        name="job",
        backend_provider="local",
        backend_name="line5",
        shots=101,
        program_format="openqasm-2",
        program="OPENQASM 2.0;",
        routing_evidence_sha256=first,
    )
    assert len(artifact) == 64
    assert artifact != changed
