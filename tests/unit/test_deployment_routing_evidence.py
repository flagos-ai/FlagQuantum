from copy import deepcopy

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.compiler import CouplingMap
from flagquantum.compiler import compile as compile_circuit
from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.deployment import CloudBackendProfile
from flagquantum.deployment.routing_evidence import (
    DEPLOYMENT_ROUTING_EVIDENCE_SCHEMA,
    DeploymentRoutingEvidenceError,
    deployment_artifact_sha256,
    stable_payload_sha256,
    validate_deployment_routing_plan,
)

pytestmark = pytest.mark.unit


def _routing_plan() -> tuple[dict, CouplingMap]:
    circuit = fq.Circuit(5)
    circuit.cx(0, 4).h(4).cx(0, 4)
    coupling = CouplingMap.line(5)
    package = fqd.create_deployment_package(
        circuit,
        backend=CloudBackendProfile(
            provider="local",
            name="line5",
            n_qubits=5,
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


def _ordered_device_plan() -> tuple[dict, DirectedCouplingMap]:
    """A routing plan the router built for a device that declares its direction.

    The device's only link runs from wire 1 to wire 0 and the source asks for
    ``cx(0, 1)``, so the plan is a plan that had to repair an ordering: it says so
    in ``direction_semantics`` and counts the repair in ``direction_swap_count``.
    """

    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.37)
    circuit.cx(0, 1)
    circuit.ry(1, theta=0.59)
    device = DirectedCouplingMap(2, ((1, 0),))
    compiled = compile_circuit(circuit, coupling_map=device, routing_strategy="auto")
    plan = dict(compiled.metadata["routing"])
    assert plan["direction_semantics"] == "directed_cx"
    assert plan["direction_swap_count"] == 1
    assert plan["coupling_edges"] == ((1, 0),)
    return plan, device


def _undirected_device_plan() -> tuple[dict, CouplingMap]:
    """The plan the same workload gets from a device that declares no direction.

    The links are the same wire pair, so the two plans differ only in what they
    say about the direction of that pair: this one routed an operation whose
    operands run across the link in one order or the other without a rule to obey.
    """

    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.37)
    circuit.cx(0, 1)
    circuit.ry(1, theta=0.59)
    device = CouplingMap(2, ((1, 0),))
    compiled = compile_circuit(
        circuit, coupling_map=device, routing_strategy="restore_after_each_gate"
    )
    plan = dict(compiled.metadata["routing"])
    assert plan["direction_semantics"] == "logical_wire_order_preserved"
    assert plan["direction_swap_count"] == 0
    assert plan["coupling_edges"] == ((0, 1),)
    return plan, device


def test_an_ordered_device_produces_a_deployable_routing_plan() -> None:
    """A plan the router built for a directed device is accepted by name.

    This validator used to name exactly one acceptable direction semantics,
    ``logical_wire_order_preserved``, which was the only value the router could
    emit. The router now reads the device it is handed, so a deployment backend
    holding a ``DirectedCouplingMap`` was refused with a message that blamed the
    plan for missing direction semantics when the plan stated them. The accepted
    pair is now per device type: each type names the single value it can legally
    produce, and the two are mutually exclusive rather than both allowed.
    """

    plan, device = _ordered_device_plan()

    assert (
        validate_deployment_routing_plan(plan, n_wires=2, coupling_map=device) == plan
    )
    # Without a device to check against, the plan's own statement is still a
    # statement about one of the two known semantics.
    assert validate_deployment_routing_plan(plan, n_wires=2, coupling_map=None) == plan


def test_a_plan_and_a_device_that_disagree_about_direction_are_refused() -> None:
    """Neither direction claim is accepted for the device that cannot make it.

    The message names the disagreement rather than the plan, because the plan is
    a faithful record of what it routed: the mismatch is between the plan and the
    backend it is being installed into.
    """

    plan, device = _ordered_device_plan()
    undirected_plan, undirected = _undirected_device_plan()

    with pytest.raises(
        DeploymentRoutingEvidenceError, match="disagree with the deployment backend"
    ):
        validate_deployment_routing_plan(plan, n_wires=2, coupling_map=undirected)
    with pytest.raises(
        DeploymentRoutingEvidenceError, match="disagree with the deployment backend"
    ):
        validate_deployment_routing_plan(
            undirected_plan, n_wires=2, coupling_map=device
        )
    # Each plan is accepted by the device whose rule it obeyed, so the refusals
    # above are about the disagreement and not about either plan being malformed.
    assert validate_deployment_routing_plan(
        undirected_plan, n_wires=2, coupling_map=undirected
    )
    assert validate_deployment_routing_plan(plan, n_wires=2, coupling_map=device)


def test_a_direction_repair_that_cannot_fit_the_total_is_refused() -> None:
    """The ordering count is a part of the SWAP total, so it is bounded by it.

    A direction repair is two SWAPs, one on each side of the gate it orients, so
    a plan cannot carry more of them than it carries SWAPs. Stating the count
    without that bound would let a plan report repairs it never emitted while the
    total stayed plausible.
    """

    plan, device = _ordered_device_plan()
    assert plan["inserted_swap_count"] == 2 * plan["direction_swap_count"]

    tampered = deepcopy(plan)
    tampered["direction_swap_count"] = plan["inserted_swap_count"]
    with pytest.raises(
        DeploymentRoutingEvidenceError, match="direction swap count exceeds"
    ):
        validate_deployment_routing_plan(tampered, n_wires=2, coupling_map=device)

    # A missing count is refused rather than read as zero: silence about the
    # ordering is what the schema exists to forbid.
    del tampered["direction_swap_count"]
    with pytest.raises(DeploymentRoutingEvidenceError, match="direction_swap_count"):
        validate_deployment_routing_plan(tampered, n_wires=2, coupling_map=device)

    tampered = deepcopy(plan)
    tampered["direction_swap_count"] = -1
    with pytest.raises(DeploymentRoutingEvidenceError, match="non-negative integer"):
        validate_deployment_routing_plan(tampered, n_wires=2, coupling_map=device)


def test_an_undirected_device_cannot_claim_a_direction_repair() -> None:
    """A link with no direction has no ordering to repair.

    The semantics clause already refuses the claim for a ``CouplingMap``, and this
    pins the second half: even a plan whose semantics were left correct cannot
    count repairs on a device that could not have needed one.
    """

    undirected = CouplingMap.line(3)
    circuit = fq.Circuit(3)
    circuit.cx(0, 2)
    plan = dict(
        compile_circuit(
            circuit, coupling_map=undirected, routing_strategy="auto"
        ).metadata["routing"]
    )
    plan["direction_swap_count"] = 1
    plan["inserted_swap_count"] = plan["planned_inserted_swap_count"] = 4

    with pytest.raises(
        DeploymentRoutingEvidenceError,
        match="undirected deployment backend cannot have required direction swaps",
    ):
        validate_deployment_routing_plan(plan, n_wires=3, coupling_map=undirected)
    # With no device to compare against, the same plan is accepted: the semantics
    # are one of the two known ones and the count fits the total. The bound a
    # device adds is the part that needs the device.
    assert validate_deployment_routing_plan(plan, n_wires=3, coupling_map=None)


def test_an_ordered_backend_carries_the_ordering_through_a_whole_package() -> None:
    """A deployment package for an ordered device installs and validates.

    The backend profile accepts a ``DirectedCouplingMap`` because its field is
    typed, not checked, so the path that matters is the whole one: build the
    package, re-validate its emitted evidence, and read the ordered edges back out
    of the plan the provider will receive.
    """

    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.37)
    circuit.cx(0, 1)
    device = DirectedCouplingMap(2, ((1, 0),))
    package = fqd.create_deployment_package(
        circuit,
        backend=CloudBackendProfile(
            provider="local",
            name="ordered-line2",
            n_qubits=2,
            coupling_map=device,
            is_simulator=True,
        ),
        routing_strategy="auto",
    )
    evidence = package.metadata["routing_evidence"]
    plan = evidence["routing_plan"]

    assert evidence["status"] == "validated"
    assert plan["direction_semantics"] == "directed_cx"
    assert plan["direction_swap_count"] == 1
    assert plan["coupling_edges"] == ((1, 0),)
    assert (
        validate_deployment_routing_plan(plan, n_wires=2, coupling_map=device) == plan
    )
    # The program the provider receives is the routed one, so the orientation
    # SWAPs the plan counts are the instructions it will run, and the sealed
    # package identity is verified against the same plan.
    assert package.qasm.count("swap") == plan["inserted_swap_count"]
    assert package.ir.metadata["routing"]["direction_swap_count"] == 1
    assert fqd.validate_deployment_package(package) is package
    assert (
        package.metadata["routing_evidence"]["routing_plan"]
        == package.metadata["routing_plan"]
    )
