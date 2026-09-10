from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.compiler.physical_plan import (
    PhysicalPlanError,
    build_physical_circuit_plan,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.compiler.target_legalization import (
    TargetLegalizationError,
    legalize_circuit_for_target,
)
from flagquantum.compiler.topology_legalization import (
    TopologyLegalizationError,
    legalize_circuit_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("allocation:0",))
_SOURCE = FactSource(kind="phase43_test", ref="phase43-evidence")


def _snapshot() -> TargetCapabilitySnapshot:
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 4096,
        "precision.effective_dtype": "complex128",
        "gates.native": ("h", "ry", "cx", "swap"),
        "measurements.results": ("samples",),
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase43-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase43-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=tuple(
            CapabilityFact(
                name=name,
                value=value,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=(
                    FactExposure.OBSERVED
                    if name == "precision.effective_dtype"
                    else FactExposure.DECLARED
                ),
                source=_SOURCE,
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="phase43-evidence",
                sha256="e" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _source(theta: object = 0.37) -> CircuitIR:
    return CircuitIR(
        2,
        (
            Instruction("ry", (0,), params={"theta": theta}),
            Instruction("h", (1,)),
            Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1)),),
    )


def _legalize(source: CircuitIR):
    graph = DirectedCouplingMap(3, ((0, 1), (1, 2)))
    result = legalize_circuit_for_target(
        source,
        backend="pytorch",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
        coupling_map=graph,
        initial_layout=(0, 2),
        max_routing_added_operations=32,
        max_direction_added_operations=64,
    )
    return graph, result


def _state(program: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        replace(program, measurements=()),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def test_idle_slot_routes_interaction_and_restores_logical_result_slots() -> None:
    source = _source()
    graph, result = _legalize(source)
    topology = result.topology_legalization
    assert topology is not None

    assert result.program.n_wires == 3
    assert result.program.measurements[0].wires == (0, 2)
    assert topology.logical_wire_count == 2
    assert topology.physical_slot_count == 3
    assert topology.initial_logical_to_physical == (0, 2)
    assert topology.logical_result_physical_slots == (0, 2)
    assert topology.initial_physical_to_logical == (0, None, 1)
    assert topology.final_physical_to_logical == (0, None, 1)
    assert topology.allocation_identity is not None
    assert len(topology.allocation_identity) == 64

    routing = topology.program.metadata["routing"]
    assert routing["schema"] == "flagquantum_directed_routing_plan_v2"
    assert routing["pre_restore_physical_to_logical"] == (None, 0, 1)
    assert routing["workspace_cleaned"] is True
    assert tuple(item.name for item in topology.program.instructions) == (
        "ry",
        "h",
        "swap",
        "cx",
        "swap",
    )
    swaps = tuple(item for item in topology.program.instructions if item.name == "swap")
    assert swaps[0].metadata["physical_to_logical_before"] == (0, None, 1)
    assert swaps[0].metadata["physical_to_logical_after"] == (None, 0, 1)
    assert swaps[1].metadata["physical_to_logical_before"] == (None, 0, 1)
    assert swaps[1].metadata["physical_to_logical_after"] == (0, None, 1)
    assert all(
        item.name != "cx" or graph.has_edge(*item.wires)
        for item in result.program.instructions
    )

    physical = _state(result.program).reshape(2, 2, 2)
    logical = _state(source).reshape(2, 2)
    torch.testing.assert_close(physical[:, 0, :], logical)
    torch.testing.assert_close(physical[:, 1, :], torch.zeros_like(logical))


def test_workspace_route_is_deterministic_and_preserves_gradient() -> None:
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    source = _source(theta)
    _, first = _legalize(source)
    _, second = _legalize(source)

    assert first.legalization_identity == second.legalization_identity
    first_topology = first.topology_legalization
    second_topology = second.topology_legalization
    assert first_topology is not None and second_topology is not None
    assert first_topology.allocation_identity == second_topology.allocation_identity

    reference_loss = _state(source).real.sum()
    physical = _state(first.program).reshape(2, 2, 2)
    routed_loss = physical[:, 0, :].real.sum()
    reference_gradient = torch.autograd.grad(reference_loss, theta, retain_graph=True)[
        0
    ]
    routed_gradient = torch.autograd.grad(routed_loss, theta)[0]
    torch.testing.assert_close(routed_loss, reference_loss)
    torch.testing.assert_close(routed_gradient, reference_gradient)


def test_allocation_validation_and_unimplemented_contracts_fail_closed() -> None:
    source = _source()
    graph, result = _legalize(source)

    with pytest.raises(PhysicalPlanError, match="physical-circuit-plan 3.0"):
        build_physical_circuit_plan(result, coupling_map=graph)
    with pytest.raises(TargetLegalizationError, match="inject every logical wire"):
        legalize_circuit_for_target(
            source,
            backend="pytorch",
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            coupling_map=graph,
            initial_layout=(0, 0),
        )
    with pytest.raises(TopologyLegalizationError, match="DirectedCouplingMap"):
        legalize_circuit_topology(
            source,
            coupling_map=CouplingMap(3, ((0, 1), (1, 2))),
            snapshot=_snapshot(),
        )
    with pytest.raises(TargetLegalizationError, match="observable remapping"):
        legalize_circuit_for_target(
            replace(
                source,
                measurements=(),
                observables=(ObservableNode("z", (0,)),),
            ),
            backend="pytorch",
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            coupling_map=graph,
            initial_layout=(0, 2),
        )
