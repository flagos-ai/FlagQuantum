from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.routing import CouplingMap
from flagquantum.compiler.topology import (
    routing_emission,
    routing_strategy,
    sabre,
    topology_legalization,
)
from flagquantum.compiler.topology.topology_legalization import (
    TopologyLegalizationError,
    legalize_circuit_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.target_capabilities import (
    CapabilityScope,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)


def _snapshot() -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase27-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase27-environment",
        ),
        scope=CapabilityScope(device_ids=("topology:0",)),
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(),
        evidence_refs=(),
    )


def _state(program: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        program,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def test_sabre_strategy_legalizes_topology_and_preserves_state() -> None:
    source = CircuitIR(
        4,
        (
            Instruction("h", (3,)),
            Instruction("cx", (0, 3)),
            Instruction("ry", (1,), params={"theta": 0.23}),
        ),
        dtype="complex128",
    )
    coupling = CouplingMap.line(4)
    result = legalize_circuit_topology(
        source,
        coupling_map=coupling,
        snapshot=_snapshot(),
        strategy="sabre",
    )

    assert result.strategy == "sabre"
    assert result.final_logical_to_physical == (0, 1, 2, 3)
    assert all(
        len(item.wires) != 2 or coupling.has_edge(*item.wires)
        for item in result.program.instructions
    )
    torch.testing.assert_close(_state(result.program), _state(source))


def test_sabre_layout_strategy_legalizes_topology_and_preserves_state() -> None:
    source = CircuitIR(
        4,
        (
            Instruction("h", (3,)),
            Instruction("cx", (0, 3)),
            Instruction("cx", (1, 2)),
            Instruction("ry", (1,), params={"theta": 0.23}),
        ),
        dtype="complex128",
    )
    coupling = CouplingMap.line(4)
    result = legalize_circuit_topology(
        source,
        coupling_map=coupling,
        snapshot=_snapshot(),
        strategy="sabre_layout",
    )

    assert result.strategy == "sabre_layout"
    # A moved initial layout must still satisfy the legalization postcondition
    # that the routed program restores the logical output layout.
    assert result.final_logical_to_physical == (0, 1, 2, 3)
    assert all(
        len(item.wires) != 2 or coupling.has_edge(*item.wires)
        for item in result.program.instructions
    )
    torch.testing.assert_close(_state(result.program), _state(source))


def test_nonlocal_gate_routes_to_edges_and_preserves_state() -> None:
    source = CircuitIR(
        4,
        (
            Instruction("h", (3,)),
            Instruction("cx", (0, 3)),
            Instruction("ry", (1,), params={"theta": 0.23}),
        ),
        dtype="complex128",
    )
    coupling = CouplingMap.line(4)
    result = legalize_circuit_topology(
        source,
        coupling_map=coupling,
        snapshot=_snapshot(),
        strategy="restore_after_each_gate",
    )

    assert result.source_program is source
    assert result.inserted_swap_count == 4
    assert result.final_logical_to_physical == (0, 1, 2, 3)
    assert all(
        len(item.wires) != 2 or coupling.has_edge(*item.wires)
        for item in result.program.instructions
    )
    torch.testing.assert_close(_state(result.program), _state(source))


def test_auto_routing_is_deterministic_and_bound_to_snapshot_and_topology() -> None:
    source = CircuitIR(
        4,
        (Instruction("cx", (0, 3)), Instruction("cx", (0, 3))),
        dtype="complex128",
    )
    first = legalize_circuit_topology(
        source,
        coupling_map=CouplingMap.line(4),
        snapshot=_snapshot(),
    )
    second = legalize_circuit_topology(
        source,
        coupling_map=CouplingMap.line(4),
        snapshot=_snapshot(),
    )

    assert first.strategy == "persistent_layout"
    assert first.topology_identity == second.topology_identity
    assert first.legalization_identity == second.legalization_identity
    assert first.program.to_json() == second.program.to_json()


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("shape", ["line", "star"])
def test_auto_reuses_only_its_own_plan_with_unchanged_evidence(
    warm_cache: bool, shape: str
) -> None:
    source = fq.Circuit(4).h(0).cx(0, 3).cx(0, 2).ry(3, theta=0.21).cx(3, 1).to_ir()
    device = (
        CouplingMap.line(4)
        if shape == "line"
        else CouplingMap(4, ((0, 1), (0, 2), (0, 3)))
    )
    if warm_cache:
        device.shortest_path(0, 3)
    cache_before = device.path_cache_info()
    for program in (source, fq.Circuit(4).h(3).cx(3, 0).cx(2, 0).to_ir(), source):
        with (
            patch.object(
                routing_strategy,
                "_plan_sabre_for_topology",
                wraps=routing_strategy._plan_sabre_for_topology,
            ) as selection_search,
            patch.object(
                routing_emission,
                "_plan_sabre_for_topology",
                wraps=routing_emission._plan_sabre_for_topology,
            ) as emission_search,
        ):
            result = legalize_circuit_topology(
                program, coupling_map=device, snapshot=_snapshot()
            )
        assert selection_search.call_count == 1
        assert emission_search.call_count == 0
        reference = legalize_circuit_topology(
            program,
            coupling_map=device,
            snapshot=_snapshot(),
            strategy=result.strategy,
        )
        assert result == reference
        assert result.program.to_json() == reference.program.to_json()
        assert device.path_cache_info() == cache_before
        torch.testing.assert_close(_state(result.program), _state(program))


def test_auto_retains_replanning_when_search_records_path_cache_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sabre, "STALLED_SWAP_LIMIT_PER_WIRE", 0)
    source = fq.Circuit(4).h(0).cx(0, 3).cx(0, 2).ry(3, theta=0.21).cx(3, 1).to_ir()
    with patch.object(
        routing_emission,
        "_plan_sabre_for_topology",
        wraps=routing_emission._plan_sabre_for_topology,
    ) as emission_search:
        result = legalize_circuit_topology(
            source, coupling_map=CouplingMap.line(4), snapshot=_snapshot()
        )
    reference = legalize_circuit_topology(
        source,
        coupling_map=CouplingMap.line(4),
        snapshot=_snapshot(),
        strategy="sabre",
    )
    assert result.strategy == "sabre"
    assert result.program.to_json() == reference.program.to_json()
    assert result.legalization_identity == reference.legalization_identity
    assert emission_search.call_count == 1
    assert result.program.metadata["routing"]["path_cache"]["delta"]["misses"] > 0


def test_auto_preserves_a_replacement_router_and_its_failures() -> None:
    source = fq.Circuit(4).h(0).cx(0, 3).cx(0, 2).cx(3, 1).to_ir()
    with patch.object(
        topology_legalization,
        "route_to_topology",
        wraps=topology_legalization.route_to_topology,
    ) as replacement:
        result = legalize_circuit_topology(
            source, coupling_map=CouplingMap.line(4), snapshot=_snapshot()
        )
    assert replacement.call_count == 1
    assert replacement.call_args.kwargs["strategy"] == result.strategy
    with (
        patch.object(
            topology_legalization,
            "route_to_topology",
            side_effect=RuntimeError("Replacement router rejected the program"),
        ),
        pytest.raises(TopologyLegalizationError, match="Replacement router"),
    ):
        legalize_circuit_topology(
            source, coupling_map=CouplingMap.line(4), snapshot=_snapshot()
        )


def test_reused_sabre_plan_preserves_a_symbolic_parameter() -> None:
    theta = fq.Parameter("theta")
    source = fq.Circuit(4).h(0).cx(0, 3).cx(0, 2).ry(3, theta=theta).cx(3, 1).to_ir()
    result = legalize_circuit_topology(
        source, coupling_map=CouplingMap.line(4), snapshot=_snapshot()
    )
    assert result.strategy == "sabre"
    assert (
        next(
            item.params["theta"]
            for item in result.program.instructions
            if item.name == "ry"
        )
        is theta
    )


def test_routing_preserves_parameter_object_and_gradient() -> None:
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    source = fq.Circuit(4).h(0).cx(0, 3).cx(0, 2).ry(3, theta=theta).cx(3, 1).to_ir()
    result = legalize_circuit_topology(
        source,
        coupling_map=CouplingMap.line(4),
        snapshot=_snapshot(),
    )
    assert result.strategy == "sabre"

    routed_parameter = next(
        item.params["theta"]
        for item in result.program.instructions
        if item.name == "ry"
    )
    assert routed_parameter is theta
    reference_loss = _state(source).real.sum()
    routed_loss = _state(result.program).real.sum()
    reference_gradient = torch.autograd.grad(reference_loss, theta, retain_graph=True)[
        0
    ]
    routed_gradient = torch.autograd.grad(routed_loss, theta)[0]
    torch.testing.assert_close(routed_loss, reference_loss)
    torch.testing.assert_close(routed_gradient, reference_gradient)


def test_nonlocal_channel_fails_topology_postcondition() -> None:
    channel = Instruction(
        "test_two_wire_channel",
        (0, 2),
        metadata={"is_channel": True},
    )
    source = CircuitIR(3, (channel,), dtype="complex128")
    with pytest.raises(TopologyLegalizationError, match="nonlocal two-qubit"):
        legalize_circuit_topology(
            source,
            coupling_map=CouplingMap.line(3),
            snapshot=_snapshot(),
        )


def test_routing_expansion_and_physical_ancilla_paths_fail_closed() -> None:
    source = CircuitIR(3, (Instruction("cx", (0, 2)),), dtype="complex128")
    with pytest.raises(TopologyLegalizationError, match="max_added_operations"):
        legalize_circuit_topology(
            source,
            coupling_map=CouplingMap.line(3),
            snapshot=_snapshot(),
            max_added_operations=1,
        )
    with pytest.raises(TopologyLegalizationError, match="explicit layout/lowering"):
        legalize_circuit_topology(
            source,
            coupling_map=CouplingMap(4, ((0, 3), (3, 2), (2, 1))),
            snapshot=_snapshot(),
        )
