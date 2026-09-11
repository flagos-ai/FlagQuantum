from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compiler.physical_plan import (
    PhysicalPlanError,
    build_physical_circuit_plan,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
from flagquantum.core.ir import CircuitIR, Instruction
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

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("physical:0",))
_SOURCE = FactSource(kind="phase36_test", ref="phase36-evidence")


def _snapshot(native_gates: tuple[object, ...]) -> TargetCapabilitySnapshot:
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 256,
        "precision.effective_dtype": "complex128",
        "gates.native": native_gates,
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase36-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase36-environment",
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
                evidence_id="phase36-evidence",
                sha256="c" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _legalize(
    circuit: CircuitIR,
    *,
    native_gates: tuple[object, ...],
    coupling_map: CouplingMap | None = None,
):
    return legalize_circuit_for_target(
        circuit,
        backend="pytorch",
        snapshot=_snapshot(native_gates),
        evaluated_at=_NOW,
        coupling_map=coupling_map,
        routing_strategy="restore_after_each_gate",
    )


def test_plan_retains_source_through_native_decomposition_without_topology() -> None:
    source = CircuitIR(
        2,
        (Instruction("x", (0,)), Instruction("h", (1,))),
        dtype="complex128",
    )
    legalization = _legalize(source, native_gates=("h", "z"))

    first = build_physical_circuit_plan(legalization)
    second = build_physical_circuit_plan(legalization)

    assert first == second
    assert first.program is legalization.program
    assert first.source_circuit_hash == source.content_hash
    assert first.mapping_transitions == ()
    assert first.schedule_depth == 3
    assert first.maximum_parallel_width == 2
    assert first.critical_path == (0, 1, 2)
    assert tuple(item.source_instruction_index for item in first.instructions) == (
        0,
        0,
        0,
        1,
    )
    assert tuple(item.origin for item in first.instructions) == (
        "native_decomposition",
        "native_decomposition",
        "native_decomposition",
        "source",
    )


def test_plan_replays_mapping_and_links_every_physical_instruction() -> None:
    source = CircuitIR(
        3,
        (Instruction("h", (0,)), Instruction("cx", (0, 2))),
        dtype="complex128",
    )
    coupling = CouplingMap.line(3)
    legalization = _legalize(
        source,
        native_gates=("h", "cx", "swap"),
        coupling_map=coupling,
    )
    plan = build_physical_circuit_plan(legalization, coupling_map=coupling)

    assert plan.source_circuit_hash == source.content_hash
    assert plan.final_logical_to_physical == (0, 1, 2)
    assert tuple(item.phase for item in plan.mapping_transitions) == (
        "forward",
        "gate_restore",
    )
    assert all(
        transition.layout_after != transition.layout_before
        for transition in plan.mapping_transitions
    )
    assert all(
        len(item.physical_wires) != 2 or coupling.has_edge(*item.physical_wires)
        for item in plan.instructions
    )
    assert tuple(item.source_instruction_index for item in plan.instructions) == (
        0,
        1,
        1,
        1,
    )
    assert tuple(item.origin for item in plan.instructions) == (
        "topology_mapped",
        "routing_swap",
        "topology_mapped",
        "routing_swap",
    )


def test_plan_tracks_native_lowering_of_inserted_routing_swaps() -> None:
    source = CircuitIR(3, (Instruction("cx", (0, 2)),), dtype="complex128")
    coupling = CouplingMap.line(3)
    legalization = _legalize(
        source,
        native_gates=("cx",),
        coupling_map=coupling,
    )
    plan = build_physical_circuit_plan(legalization, coupling_map=coupling)

    assert tuple(item.origin for item in plan.instructions) == (
        "routing_swap_decomposition",
        "routing_swap_decomposition",
        "routing_swap_decomposition",
        "topology_mapped",
        "routing_swap_decomposition",
        "routing_swap_decomposition",
        "routing_swap_decomposition",
    )
    assert {item.source_instruction_index for item in plan.instructions} == {0}
    assert tuple(item.native_replacement_ordinal for item in plan.instructions) == (
        0,
        1,
        2,
        0,
        0,
        1,
        2,
    )


def test_plan_fails_closed_on_topology_mismatch_or_identity_tampering() -> None:
    source = CircuitIR(3, (Instruction("cx", (0, 2)),), dtype="complex128")
    coupling = CouplingMap.line(3)
    legalization = _legalize(
        source,
        native_gates=("cx", "swap"),
        coupling_map=coupling,
    )
    plan = build_physical_circuit_plan(legalization, coupling_map=coupling)

    with pytest.raises(PhysicalPlanError, match="does not match"):
        build_physical_circuit_plan(
            legalization,
            coupling_map=CouplingMap(3, ((0, 2), (1, 2))),
        )
    with pytest.raises(PhysicalPlanError, match="plan_identity"):
        replace(plan, plan_identity="0" * 64)
    with pytest.raises(PhysicalPlanError, match="topology identity"):
        replace(plan, coupling_edges=((0, 2),), plan_identity="")
    with pytest.raises(PhysicalPlanError, match="instruction record"):
        replace(
            plan,
            instructions=(replace(plan.instructions[0], layer=99),)
            + plan.instructions[1:],
        )
