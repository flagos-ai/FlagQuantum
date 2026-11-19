"""Generate v4 lineage from real routing, native lowering and block synthesis."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compiler.artifacts.native_block_evidence import (
    build_native_block_physical_evidence,
)
from flagquantum.compiler.artifacts.physical_plan import (
    PhysicalCircuitPlan,
    PhysicalPlanError,
    build_physical_circuit_plan,
)
from flagquantum.compiler.lowering.native_block_optimization import (
    NativeBlockOptimizationError,
    NativeBlockOptimizationResult,
    optimize_native_blocks,
)
from flagquantum.compiler.lowering.native_gate_legalization import (
    NativeGateLegalizationError,
)
from flagquantum.compiler.lowering.schedule_legalization import (
    CircuitSchedule,
    schedule_circuit_dependencies,
)
from flagquantum.compiler.lowering.target_legalization import (
    legalize_circuit_for_target,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.core._compilation_evidence_v4 import PhysicalPlanEvidenceV4
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
_NOW = datetime(2026, 11, 19, tzinfo=timezone.utc)


def _snapshot() -> TargetCapabilitySnapshot:
    scope = CapabilityScope(device_ids=("block-evidence-test",))
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 1024,
        "precision.effective_dtype": "complex128",
        "gates.native": ("rx", "rz", "cz"),
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="block-evidence-test",
            target_class="test",
            provider="test",
            provider_version="1",
            target_revision="1",
            environment_id="offline",
        ),
        scope=scope,
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
                source=FactSource(kind="test", ref="block-evidence"),
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="block-evidence",
                sha256="a" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=scope,
            ),
        ),
    )


def _stages(
    source: CircuitIR,
    *,
    coupling: CouplingMap | None = None,
    strategy: str = "auto",
) -> tuple[
    PhysicalCircuitPlan,
    NativeBlockOptimizationResult,
    CircuitSchedule,
    TargetCapabilitySnapshot,
]:
    snapshot = _snapshot()
    target = legalize_circuit_for_target(
        source,
        backend="pytorch",
        snapshot=snapshot,
        evaluated_at=_NOW,
        coupling_map=coupling,
        routing_strategy=strategy,
    )
    baseline = build_physical_circuit_plan(target, coupling_map=coupling)
    optimization = optimize_native_blocks(
        target.native_gate_legalization,
        snapshot=snapshot,
        evaluated_at=_NOW,
        topology=target.topology_legalization,
        coupling_map=coupling,
    )
    assert optimization is not None
    schedule = schedule_circuit_dependencies(
        optimization.program, target_snapshot_id=snapshot.snapshot_id
    )
    return baseline, optimization, schedule, snapshot


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_generated_plan_keeps_many_to_many_ownership(pair: tuple[int, int]) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", pair),
            Instruction("rz", (2,), {"theta": 0.17}),
            Instruction("cx", pair[::-1]),
        ),
        dtype="complex128",
    )
    baseline, optimization, schedule, snapshot = _stages(source)
    evidence = build_native_block_physical_evidence(
        baseline, optimization, schedule, snapshot=snapshot, evaluated_at=_NOW
    )
    assert (
        PhysicalPlanEvidenceV4.from_dict(
            evidence.to_dict(), target_snapshot_id=snapshot.snapshot_id
        )
        == evidence
    )
    assert evidence.source_circuit_hash == source.content_hash
    assert evidence.native_input_circuit_hash == baseline.program.content_hash
    assert evidence.physical_circuit_hash == optimization.program.content_hash
    assert sum(item.opcode == "cz" for item in evidence.native_input_instructions) == 4
    assert sum(item.opcode == "cz" for item in evidence.instructions) == 2
    assert len(evidence.block_rewrites) == 1
    block = evidence.block_rewrites[0]
    assert {
        evidence.native_input_instructions[index].source_instruction_index
        for index in block.input_instruction_indices
    } == {0, 2}
    unchanged = [
        item
        for item in evidence.instructions
        if item.native_input_instruction_index is not None
    ]
    assert len(unchanged) == 1
    for item in unchanged:
        assert item.native_input_instruction_index is not None
        native = evidence.native_input_instructions[item.native_input_instruction_index]
        assert native.source_instruction_index == 1
        assert item.block_rewrite_index is None
    for index in block.output_instruction_indices:
        assert evidence.instructions[index].native_input_instruction_index is None
        assert evidence.instructions[index].block_rewrite_index == 0
    assert evidence.schedule_identity == schedule.schedule_identity
    assert evidence.schedule_depth == schedule.depth


@pytest.mark.parametrize(
    "strategy", ["restore_after_each_gate", "persistent_layout", "sabre", "auto"]
)
def test_routing_transitions_remain_in_topology_space(strategy: str) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", (0, 2)),
            Instruction("cx", (2, 0)),
            Instruction("cx", (1, 2)),
        ),
        dtype="complex128",
    )
    baseline, optimization, schedule, snapshot = _stages(
        source, coupling=CouplingMap.line(3), strategy=strategy
    )
    evidence = build_native_block_physical_evidence(
        baseline, optimization, schedule, snapshot=snapshot, evaluated_at=_NOW
    )
    assert (
        PhysicalPlanEvidenceV4.from_dict(
            evidence.to_dict(), target_snapshot_id=snapshot.snapshot_id
        )
        == evidence
    )
    assert evidence.mapping_transition_space == "topology_program"
    assert evidence.mapping_transitions
    assert evidence.initial_logical_to_physical == (0, 1, 2)
    assert evidence.final_logical_to_physical == (0, 1, 2)
    assert len(evidence.mapping_transitions) == len(baseline.mapping_transitions)
    for transition, retained in zip(
        evidence.mapping_transitions, baseline.mapping_transitions, strict=True
    ):
        assert transition.routed_instruction_index == retained.routed_instruction_index
        assert transition.layout_before == retained.layout_before
        assert transition.layout_after == retained.layout_after
    assert sum(item.opcode == "cz" for item in evidence.instructions) < sum(
        item.opcode == "cz" for item in evidence.native_input_instructions
    )


def test_complete_cancellation_retains_consumed_inputs() -> None:
    source = CircuitIR(
        2, (Instruction("cz", (0, 1)), Instruction("cz", (1, 0))), dtype="complex128"
    )
    baseline, optimization, schedule, snapshot = _stages(source)
    evidence = build_native_block_physical_evidence(
        baseline, optimization, schedule, snapshot=snapshot, evaluated_at=_NOW
    )
    assert len(evidence.native_input_instructions) == 2
    assert evidence.instructions == evidence.critical_path == ()
    assert evidence.block_rewrites[0].output_instruction_indices == ()
    assert evidence.schedule_depth == 0


def test_nonidentity_initial_placement_keeps_the_existing_artifact_refusal() -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", (0, 2)),
            Instruction("cx", (2, 0)),
            Instruction("cx", (1, 2)),
        ),
        dtype="complex128",
    )
    with pytest.raises(PhysicalPlanError, match="initial layout must be identity"):
        _stages(source, coupling=CouplingMap.line(3), strategy="sabre_layout")


@pytest.mark.parametrize("mutation", ["schedule", "program", "target", "budget"])
def test_builder_rejects_inconsistent_retained_stage_chain(mutation: str) -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    baseline, optimization, schedule, snapshot = _stages(source)
    budget = 256
    if mutation == "schedule":
        schedule = replace(schedule, depth=schedule.depth + 1)
    elif mutation == "program":
        schedule = replace(schedule, program=baseline.program)
    elif mutation == "target":
        schedule = replace(schedule, target_snapshot_id="b" * 64)
    else:
        budget = 0
    with pytest.raises(
        (PhysicalPlanError, NativeBlockOptimizationError, NativeGateLegalizationError)
    ):
        build_native_block_physical_evidence(
            baseline,
            optimization,
            schedule,
            snapshot=snapshot,
            evaluated_at=_NOW,
            max_added_operations=budget,
        )


def test_no_rewrite_keeps_the_existing_plan_version() -> None:
    source = CircuitIR(2, (Instruction("cz", (0, 1)),), dtype="complex128")
    baseline, optimization, schedule, snapshot = _stages(source)
    assert optimization.rewrites == ()
    assert baseline.version == "1.0"
    with pytest.raises(PhysicalPlanError, match="block rewrite"):
        build_native_block_physical_evidence(
            baseline, optimization, schedule, snapshot=snapshot, evaluated_at=_NOW
        )


def test_resealed_baseline_ancestry_is_not_accepted_as_its_own_proof() -> None:
    source = CircuitIR(
        2,
        (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))),
        dtype="complex128",
    )
    baseline, optimization, schedule, snapshot = _stages(source)
    forged = replace(
        baseline,
        instructions=(
            replace(baseline.instructions[0], native_replacement_ordinal=99),
            *baseline.instructions[1:],
        ),
        plan_identity="",
    )
    with pytest.raises(PhysicalPlanError, match="physical plan failed replay"):
        build_native_block_physical_evidence(
            forged, optimization, schedule, snapshot=snapshot, evaluated_at=_NOW
        )
