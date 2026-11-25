"""Generate v4 lineage from real routing, native lowering and block synthesis."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.artifacts.artifact_compilation import (
    ArtifactCompilationError,
    ArtifactCompilationResult,
    compile_circuit_artifact_for_target,
)
from flagquantum.compiler.artifacts.compilation_evidence import (
    build_compilation_evidence_bundle,
    verify_compilation_evidence_bundle,
)
from flagquantum.compiler.artifacts.physical_plan import (
    PhysicalPlanError,
)
from flagquantum.compiler.lowering.target_legalization import (
    legalize_circuit_for_target,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.core._artifacts import ProgramArtifactV2
from flagquantum.core._compilation_evidence import read_compilation_evidence_bundle_json
from flagquantum.core._compilation_evidence_v4 import (
    CompilationEvidenceBundleV4,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.parameters import Parameter
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
from flagquantum.deployment.artifact_dry_run import prepare_artifact_deployment
from flagquantum.errors import ExecutionError
from flagquantum.runtime.compilation_evidence import (
    validate_executable_result_samples,
    verify_compilation_evidence_handoff,
)
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.integration
_NOW = datetime(2026, 11, 19, tzinfo=timezone.utc)


def _snapshot() -> TargetCapabilitySnapshot:
    scope = CapabilityScope(device_ids=("block-evidence-test",))
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 1024,
        "precision.effective_dtype": "complex128",
        "gates.native": ("rx", "rz", "cz"),
        "artifacts.profiles": ("openqasm-3.0",),
        "measurements.results": ("samples",),
        "limits.maximum_shots": 1024,
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


def _compile(
    source: CircuitIR,
    *,
    routed: bool = False,
    budget: int = 256,
    strategy: str = "auto",
) -> ArtifactCompilationResult:
    source = replace(
        source, measurements=(MeasurementNode("samples", tuple(range(source.n_wires))),)
    )
    return compile_circuit_artifact_for_target(
        ProgramArtifactV2.from_circuit_ir(source, producer="block-test"),
        backend="qasm",
        profile="openqasm-3.0",
        producer="block-test",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
        coupling_map=CouplingMap.line(source.n_wires) if routed else None,
        max_native_added_operations=budget,
        routing_strategy=strategy,
    )


def _operator(program: CircuitIR) -> torch.Tensor:
    size = 1 << program.n_wires
    result = torch.eye(size, dtype=torch.complex128).reshape(
        *([2] * program.n_wires), size
    )
    for instruction in program.instructions:
        order = (
            *instruction.wires,
            *(q for q in range(program.n_wires) if q not in instruction.wires),
            program.n_wires,
        )
        inverse = tuple(order.index(q) for q in range(program.n_wires + 1))
        local_size = 1 << len(instruction.wires)
        local = gate_matrix(
            instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
        ).reshape(local_size, local_size)
        result = (
            (local @ result.permute(order).reshape(local_size, -1))
            .reshape(*([2] * program.n_wires), size)
            .permute(inverse)
        )
    return result.reshape(size, size)


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_automatic_target_artifact_has_v4_lineage_and_independent_operator(
    routed: bool, reverse: bool
) -> None:
    pair = (2, 0) if reverse else (0, 2)
    source = CircuitIR(
        3,
        (Instruction("swap", pair), Instruction("cx", pair[::-1])),
        dtype="complex128",
    )
    compiled = _compile(source, routed=routed)
    stage = compiled.legalization.native_block_optimization
    assert stage is not None and stage.rewrites
    assert (
        compiled.legalization.native_gate_legalization.program is stage.source_program
    )
    assert compiled.physical_plan.program is stage.program
    assert compiled.legalization.schedule.program is stage.program
    assert compiled.physical_plan.version == "4.0"
    assert sum(item.name == "cz" for item in stage.program.instructions) < sum(
        item.name == "cz" for item in stage.source_program.instructions
    )
    expected, actual = _operator(source), _operator(stage.program)
    phase = torch.trace(actual.conj().T @ expected) / len(expected)
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)
    assert float(abs(abs(phase) - 1)) <= 1e-11
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    decoded = read_compilation_evidence_bundle_json(bundle.to_json())
    assert decoded == bundle
    verify_compilation_evidence_bundle(decoded, compiled, snapshot=_snapshot())
    assert bundle.physical_plan.plan_identity == compiled.physical_plan.plan_identity
    assert (
        _compile(source, routed=routed).compilation_identity
        == compiled.compilation_identity
    )


def test_identity_replacement_does_not_change_target_or_artifact_consumers() -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    optimized = _compile(source)
    with patch(
        "flagquantum.compiler.lowering.target_legalization.optimize_native_blocks",
        return_value=None,
    ):
        baseline = _compile(source)
        legacy_bundle = build_compilation_evidence_bundle(
            baseline, snapshot=_snapshot(), producer="block-test"
        )
    assert baseline.physical_plan.version == legacy_bundle.version == "1.0"
    assert optimized.physical_plan.version == "4.0"
    assert baseline.legalization.native_block_optimization is None
    assert (
        sum(item.name == "cz" for item in baseline.legalization.program.instructions)
        == 4
    )
    assert (
        sum(item.name == "cz" for item in optimized.legalization.program.instructions)
        == 2
    )
    expected, actual = _operator(baseline.legalization.program), _operator(
        optimized.legalization.program
    )
    phase = torch.trace(actual.conj().T @ expected) / len(expected)
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)


def test_no_accepted_rewrite_preserves_legacy_artifact_bytes_and_identities() -> None:
    source = CircuitIR(2, (Instruction("cz", (0, 1)),), dtype="complex128")
    enabled = _compile(source)
    with patch(
        "flagquantum.compiler.lowering.target_legalization.optimize_native_blocks",
        return_value=None,
    ):
        baseline = _compile(source)
    stage = enabled.legalization.native_block_optimization
    assert stage is not None and stage.rewrites == ()
    assert stage.program is enabled.legalization.native_gate_legalization.program
    assert enabled.physical_plan.version == "1.0"
    assert (
        enabled.executable_artifact.to_json() == baseline.executable_artifact.to_json()
    )
    assert enabled.compilation_identity == baseline.compilation_identity
    assert (
        build_compilation_evidence_bundle(
            enabled, snapshot=_snapshot(), producer="block-test"
        ).to_json()
        == build_compilation_evidence_bundle(
            baseline, snapshot=_snapshot(), producer="block-test"
        ).to_json()
    )


@pytest.mark.parametrize(
    "theta",
    [
        Parameter("theta"),
        torch.tensor(0.2, requires_grad=True),
        torch.tensor([0.2, 0.3]),
    ],
)
def test_automatic_stage_preserves_parameters_as_boundaries(theta: object) -> None:
    source = CircuitIR(
        2,
        (
            Instruction("swap", (0, 1)),
            Instruction("rx", (0,), {"theta": theta}),
            Instruction("cx", (1, 0)),
        ),
        dtype="complex128",
    )
    target = legalize_circuit_for_target(
        source, backend="pytorch", snapshot=_snapshot(), evaluated_at=_NOW
    )
    stage = target.native_block_optimization
    assert stage is not None and stage.rewrites == ()
    assert any(
        item.params.get("theta") is theta for item in target.program.instructions
    )
    assert target.program is target.native_gate_legalization.program


@pytest.mark.parametrize("budget", [0, 22])
def test_automatic_stage_cannot_rescue_native_budget_exhaustion(budget: int) -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    with pytest.raises(ArtifactCompilationError, match="max_added_operations"):
        _compile(source, budget=budget)


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_collective_ownership_keeps_disjoint_spectator_and_original_sources(
    pair: tuple[int, int],
) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", pair),
            Instruction("rz", (2,), {"theta": 0.17}),
            Instruction("cx", pair[::-1]),
        ),
        dtype="complex128",
    )
    compiled = _compile(source)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    plan = bundle.physical_plan
    assert len(plan.block_rewrites) == 1
    block = plan.block_rewrites[0]
    assert {
        plan.native_input_instructions[index].source_instruction_index
        for index in block.input_instruction_indices
    } == {0, 2}
    unchanged = [
        item
        for item in plan.instructions
        if item.native_input_instruction_index is not None
    ]
    assert len(unchanged) == 1
    index = unchanged[0].native_input_instruction_index
    assert index is not None
    assert plan.native_input_instructions[index].source_instruction_index == 1
    for index in block.output_instruction_indices:
        assert plan.instructions[index].native_input_instruction_index is None
        assert plan.instructions[index].block_rewrite_index == 0


@pytest.mark.parametrize(
    "strategy", ["restore_after_each_gate", "persistent_layout", "sabre", "auto"]
)
def test_automatic_routed_plan_retains_topology_space_transitions(
    strategy: str,
) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", (0, 2)),
            Instruction("cx", (2, 0)),
            Instruction("cx", (1, 2)),
        ),
        dtype="complex128",
    )
    compiled = _compile(source, routed=True, strategy=strategy)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    plan = bundle.physical_plan
    assert plan.mapping_transition_space == "topology_program"
    assert plan.mapping_transitions
    assert (
        plan.initial_logical_to_physical == plan.final_logical_to_physical == (0, 1, 2)
    )
    for transition in plan.mapping_transitions:
        related = [
            item
            for item in plan.native_input_instructions
            if item.topology_instruction_index == transition.routed_instruction_index
        ]
        assert related
        assert all(
            item.origin in {"routing_swap", "routing_swap_decomposition"}
            for item in related
        )
        assert all(
            item.source_instruction_index == transition.source_instruction_index
            for item in related
        )
    expected, actual = _operator(source), _operator(compiled.legalization.program)
    phase = torch.trace(actual.conj().T @ expected) / len(expected)
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_inverse_swap_prefixes_preserve_bounded_collective_artifact_evidence(
    routed: bool, pair: tuple[int, int]
) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("rx", (0,), {"theta": 0.17}),
            Instruction("swap", pair),
            Instruction("swap", pair[::-1]),
            Instruction("rz", (2,), {"theta": 0.31}),
            Instruction("swap", pair),
            Instruction("swap", pair[::-1]),
        ),
        dtype="complex128",
    )
    compiled = _compile(source, routed=routed)
    stage = compiled.legalization.native_block_optimization
    assert stage is not None
    assert len(stage.source_program.instructions) > 64
    assert sum(item.name == "cz" for item in stage.program.instructions) == 0
    assert len(stage.rewrites) >= 2
    assert all(
        len(rewrite.input_instruction_indices) <= 64 for rewrite in stage.rewrites
    )
    expected, actual = _operator(source), _operator(stage.program)
    phase = torch.trace(actual.conj().T @ expected) / len(expected)
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)
    assert float(abs(abs(phase) - 1)) <= 1e-11
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    assert len(bundle.physical_plan.native_input_instructions) == len(
        stage.source_program.instructions
    )
    decoded = read_compilation_evidence_bundle_json(bundle.to_json())
    assert decoded == bundle
    verify_compilation_evidence_bundle(decoded, compiled, snapshot=_snapshot())


def test_complete_cancellation_retains_consumed_inputs_in_the_artifact_bundle() -> None:
    source = CircuitIR(
        2, (Instruction("cz", (0, 1)), Instruction("cz", (1, 0))), dtype="complex128"
    )
    compiled = _compile(source)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    assert len(bundle.physical_plan.native_input_instructions) == 2
    assert bundle.physical_plan.instructions == bundle.physical_plan.critical_path == ()
    assert bundle.physical_plan.block_rewrites[0].output_instruction_indices == ()
    assert bundle.physical_plan.schedule_depth == 0


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
    with pytest.raises(
        ArtifactCompilationError, match="initial layout must be identity"
    ):
        _compile(source, routed=True, strategy="sabre_layout")


@pytest.mark.parametrize(
    "mutation",
    [
        "native_ancestry",
        "block_identity",
        "output_owner",
        "schedule",
        "target",
        "program",
    ],
)
def test_actual_collective_plan_rejects_resealed_inconsistent_stage_evidence(
    mutation: str,
) -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    compiled = _compile(source)
    plan = compiled.physical_plan
    with pytest.raises((PhysicalPlanError, ValueError)):
        if mutation == "native_ancestry":
            replace(
                plan,
                native_input_instructions=(
                    replace(
                        plan.native_input_instructions[0], native_replacement_ordinal=99
                    ),
                    *plan.native_input_instructions[1:],
                ),
                plan_identity="",
            )
        elif mutation == "block_identity":
            replace(plan, native_block_optimization_identity="b" * 64, plan_identity="")
        elif mutation == "output_owner":
            record = plan._v4_evidence().instructions[0]
            replace(
                plan,
                instructions=(
                    replace(record, block_replacement_ordinal=99),
                    *plan._v4_evidence().instructions[1:],
                ),
                plan_identity="",
            )
        elif mutation == "schedule":
            legalization = replace(
                compiled.legalization,
                schedule=replace(compiled.legalization.schedule, depth=1),
            )
            replace(plan, legalization=legalization, schedule_depth=1, plan_identity="")
        elif mutation == "target":
            replace(plan, target_snapshot_id="b" * 64, plan_identity="")
        else:
            replace(
                plan,
                program=compiled.legalization.native_gate_legalization.program,
                plan_identity="",
            )


def test_runtime_accepts_v4_without_recompilation_or_evidence_mutation() -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    compiled = _compile(source)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    before = bundle.to_json()
    with patch(
        "flagquantum.compiler.lowering.target_legalization.optimize_native_blocks",
        side_effect=AssertionError("Runtime must remain read-only"),
    ):
        verify_compilation_evidence_handoff(
            bundle,
            compiled.input_artifact,
            compiled.executable_artifact,
            snapshot=_snapshot(),
        )
    assert bundle.to_json() == before
    samples = torch.tensor([[0, 1], [1, 0]])
    assert (
        validate_executable_result_samples(compiled.executable_artifact, samples)
        is samples
    )


@pytest.mark.parametrize(
    "role,field",
    [
        ("source", "source_artifact_identity"),
        ("target", "target_legalization_identity"),
        ("output", "payload_sha256"),
    ],
)
def test_runtime_rejects_rehashed_v4_handoff_role_substitution(
    role: str, field: str
) -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    compiled = _compile(source)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    fields = dict(
        bundle.source
        if role == "source"
        else bundle.target if role == "target" else bundle.output
    )
    fields[field] = "b" * 64
    if role == "source":
        forged = replace(bundle, source=fields, bundle_identity="")
    elif role == "target":
        forged = replace(bundle, target=fields, bundle_identity="")
    else:
        forged = replace(bundle, output=fields, bundle_identity="")
    with pytest.raises(ExecutionError, match="does not match"):
        verify_compilation_evidence_handoff(
            forged,
            compiled.input_artifact,
            compiled.executable_artifact,
            snapshot=_snapshot(),
        )


def test_v4_target_artifact_completes_offline_deployment_preparation() -> None:
    source = CircuitIR(
        3, (Instruction("swap", (0, 2)), Instruction("cx", (2, 0))), dtype="complex128"
    )
    compiled = _compile(source, routed=True)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    assert isinstance(bundle, CompilationEvidenceBundleV4)
    restored = read_compilation_evidence_bundle_json(bundle.to_json())
    prepared = prepare_artifact_deployment(
        compiled.executable_artifact,
        snapshot=_snapshot(),
        source=compiled.input_artifact,
        compilation_evidence=restored,
        provider="test",
        target_id="block-evidence-test",
        shots=128,
        evaluated_at=_NOW,
    )
    assert prepared.capability_match.executable
    assert prepared.compilation_evidence_identity == bundle.bundle_identity
    assert prepared.artifact_identity == compiled.executable_artifact.artifact_identity
    assert prepared.logical_result_width == 3
    assert prepared.physical_result_slots is None
    assert prepared.program == compiled.emission.text


@pytest.mark.parametrize("refusal", ["expired", "shots", "provider", "target"])
def test_v4_evidence_does_not_bypass_deployment_preflight(refusal: str) -> None:
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))), dtype="complex128"
    )
    compiled = _compile(source)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    with pytest.raises(ExecutionError):
        prepare_artifact_deployment(
            compiled.executable_artifact,
            snapshot=_snapshot(),
            source=compiled.input_artifact,
            compilation_evidence=bundle,
            provider="other" if refusal == "provider" else "test",
            target_id="other" if refusal == "target" else "block-evidence-test",
            shots=2048 if refusal == "shots" else 128,
            evaluated_at=_NOW + timedelta(hours=2) if refusal == "expired" else _NOW,
        )


@pytest.mark.parametrize("routed", [False, True])
def test_commuting_rotation_cleanup_reaches_verified_target_artifact(
    routed: bool,
) -> None:
    source = CircuitIR(
        2,
        tuple(
            instruction
            for x_angle, z_angle in (
                (0.17, 0.23),
                (-0.31, 0.41),
                (0.53, -0.61),
                (-0.71, 0.83),
            )
            for instruction in (
                Instruction("cz", (0, 1)),
                Instruction("rx", (0,), {"theta": x_angle}),
                Instruction("rz", (1,), {"theta": z_angle}),
            )
        ),
        dtype="complex128",
    )
    compiled = _compile(source, routed=routed)
    stage = compiled.legalization.native_block_optimization
    assert stage is not None and len(stage.rewrites) == 1
    assert len(stage.program.instructions) <= 16
    assert sum(item.name == "cz" for item in stage.program.instructions) == 2
    assert stage.rewrites[0].input_instruction_indices == tuple(range(12))
    assert stage.rewrites[0].output_instruction_indices == tuple(
        range(len(stage.program.instructions))
    )
    assert compiled.physical_plan.program is stage.program
    assert compiled.legalization.schedule.program is stage.program
    expected, actual = _operator(source), _operator(stage.program)
    phase = torch.trace(actual.conj().T @ expected) / len(expected)
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)
    assert float(abs(abs(phase) - 1)) <= 1e-11
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    decoded = read_compilation_evidence_bundle_json(bundle.to_json())
    assert decoded == bundle
    verify_compilation_evidence_bundle(decoded, compiled, snapshot=_snapshot())


@pytest.mark.parametrize("routed", [False, True])
def test_direct_cz_translation_reaches_the_unchanged_artifact_consumers(
    routed: bool,
) -> None:
    # A controlled phase needs two CZs already, so strict CZ block reduction
    # correctly leaves it alone. Shorter native lowering still reaches emission.
    source = CircuitIR(
        2, (Instruction("cphase", (1, 0), {"theta": 0.37}),), dtype="complex128"
    )
    compiled = _compile(source, routed=routed, budget=10)
    native = compiled.legalization.native_gate_legalization
    stage = compiled.legalization.native_block_optimization
    assert len(native.program.instructions) == 11
    assert native.decompositions[0].replacement_opcodes == tuple(
        item.name for item in native.program.instructions
    )
    assert stage is not None and stage.rewrites == ()
    assert stage.program is native.program
    assert compiled.physical_plan.version == "1.0"
    expected, actual = _operator(source), _operator(compiled.legalization.program)
    phase = torch.vdot(actual.flatten(), expected.flatten()) / 4
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)
    bundle = build_compilation_evidence_bundle(
        compiled, snapshot=_snapshot(), producer="block-test"
    )
    decoded = read_compilation_evidence_bundle_json(bundle.to_json())
    assert decoded == bundle
    verify_compilation_evidence_bundle(decoded, compiled, snapshot=_snapshot())
    with pytest.raises(ArtifactCompilationError, match="max_added_operations"):
        _compile(source, routed=routed, budget=9)
