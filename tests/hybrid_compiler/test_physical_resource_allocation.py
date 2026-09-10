from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.artifact_compilation import (
    compile_circuit_artifact_for_target,
)
from flagquantum.compiler.compilation_evidence import (
    build_compilation_evidence_bundle,
    verify_compilation_evidence_bundle,
)
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
from flagquantum.core._artifacts import (
    ProgramArtifactV2,
    ProgramArtifactV3,
    read_program_artifact_json,
)
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v3 import CompilationEvidenceBundleV3
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
from flagquantum.deployment.artifact_dry_run import prepare_artifact_deployment
from flagquantum.errors import ExecutionError
from flagquantum.runtime.artifact_preflight import preflight_executable_artifact
from flagquantum.runtime.compilation_evidence import (
    validate_executable_result_samples,
    verify_compilation_evidence_handoff,
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
        "artifacts.profiles": ("openqasm-2.0", "openqasm-3.0"),
        "limits.maximum_shots": 4096,
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

    plan = build_physical_circuit_plan(result, coupling_map=graph)
    assert plan.version == "3.0"
    assert plan.logical_wire_count == 2
    assert plan.physical_slot_count == 3
    assert plan.initial_logical_to_physical == (0, 2)
    assert plan.pre_restore_logical_to_physical == (1, 2)
    assert plan.final_logical_to_physical == (0, 2)
    assert plan.initial_physical_to_logical == (0, None, 1)
    assert plan.pre_restore_physical_to_logical == (None, 0, 1)
    assert plan.final_physical_to_logical == (0, None, 1)
    assert plan.logical_result_physical_slots == (0, 2)
    assert plan.allocation_identity is not None
    assert tuple(
        transition.physical_to_logical_before for transition in plan.mapping_transitions
    ) == ((0, None, 1), (None, 0, 1))
    assert tuple(
        transition.physical_to_logical_after for transition in plan.mapping_transitions
    ) == ((None, 0, 1), (0, None, 1))
    assert (
        build_physical_circuit_plan(result, coupling_map=graph).plan_identity
        == plan.plan_identity
    )
    with pytest.raises(PhysicalPlanError, match="occupancy"):
        replace(
            plan,
            mapping_transitions=(
                replace(
                    plan.mapping_transitions[0],
                    physical_to_logical_after=(0, None, 1),
                ),
            )
            + plan.mapping_transitions[1:],
            plan_identity="",
        )
    with pytest.raises(PhysicalPlanError, match="allocation evidence"):
        replace(plan, final_physical_to_logical=(None, 0, 1), plan_identity="")
    with pytest.raises(TargetLegalizationError, match="inject every logical wire"):
        legalize_circuit_for_target(
            source,
            backend="pytorch",
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            coupling_map=graph,
            initial_layout=(0, 0),
        )
    with pytest.raises(TopologyLegalizationError, match="explicit layout/lowering"):
        legalize_circuit_topology(
            source,
            coupling_map=CouplingMap(3, ((0, 2), (2, 1))),
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


@pytest.mark.parametrize("profile", ("openqasm-2.0", "openqasm-3.0"))
def test_allocated_plan_emits_program_artifact_v3_with_logical_projection(
    profile: str,
) -> None:
    source = _source()
    source_artifact = ProgramArtifactV2.from_circuit_ir(
        source,
        producer="phase45-source",
    )
    graph = DirectedCouplingMap(3, ((0, 1), (1, 2)))

    result = compile_circuit_artifact_for_target(
        source_artifact,
        backend="qasm",
        profile=profile,
        snapshot=_snapshot(),
        producer="flagquantum.compiler",
        evaluated_at=_NOW,
        coupling_map=graph,
        initial_layout=(0, 2),
        max_routing_added_operations=32,
        max_direction_added_operations=64,
    )

    artifact = result.executable_artifact
    assert isinstance(artifact, ProgramArtifactV3)
    assert artifact.compilation["physical_plan_identity"] == (
        result.physical_plan.plan_identity
    )
    assert artifact.compilation["allocation_identity"] == (
        result.physical_plan.allocation_identity
    )
    assert artifact.result_schema == {
        "kind": "samples",
        "logical_wires": (0, 1),
        "physical_result_slots": (0, 2),
        "ordering": "logical_wire_order",
        "shots_source": "execution_request",
    }
    assert result.conformance.reconstructed_program.measurements[0].wires == (0, 2)
    assert read_program_artifact_json(artifact.to_json()) == artifact
    evidence = build_compilation_evidence_bundle(
        result,
        snapshot=_snapshot(),
        producer="phase46-compiler",
    )
    assert isinstance(evidence, CompilationEvidenceBundleV3)
    assert evidence.physical_plan.plan_identity == result.physical_plan.plan_identity
    assert evidence.physical_plan.allocation_identity == (
        result.physical_plan.allocation_identity
    )
    assert evidence.physical_plan.logical_result_physical_slots == (0, 2)
    assert read_compilation_evidence_bundle_json(evidence.to_json()) == evidence
    verify_compilation_evidence_bundle(evidence, result, snapshot=_snapshot())
    verify_compilation_evidence_handoff(
        evidence,
        source_artifact,
        artifact,
        snapshot=_snapshot(),
    )
    with pytest.raises(ExecutionError, match="versions are incompatible"):
        verify_compilation_evidence_handoff(
            evidence,
            source_artifact,
            source_artifact,
            snapshot=_snapshot(),
        )
    assert preflight_executable_artifact(
        artifact,
        snapshot=_snapshot(),
        shots=8,
        evaluated_at=_NOW,
    ).executable
    with pytest.raises(TypeError, match="currently requires ProgramArtifactV2"):
        prepare_artifact_deployment(
            artifact,  # type: ignore[arg-type]
            snapshot=_snapshot(),
            source=source_artifact,
            compilation_evidence=evidence,  # type: ignore[arg-type]
            provider="flagquantum.test",
            target_id="phase43-target",
            shots=8,
            evaluated_at=_NOW,
        )
    samples = torch.zeros((1, 8, 2), dtype=torch.int64)
    assert validate_executable_result_samples(artifact, samples) is samples
    with pytest.raises(ExecutionError, match="sample width"):
        validate_executable_result_samples(
            artifact,
            torch.zeros((1, 8, 3), dtype=torch.int64),
        )
    with pytest.raises(ExecutionError, match="result-width dimension"):
        validate_executable_result_samples(artifact, torch.tensor(0))
    with pytest.raises(TypeError, match="torch.Tensor"):
        validate_executable_result_samples(artifact, [[0, 0]])  # type: ignore[arg-type]
    if profile == "openqasm-2.0":
        assert "qreg q[3];" in result.emission.text
        assert "creg c[2];" in result.emission.text
        assert result.emission.text.endswith(
            "measure q[0] -> c[0];\nmeasure q[2] -> c[1];"
        )
    else:
        assert "qubit[3] q;" in result.emission.text
        assert "bit[2] c;" in result.emission.text
        assert result.emission.text.endswith(
            "c[0] = measure q[0];\nc[1] = measure q[2];"
        )
