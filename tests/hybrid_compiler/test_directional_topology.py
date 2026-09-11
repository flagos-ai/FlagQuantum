from __future__ import annotations

import itertools
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
from flagquantum.compiler.physical_plan import build_physical_circuit_plan
from flagquantum.compiler.target_legalization import (
    TargetLegalizationError,
    legalize_circuit_for_target,
)
from flagquantum.core._artifacts import ProgramArtifactV2
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v2 import CompilationEvidenceBundleV2
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
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
    verify_compilation_evidence_handoff,
)
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("directional:0",))
_SOURCE = FactSource(kind="phase39_test", ref="phase39-evidence")


def _snapshot(
    native_gates: tuple[str, ...] = ("h", "cx", "ry")
) -> TargetCapabilitySnapshot:
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 4096,
        "precision.effective_dtype": "complex128",
        "gates.native": native_gates,
        "artifacts.profiles": ("openqasm-3.0",),
        "measurements.results": ("samples",),
        "limits.maximum_shots": 4096,
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase39-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase39-environment",
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
                evidence_id="phase39-evidence",
                sha256="d" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _state(program: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        program,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def _legalize(
    source: CircuitIR,
    graph: DirectedCouplingMap,
    layout: tuple[int, ...],
    *,
    native_gates: tuple[str, ...] = ("h", "cx", "ry"),
):
    return legalize_circuit_for_target(
        source,
        backend="pytorch",
        snapshot=_snapshot(native_gates),
        evaluated_at=_NOW,
        coupling_map=graph,
        initial_layout=layout,
        max_routing_added_operations=1024,
        max_direction_added_operations=4096,
    )


def test_directed_map_is_canonical_and_directional() -> None:
    graph = DirectedCouplingMap(3, ((2, 1), (0, 1)))

    assert graph.edges == ((0, 1), (2, 1))
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(1, 0)
    assert graph.has_weak_edge(1, 0)
    assert graph.shortest_path(0, 2) == (0, 1, 2)
    assert len(graph.topology_identity) == 64

    with pytest.raises(ValueError, match="unique"):
        DirectedCouplingMap(2, ((0, 1), (0, 1)))
    with pytest.raises(ValueError, match="self"):
        DirectedCouplingMap(2, ((0, 0),))
    with pytest.raises(ValueError, match="outside"):
        DirectedCouplingMap(2, ((0, 2),))


@pytest.mark.parametrize(
    ("layout", "control", "target"),
    tuple(
        (layout, control, target)
        for layout in itertools.permutations(range(3))
        for control, target in itertools.permutations(range(3), 2)
    ),
)
def test_all_three_wire_layouts_and_cx_orders_preserve_state(
    layout: tuple[int, ...], control: int, target: int
) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("ry", (control,), params={"theta": 0.31}),
            Instruction("ry", (target,), params={"theta": -0.23}),
            Instruction("cx", (control, target)),
        ),
        dtype="complex128",
    )
    graph = DirectedCouplingMap(3, ((0, 1), (2, 1)))
    legalization = _legalize(source, graph, layout)
    plan = build_physical_circuit_plan(legalization, coupling_map=graph)

    assert legalization.direction_legalization is not None
    assert plan.initial_logical_to_physical == layout
    assert plan.final_logical_to_physical == (0, 1, 2)
    assert plan.coupling_direction_semantics == "directed_cx"
    assert all(
        item.name != "cx" or graph.has_edge(*item.wires)
        for item in legalization.program.instructions
    )
    torch.testing.assert_close(_state(legalization.program), _state(source))


def test_reverse_cx_records_direction_lineage_and_is_deterministic() -> None:
    source = CircuitIR(
        2,
        (
            Instruction("ry", (0,), params={"theta": 0.41}),
            Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
    )
    graph = DirectedCouplingMap(2, ((1, 0),))
    first = _legalize(source, graph, (0, 1))
    second = _legalize(source, graph, (0, 1))
    plan = build_physical_circuit_plan(first, coupling_map=graph)

    assert first.legalization_identity == second.legalization_identity
    assert first.direction_legalization is not None
    assert first.direction_legalization.reversed_cx_count == 1
    rewritten = tuple(
        item for item in plan.instructions if item.direction_rewrite != "none"
    )
    assert len(rewritten) == 5
    assert {item.native_instruction_index for item in rewritten} == {1}
    assert tuple(item.direction_replacement_ordinal for item in rewritten) == tuple(
        range(5)
    )
    assert plan.reversed_cx_count == 1
    assert plan.direction_legalization_identity == (
        first.direction_legalization.legalization_identity
    )
    with pytest.raises(ValueError, match="legalization_identity"):
        replace(first.direction_legalization, legalization_identity="0" * 64)


def test_directed_routing_preserves_trainable_gradient() -> None:
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    source = CircuitIR(
        3,
        (
            Instruction("ry", (0,), params={"theta": theta}),
            Instruction("cx", (0, 2)),
            Instruction("ry", (2,), params={"theta": 0.17}),
        ),
        dtype="complex128",
    )
    result = _legalize(
        source,
        DirectedCouplingMap(3, ((0, 1), (2, 1))),
        (2, 0, 1),
    )

    reference_loss = _state(source).real.sum()
    routed_loss = _state(result.program).real.sum()
    reference_gradient = torch.autograd.grad(reference_loss, theta, retain_graph=True)[
        0
    ]
    routed_gradient = torch.autograd.grad(routed_loss, theta)[0]
    torch.testing.assert_close(routed_loss, reference_loss)
    torch.testing.assert_close(routed_gradient, reference_gradient)


def test_target_native_swap_uses_weak_physical_link() -> None:
    source = CircuitIR(
        3,
        (Instruction("h", (0,)), Instruction("cx", (0, 2))),
        dtype="complex128",
    )
    graph = DirectedCouplingMap(3, ((1, 0), (2, 1)))
    result = _legalize(
        source,
        graph,
        (0, 1, 2),
        native_gates=("h", "cx", "swap"),
    )
    plan = build_physical_circuit_plan(result, coupling_map=graph)

    assert any(item.name == "swap" for item in result.program.instructions)
    assert all(
        item.name != "swap" or graph.has_weak_edge(*item.wires)
        for item in result.program.instructions
    )
    assert plan.coupling_direction_semantics == "directed_cx"
    torch.testing.assert_close(_state(result.program), _state(source))


def test_invalid_layout_disconnected_graph_and_missing_h_fail_closed() -> None:
    source = CircuitIR(3, (Instruction("cx", (0, 2)),), dtype="complex128")

    with pytest.raises(TargetLegalizationError, match="complete.*permutation"):
        _legalize(source, DirectedCouplingMap(3, ((0, 1), (1, 2))), (0, 0, 2))
    with pytest.raises(TargetLegalizationError, match="No coupling path"):
        _legalize(source, DirectedCouplingMap(3, ((0, 1),)), (0, 1, 2))
    with pytest.raises(TargetLegalizationError, match="target-native h and cx"):
        _legalize(
            CircuitIR(2, (Instruction("cx", (0, 1)),), dtype="complex128"),
            DirectedCouplingMap(2, ((1, 0),)),
            (0, 1),
            native_gates=("cx",),
        )
    with pytest.raises(TargetLegalizationError, match="supports only"):
        legalize_circuit_for_target(
            source,
            backend="pytorch",
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            coupling_map=DirectedCouplingMap(3, ((0, 1), (1, 2))),
            routing_strategy="restore_after_each_gate",
        )


def test_directed_plan_reaches_executable_artifact_without_a_second_ir() -> None:
    source = CircuitIR(
        3,
        (Instruction("h", (0,)), Instruction("cx", (0, 2))),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1, 2)),),
    )
    artifact = ProgramArtifactV2.from_circuit_ir(source, producer="phase39-source")
    graph = DirectedCouplingMap(3, ((0, 1), (2, 1)))

    result = compile_circuit_artifact_for_target(
        artifact,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=_snapshot(),
        producer="flagquantum.compiler",
        evaluated_at=_NOW,
        coupling_map=graph,
        initial_layout=(2, 0, 1),
        max_routing_added_operations=1024,
        max_direction_added_operations=4096,
    )

    assert result.physical_plan.program is result.legalization.program
    assert result.physical_plan.initial_logical_to_physical == (2, 0, 1)
    assert result.physical_plan.final_logical_to_physical == (0, 1, 2)
    assert result.executable_artifact.profile["name"] == "openqasm-3.0"
    assert "OPENQASM 3.0;" in result.emission.text
    bundle = build_compilation_evidence_bundle(
        result,
        snapshot=_snapshot(),
        producer="flagquantum.compiler",
    )
    assert isinstance(bundle, CompilationEvidenceBundleV2)
    restored = read_compilation_evidence_bundle_json(bundle.to_json())
    assert restored == bundle
    assert restored.physical_plan.plan_identity == result.physical_plan.plan_identity
    verify_compilation_evidence_bundle(restored, result, snapshot=_snapshot())
    verify_compilation_evidence_handoff(
        restored,
        artifact,
        result.executable_artifact,
        snapshot=_snapshot(),
    )
    prepared = prepare_artifact_deployment(
        result.executable_artifact,
        snapshot=_snapshot(),
        source=artifact,
        compilation_evidence=restored,
        provider="flagquantum.test",
        target_id="phase39-target",
        shots=128,
        evaluated_at=_NOW,
    )
    assert prepared.compilation_evidence is restored
    assert prepared.compilation_evidence_identity == restored.bundle_identity
    with pytest.raises(ExecutionError, match="source lineage"):
        verify_compilation_evidence_handoff(
            restored,
            ProgramArtifactV2.from_circuit_ir(
                CircuitIR(1, (Instruction("h", (0,)),), dtype="complex128"),
                producer="wrong-source",
            ),
            result.executable_artifact,
            snapshot=_snapshot(),
        )
