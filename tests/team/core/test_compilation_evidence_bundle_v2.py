from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from flagquantum.core._compilation_evidence import (
    MappingTransitionEvidence,
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v2 import (
    CompilationEvidenceBundleV2,
    DirectedCouplingEvidence,
    PhysicalInstructionEvidenceV2,
    PhysicalPlanEvidenceV2,
)

pytestmark = pytest.mark.unit


def _instruction(
    index: int,
    opcode: str,
    wires: tuple[int, ...],
    *,
    layer: int,
    predecessors: tuple[int, ...],
) -> PhysicalInstructionEvidenceV2:
    return PhysicalInstructionEvidenceV2(
        instruction_index=index,
        source_instruction_index=0,
        topology_instruction_index=0,
        native_replacement_ordinal=0,
        native_instruction_index=0,
        direction_replacement_ordinal=index,
        direction_rewrite="reverse_cx_h_conjugation",
        origin="direction_rewrite",
        opcode=opcode,
        logical_wires=(0, 1),
        physical_wires=wires,
        layer=layer,
        predecessors=predecessors,
        dependency_kinds=(() if not predecessors else ("wire",)),
    )


def _plan() -> PhysicalPlanEvidenceV2:
    coupling = DirectedCouplingEvidence(2, ((1, 0),))
    instructions = (
        _instruction(0, "h", (0,), layer=0, predecessors=()),
        _instruction(1, "h", (1,), layer=0, predecessors=()),
        _instruction(2, "cx", (1, 0), layer=1, predecessors=(0, 1)),
        _instruction(3, "h", (0,), layer=2, predecessors=(2,)),
        _instruction(4, "h", (1,), layer=2, predecessors=(2,)),
    )
    return PhysicalPlanEvidenceV2(
        source_circuit_hash="a" * 64,
        physical_circuit_hash="b" * 64,
        target_snapshot_id="c" * 64,
        topology_identity=coupling.topology_identity,
        coupling=coupling,
        initial_logical_to_physical=(0, 1),
        pre_restore_logical_to_physical=(0, 1),
        final_logical_to_physical=(0, 1),
        mapping_transitions=(),
        instructions=instructions,
        topology_legalization_identity="d" * 64,
        native_gate_legalization_identity="e" * 64,
        direction_legalization_identity="f" * 64,
        reversed_cx_count=1,
        schedule_identity="1" * 64,
        schedule_depth=3,
        maximum_parallel_width=2,
        critical_path=(0, 2, 3),
    )


def _bundle() -> CompilationEvidenceBundleV2:
    return CompilationEvidenceBundleV2(
        producer="phase40-core-tests",
        source={
            "source_artifact_identity": "2" * 64,
            "circuit_artifact_identity": "3" * 64,
            "binding_identity": None,
            "source_circuit_hash": "a" * 64,
            "final_circuit_hash": "b" * 64,
        },
        target={
            "snapshot_id": "c" * 64,
            "target_legalization_identity": "4" * 64,
        },
        physical_plan=_plan(),
        output={
            "profile": "openqasm-3.0",
            "payload_sha256": "5" * 64,
            "emission_identity": "6" * 64,
            "conformance_identity": "7" * 64,
            "executable_artifact_identity": "8" * 64,
            "artifact_compilation_identity": "9" * 64,
        },
    )


def test_v2_round_trip_is_canonical_and_dispatches_explicitly() -> None:
    bundle = _bundle()
    restored = read_compilation_evidence_bundle_json(bundle.to_json())

    assert isinstance(restored, CompilationEvidenceBundleV2)
    assert restored == bundle
    assert restored.to_json() == bundle.to_json()
    assert restored.physical_plan.coupling.to_dict() == {
        "n_wires": 2,
        "directed_edges": [[1, 0]],
        "direction_semantics": "directed_cx",
    }
    with pytest.raises(TypeError):
        restored.source["binding_identity"] = "0" * 64  # type: ignore[index]


def test_v2_rejects_direction_topology_and_identity_tampering() -> None:
    payload = json.loads(_bundle().to_json())
    payload["physical_plan"]["instructions"][2]["physical_wires"] = [0, 1]
    with pytest.raises(ValueError, match="directed coupling"):
        CompilationEvidenceBundleV2.from_dict(payload)

    payload = json.loads(_bundle().to_json())
    payload["physical_plan"]["instructions"][0]["opcode"] = "x"
    with pytest.raises(ValueError, match="direction group"):
        CompilationEvidenceBundleV2.from_dict(payload)

    with pytest.raises(ValueError, match="bundle_identity"):
        replace(_bundle(), bundle_identity="0" * 64)


def test_v2_accepts_nonidentity_initial_layout_only_with_replay() -> None:
    plan = _plan()
    transition = MappingTransitionEvidence(
        routed_instruction_index=0,
        source_instruction_index=0,
        phase="final_restore",
        physical_wires=(0, 1),
        layout_before=(1, 0),
        layout_after=(0, 1),
    )
    replaced = replace(
        plan,
        initial_logical_to_physical=(1, 0),
        pre_restore_logical_to_physical=(1, 0),
        mapping_transitions=(transition,),
        plan_identity="",
    )

    assert replaced.initial_logical_to_physical == (1, 0)
    with pytest.raises(ValueError, match="do not reach final layout"):
        replace(
            plan,
            initial_logical_to_physical=(1, 0),
            pre_restore_logical_to_physical=(1, 0),
            plan_identity="",
        )


def test_v2_rejects_unknown_fields_and_record_limits() -> None:
    payload = _bundle().to_dict()
    with pytest.raises(ValueError, match="unknown compilation evidence bundle"):
        CompilationEvidenceBundleV2.from_dict({**payload, "future": True})

    oversized = copy.deepcopy(payload)
    oversized["physical_plan"]["instructions"] *= 4097
    with pytest.raises(ValueError, match="record limits"):
        CompilationEvidenceBundleV2.from_dict(oversized)


def test_v2_bundle_identity_is_stable_across_python_hash_seeds() -> None:
    script = (
        "import sys; "
        "from flagquantum.core._compilation_evidence import "
        "read_compilation_evidence_bundle_json; "
        "print(read_compilation_evidence_bundle_json(sys.stdin.read()).bundle_identity)"
    )
    identities = []
    for seed in ("1", "8675309"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=_bundle().to_json(),
            text=True,
            capture_output=True,
            check=True,
            env=environment,
        )
        identities.append(completed.stdout.strip())

    assert identities == [_bundle().bundle_identity] * 2
