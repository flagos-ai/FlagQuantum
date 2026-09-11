from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v2 import (
    DirectedCouplingEvidence,
    PhysicalInstructionEvidenceV2,
)
from flagquantum.core._compilation_evidence_v3 import (
    CompilationEvidenceBundleV3,
    MappingTransitionEvidenceV3,
    PhysicalPlanEvidenceV3,
)

pytestmark = pytest.mark.unit


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan() -> PhysicalPlanEvidenceV3:
    coupling = DirectedCouplingEvidence(3, ((0, 1), (1, 2)))
    initial_layout = (0, 2)
    initial_occupancy = (0, None, 1)
    result_slots = (0, 2)
    allocation_identity = _digest(
        {
            "logical_wire_count": 2,
            "physical_slot_count": 3,
            "initial_logical_to_physical": initial_layout,
            "initial_physical_to_logical": initial_occupancy,
            "logical_result_physical_slots": result_slots,
            "workspace_initial_state": "standard_zero",
            "workspace_cleanup": "inverse_routing_swaps",
        }
    )
    transitions = (
        MappingTransitionEvidenceV3(
            routed_instruction_index=0,
            source_instruction_index=0,
            phase="forward",
            physical_wires=(0, 1),
            layout_before=(0, 2),
            layout_after=(1, 2),
            physical_to_logical_before=(0, None, 1),
            physical_to_logical_after=(None, 0, 1),
        ),
        MappingTransitionEvidenceV3(
            routed_instruction_index=2,
            source_instruction_index=0,
            phase="final_restore",
            physical_wires=(0, 1),
            layout_before=(1, 2),
            layout_after=(0, 2),
            physical_to_logical_before=(None, 0, 1),
            physical_to_logical_after=(0, None, 1),
        ),
    )
    instructions = (
        PhysicalInstructionEvidenceV2(
            instruction_index=0,
            source_instruction_index=0,
            topology_instruction_index=0,
            native_replacement_ordinal=0,
            native_instruction_index=0,
            direction_replacement_ordinal=0,
            direction_rewrite="none",
            origin="routing_swap",
            opcode="swap",
            logical_wires=(0, 1),
            physical_wires=(0, 1),
            layer=0,
            predecessors=(),
            dependency_kinds=(),
        ),
        PhysicalInstructionEvidenceV2(
            instruction_index=1,
            source_instruction_index=0,
            topology_instruction_index=1,
            native_replacement_ordinal=0,
            native_instruction_index=1,
            direction_replacement_ordinal=0,
            direction_rewrite="none",
            origin="topology_mapped",
            opcode="cx",
            logical_wires=(0, 1),
            physical_wires=(1, 2),
            layer=1,
            predecessors=(0,),
            dependency_kinds=("wire",),
        ),
        PhysicalInstructionEvidenceV2(
            instruction_index=2,
            source_instruction_index=0,
            topology_instruction_index=2,
            native_replacement_ordinal=0,
            native_instruction_index=2,
            direction_replacement_ordinal=0,
            direction_rewrite="none",
            origin="routing_swap",
            opcode="swap",
            logical_wires=(0, 1),
            physical_wires=(0, 1),
            layer=2,
            predecessors=(1,),
            dependency_kinds=("wire",),
        ),
    )
    return PhysicalPlanEvidenceV3(
        source_circuit_hash="a" * 64,
        physical_circuit_hash="b" * 64,
        target_snapshot_id="c" * 64,
        topology_identity=coupling.topology_identity,
        coupling=coupling,
        initial_logical_to_physical=initial_layout,
        pre_restore_logical_to_physical=(1, 2),
        final_logical_to_physical=result_slots,
        mapping_transitions=transitions,
        instructions=instructions,
        topology_legalization_identity="d" * 64,
        native_gate_legalization_identity="e" * 64,
        direction_legalization_identity="f" * 64,
        reversed_cx_count=0,
        schedule_identity="1" * 64,
        schedule_depth=3,
        maximum_parallel_width=1,
        critical_path=(0, 1, 2),
        logical_wire_count=2,
        physical_slot_count=3,
        initial_physical_to_logical=initial_occupancy,
        pre_restore_physical_to_logical=(None, 0, 1),
        final_physical_to_logical=initial_occupancy,
        logical_result_physical_slots=result_slots,
        allocation_identity=allocation_identity,
    )


def _bundle() -> CompilationEvidenceBundleV3:
    return CompilationEvidenceBundleV3(
        producer="phase46-core-tests",
        source={
            "source_artifact_identity": "2" * 64,
            "circuit_artifact_identity": "3" * 64,
            "binding_identity": None,
            "source_circuit_hash": "a" * 64,
            "final_circuit_hash": "b" * 64,
        },
        target={"snapshot_id": "c" * 64, "target_legalization_identity": "4" * 64},
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


def test_v3_round_trip_is_canonical_and_immutable() -> None:
    bundle = _bundle()
    restored = read_compilation_evidence_bundle_json(bundle.to_json())
    assert isinstance(restored, CompilationEvidenceBundleV3)
    assert restored == bundle
    assert restored.to_json() == bundle.to_json()
    with pytest.raises(TypeError):
        restored.source["binding_identity"] = "0" * 64  # type: ignore[index]


def test_v3_rejects_occupancy_projection_and_identity_tampering() -> None:
    payload = json.loads(_bundle().to_json())
    payload["physical_plan"]["mapping_transitions"][0]["physical_to_logical_after"] = [
        0,
        None,
        1,
    ]
    with pytest.raises(ValueError, match="occupancy"):
        CompilationEvidenceBundleV3.from_dict(payload)

    payload = json.loads(_bundle().to_json())
    payload["physical_plan"]["logical_result_physical_slots"] = [1, 2]
    with pytest.raises(ValueError, match="result projection"):
        CompilationEvidenceBundleV3.from_dict(payload)

    with pytest.raises(ValueError, match="bundle_identity"):
        replace(_bundle(), bundle_identity="0" * 64)


def test_v3_rejects_unknown_fields_and_record_limits() -> None:
    payload = _bundle().to_dict()
    with pytest.raises(ValueError, match="unknown compilation evidence bundle"):
        CompilationEvidenceBundleV3.from_dict({**payload, "future": True})
    oversized = copy.deepcopy(payload)
    oversized["physical_plan"]["instructions"] *= 4097
    with pytest.raises(ValueError, match="record limits"):
        CompilationEvidenceBundleV3.from_dict(oversized)


def test_v3_bundle_identity_is_stable_across_python_hash_seeds() -> None:
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
