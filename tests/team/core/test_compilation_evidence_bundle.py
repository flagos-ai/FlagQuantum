from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from flagquantum.core._compilation_evidence import (
    CompilationEvidenceBundle,
    PhysicalInstructionEvidence,
    PhysicalPlanEvidence,
    read_compilation_evidence_bundle_json,
)

pytestmark = pytest.mark.unit


def _plan() -> PhysicalPlanEvidence:
    return PhysicalPlanEvidence(
        source_circuit_hash="a" * 64,
        physical_circuit_hash="b" * 64,
        target_snapshot_id="c" * 64,
        topology_identity=None,
        coupling=None,
        initial_logical_to_physical=(0,),
        pre_restore_logical_to_physical=(0,),
        final_logical_to_physical=(0,),
        mapping_transitions=(),
        instructions=(
            PhysicalInstructionEvidence(
                instruction_index=0,
                source_instruction_index=0,
                topology_instruction_index=0,
                native_replacement_ordinal=0,
                origin="source",
                opcode="h",
                logical_wires=(0,),
                physical_wires=(0,),
                layer=0,
                predecessors=(),
                dependency_kinds=(),
            ),
        ),
        topology_legalization_identity=None,
        native_gate_legalization_identity="d" * 64,
        schedule_identity="e" * 64,
        schedule_depth=1,
        maximum_parallel_width=1,
        critical_path=(0,),
    )


def _bundle() -> CompilationEvidenceBundle:
    return CompilationEvidenceBundle(
        producer="phase37-core-tests",
        source={
            "source_artifact_identity": "f" * 64,
            "circuit_artifact_identity": "1" * 64,
            "binding_identity": None,
            "source_circuit_hash": "a" * 64,
            "final_circuit_hash": "b" * 64,
        },
        target={
            "snapshot_id": "c" * 64,
            "target_legalization_identity": "2" * 64,
        },
        physical_plan=_plan(),
        output={
            "profile": "openqasm-3.0",
            "payload_sha256": "3" * 64,
            "emission_identity": "4" * 64,
            "conformance_identity": "5" * 64,
            "executable_artifact_identity": "6" * 64,
            "artifact_compilation_identity": "7" * 64,
        },
    )


def test_bundle_round_trip_is_canonical_deterministic_and_immutable() -> None:
    bundle = _bundle()
    restored = read_compilation_evidence_bundle_json(bundle.to_json())

    assert restored == bundle
    assert restored.to_json() == bundle.to_json()
    assert len(bundle.bundle_identity) == 64
    assert bundle.physical_plan.plan_identity == restored.physical_plan.plan_identity
    with pytest.raises(TypeError):
        bundle.source["binding_identity"] = "8" * 64  # type: ignore[index]


def test_bundle_rejects_unknown_missing_duplicate_and_nonfinite_fields() -> None:
    payload = _bundle().to_dict()
    with pytest.raises(ValueError, match="unknown compilation evidence bundle"):
        CompilationEvidenceBundle.from_dict({**payload, "future": True})
    incomplete = dict(payload)
    incomplete.pop("target")
    with pytest.raises(ValueError, match="missing compilation evidence bundle"):
        CompilationEvidenceBundle.from_dict(incomplete)
    with pytest.raises(ValueError, match="duplicate compilation evidence field"):
        read_compilation_evidence_bundle_json('{"schema":"x","schema":"y"}')
    nonfinite = copy.deepcopy(payload)
    nonfinite["physical_plan"]["schedule_depth"] = float("nan")  # type: ignore[index]
    with pytest.raises(ValueError, match="schedule_depth"):
        CompilationEvidenceBundle.from_dict(nonfinite)


def test_bundle_rejects_nested_and_identity_tampering() -> None:
    payload = json.loads(_bundle().to_json())
    payload["physical_plan"]["instructions"][0]["opcode"] = "x"
    with pytest.raises(ValueError, match="plan_identity"):
        CompilationEvidenceBundle.from_dict(payload)

    payload = _bundle().to_dict()
    payload["bundle_identity"] = "0" * 64
    with pytest.raises(ValueError, match="bundle_identity"):
        CompilationEvidenceBundle.from_dict(payload)


def test_plan_rejects_inconsistent_schedule_lineage_and_record_limits() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="schedule summary"):
        replace(plan, schedule_depth=2, plan_identity="")
    with pytest.raises(ValueError, match="lineage"):
        replace(
            _bundle(),
            source={**_bundle().source, "final_circuit_hash": "9" * 64},
            bundle_identity="",
        )
    with pytest.raises(ValueError, match="too many instruction records"):
        replace(plan, instructions=plan.instructions * 4097, plan_identity="")


def test_bundle_rejects_locators_and_preserves_version_boundary() -> None:
    with pytest.raises(ValueError, match="prohibited locator"):
        replace(_bundle(), producer="https://example.test/evidence", bundle_identity="")
    with pytest.raises(ValueError, match="unsupported compilation evidence version"):
        replace(_bundle(), version="2.0", bundle_identity="")


def test_bundle_identity_is_stable_across_python_hash_seeds() -> None:
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
