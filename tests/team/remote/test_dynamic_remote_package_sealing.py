"""Offline conformance for dynamic remote deployment-package sealing."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import flagquantum as fq
import flagquantum.deployment.dynamic_sealing as sealing
from flagquantum.compiler import CouplingMap
from flagquantum.deployment import CloudBackendProfile
from flagquantum.deployment.dynamic_rematch import evaluate_dynamic_routing_rematch
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.remote.qpu.quafu_dynamic import (
    _quafu_dynamic_snapshot_from_fixture,
)
from flagquantum.remote.qpu.quafu_dynamic_legality import (
    _preview_quafu_dynamic_request,
)
from flagquantum.runtime.dynamic.remote_capabilities import dynamic_remote_requirements
from flagquantum.runtime.dynamic.routing import route_dynamic_circuit

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "quafu_dynamic_capability_complete_candidate.json"
)
EVALUATED_AT = datetime(2026, 10, 24, 10, 5, tzinfo=timezone.utc)


def _fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return cast(dict[str, Any], payload)


def _inputs(*, provider_legality: str):
    circuit = DynamicCircuit(4).measure(0, classical_bit=0)
    circuit.conditional("rx", 1, classical_bit=0, params={"theta": 0.5})
    coupling = CouplingMap.line(4)
    routed = route_dynamic_circuit(circuit, coupling)
    snapshot = _quafu_dynamic_snapshot_from_fixture(_fixture())
    requirements = dynamic_remote_requirements(circuit)
    rematch = evaluate_dynamic_routing_rematch(
        requirements,
        routed,
        presented_source_requirement_set_id=requirements.requirement_set_id,
        requested_target_identity=snapshot.target_identity,
        preflight_snapshot_id=snapshot.snapshot_id,
        postrouting_snapshot=snapshot,
        evaluated_at=EVALUATED_AT,
        coupling_map=coupling,
        provider_legality=provider_legality,
        routing_reused=True,
    )
    preview = _preview_quafu_dynamic_request(
        routed,
        snapshot=snapshot,
        preflight_snapshot_id=snapshot.snapshot_id,
        target="quafu:Baihua",
        shots=1024,
        evaluated_at=EVALUATED_AT,
    )
    backend = CloudBackendProfile(
        provider="quafu",
        name="Baihua",
        n_qubits=4,
        basis_gates=("cz", "h", "rx", "ry", "rz"),
        coupling_map=coupling,
        supports_dynamic_circuits=True,
        max_classical_bits=1,
        dynamic_dialect="openqasm3",
    )
    return routed, backend, rematch, preview


def test_current_quafu_contract_refuses_before_package_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed, backend, rematch, preview = _inputs(provider_legality="rejected")

    def reject_package(*args: object, **kwargs: object) -> None:
        raise AssertionError("a rejected legality decision must not create a package")

    monkeypatch.setattr(sealing, "create_dynamic_deployment_package", reject_package)
    with pytest.raises(
        CapabilityError,
        match="dynamic_package_sealing_blocked:post_routing_rematch",
    ):
        sealing._seal_dynamic_remote_package(
            routed,
            backend=backend,
            rematch=rematch,
            provider_legality=preview.summary(),
            shots=1024,
        )


def test_accepted_offline_contract_seals_the_complete_identity_chain() -> None:
    routed, backend, rematch, preview = _inputs(provider_legality="accepted")
    accepted = replace(
        preview,
        physical_mapping_binding="provider_precompiled_program",
        provider_legality="accepted",
        blockers=(),
    )

    package = sealing._seal_dynamic_remote_package(
        routed,
        backend=backend,
        rematch=rematch,
        provider_legality=accepted.summary(),
        shots=1024,
    )
    seal = package.metadata["dynamic_remote_package_seal"]

    assert seal == {
        "schema": "flagquantum_dynamic_remote_package_seal_v1",
        "source_requirement_set_id": rematch.rematch.source_requirement_set_id,
        "target_snapshot_id": rematch.rematch.snapshot_id,
        "calibration_id": "fixture-calibration-1",
        "routed_artifact_sha256": rematch.routed_artifact_sha256,
        "routing_evidence_sha256": rematch.routing_evidence_sha256,
        "provider_legality_identity": seal["provider_legality_identity"],
        "package_artifact_sha256": package.metadata["deployment_artifact_sha256"],
    }
    assert len(seal["provider_legality_identity"]) == 64
    assert sealing._validate_dynamic_remote_package_seal(package) is package


def test_tampered_private_seal_is_rejected() -> None:
    routed, backend, rematch, preview = _inputs(provider_legality="accepted")
    accepted = replace(
        preview,
        physical_mapping_binding="provider_precompiled_program",
        provider_legality="accepted",
        blockers=(),
    )
    package = sealing._seal_dynamic_remote_package(
        routed,
        backend=backend,
        rematch=rematch,
        provider_legality=accepted.summary(),
    )
    tampered = replace(
        package,
        metadata={
            **package.metadata,
            "dynamic_remote_package_seal": {
                **package.metadata["dynamic_remote_package_seal"],
                "calibration_id": "different-calibration",
            },
        },
    )

    with pytest.raises(ValueError, match="seal digest mismatch"):
        sealing._validate_dynamic_remote_package_seal(tampered)


def test_offline_sealing_is_not_public() -> None:
    assert "seal_dynamic_remote_package" not in fq.__all__
