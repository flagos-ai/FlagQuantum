"""Offline conformance for Quafu dynamic provider legality preview."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap
from flagquantum.core.target_capabilities import TargetCapabilitySnapshot
from flagquantum.dynamic import DynamicCircuit
from flagquantum.remote.qpu.quafu import QuafuProvider
from flagquantum.remote.qpu.quafu_dynamic import (
    _quafu_dynamic_snapshot_from_fixture,
)
from flagquantum.remote.qpu.quafu_dynamic_legality import (
    _preview_quafu_dynamic_request,
    _QuafuDynamicLegalityPreview,
)
from flagquantum.runtime.dynamic.routing import route_dynamic_circuit

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "quafu_dynamic_capability_complete_candidate.json"
)
EVALUATED_AT = datetime(2026, 10, 24, 10, 5, tzinfo=timezone.utc)


def _fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Quafu capability fixture must be a JSON object")
    return cast(dict[str, Any], payload)


def _snapshot() -> TargetCapabilitySnapshot:
    return _quafu_dynamic_snapshot_from_fixture(_fixture())


def _routed_feedback() -> DynamicCircuit:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    circuit.conditional("rx", 1, classical_bit=0, params={"theta": 0.5})
    return route_dynamic_circuit(circuit, CouplingMap.line(2))


def _preview(
    circuit: DynamicCircuit | None = None, **changes: object
) -> _QuafuDynamicLegalityPreview:
    snapshot = _snapshot()
    values: dict[str, object] = {
        "snapshot": snapshot,
        "preflight_snapshot_id": snapshot.snapshot_id,
        "target": "quafu:Baihua",
        "shots": 1024,
        "evaluated_at": EVALUATED_AT,
    }
    values.update(changes)
    return _preview_quafu_dynamic_request(
        circuit or _routed_feedback(),
        **values,  # type: ignore[arg-type]
    )


def test_preview_is_redacted_and_exposes_the_current_task_api_shape() -> None:
    preview = _preview()

    assert not preview.compatible
    assert preview.provider_legality == "rejected"
    assert preview.blockers == (
        "quafu_dynamic_request_shape_unsupported:physical_mapping_unbound",
    )
    assert preview.program_format == "openqasm-3.0"
    assert preview.program_sha256 is not None
    assert len(preview.program_sha256) == 64
    assert preview.program_bytes > 0
    assert preview.request_fields == (
        "circuits[].qasm",
        "client_ref",
        "mode",
        "shots",
        "target",
    )
    assert preview.physical_mapping_binding == "not_exposed"
    assert "program" not in preview.summary()
    assert "credential" not in preview.summary()


def test_snapshot_and_target_identity_must_match_preflight() -> None:
    preview = _preview(
        preflight_snapshot_id="different-snapshot",
        target="quafu:other",
    )

    assert preview.blockers[0] == "quafu_dynamic_snapshot_identity_mismatch"


def test_invalid_hardware_shots_are_rejected_offline() -> None:
    preview = _preview(shots=100)

    assert "quafu_dynamic_request_shape_unsupported:shots" in preview.blockers


def test_missing_native_operation_is_rejected() -> None:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    preview = _preview(route_dynamic_circuit(circuit, CouplingMap.line(2)))

    assert "quafu_dynamic_provider_legality_rejected:operation:x" in preview.blockers
    assert any("supported_conditional_gates" in item for item in preview.blockers)


def test_directed_topology_is_checked_for_the_exact_routed_artifact() -> None:
    circuit = DynamicCircuit(3)
    circuit.gate("cz", (0, 2))
    circuit.measure(0, classical_bit=0)
    circuit.conditional("rx", 1, classical_bit=0, params={"theta": 0.25})
    routed = route_dynamic_circuit(circuit, CouplingMap(3, ((0, 2),)))
    preview = _preview(routed)

    assert (
        "quafu_dynamic_provider_legality_rejected:directed_edge:0->2"
        in preview.blockers
    )


def test_unrouted_program_is_rejected_before_request_construction() -> None:
    circuit = DynamicCircuit(2).measure(0, classical_bit=0)
    circuit.conditional("rx", 1, classical_bit=0, params={"theta": 0.5})
    preview = _preview(circuit)

    assert (
        "quafu_dynamic_request_shape_unsupported:missing_routing_evidence"
        in preview.blockers
    )


def test_legality_preview_is_not_public() -> None:
    assert "preview_quafu_dynamic_request" not in fq.__all__


def test_preview_constructs_no_provider_and_consumes_no_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("legality preview must not construct a provider")

    monkeypatch.setattr(QuafuProvider, "__init__", reject_provider)
    before = torch.random.get_rng_state()

    _preview()

    assert torch.equal(torch.random.get_rng_state(), before)
