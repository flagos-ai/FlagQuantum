"""Frozen synthetic pre-v4 evidence; never regenerate to admit a reader change.

Captured from the existing Core test builders at c5f212000392df18a667bf6ac164f08bd8f33942.
These fixtures establish codec compatibility, not numerical or hardware evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flagquantum.core._compilation_evidence import (
    CompilationEvidenceBundle,
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v2 import CompilationEvidenceBundleV2
from flagquantum.core._compilation_evidence_v3 import CompilationEvidenceBundleV3

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    "version,expected_type,text_digest,bundle_identity,plan_identity",
    [
        (
            1,
            CompilationEvidenceBundle,
            "2120106b9ad8aebc25d00bf474f39e2422722e9625e0744ffd115329cd735389",
            "5a5c7d01d3fb676ca3588ce55360eea678a5c591e7258449d23aa5ff2396f067",
            "93135402bea178f3452e09b3ca9982d496dbb6bfaf8ebec79b7f45c79eab6435",
        ),
        (
            2,
            CompilationEvidenceBundleV2,
            "2a1b490f58dbdf40bf07c76417ac8b0536e0945d7d3217491ec53eb6e7205779",
            "af1f5c227d4b5a74549e7986ad364d2eca857ee7f5667f57bc87bdfe8d2ee9e1",
            "bf1ed1283185b667d51daf9b834e6b61039966f7cb650e4f76444393352c08dd",
        ),
        (
            3,
            CompilationEvidenceBundleV3,
            "2c5c157c754a18dbb36ce88084f2e59b242fcaac3c17c9189f0b9eb1c0b2233e",
            "7c47346f091e2f25eb628127544b48e2b602de8418a3719fefc6e57781208515",
            "2efca98888882bed855f0317da0ca2ceddef1ce0349abaeb05393ddfb8f57c1b",
        ),
    ],
)
def test_frozen_pre_v4_encoding_dispatch_and_identities(
    version: int,
    expected_type: type[
        CompilationEvidenceBundle
        | CompilationEvidenceBundleV2
        | CompilationEvidenceBundleV3
    ],
    text_digest: str,
    bundle_identity: str,
    plan_identity: str,
) -> None:
    text = (
        (_FIXTURES / f"compilation_evidence_v{version}_identity.json")
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == text_digest
    restored = read_compilation_evidence_bundle_json(text)
    assert type(restored) is expected_type
    assert restored.version == f"{version}.0"
    assert restored.bundle_identity == bundle_identity
    assert restored.physical_plan.plan_identity == plan_identity
    assert restored.to_json() == text
    # Whitespace does not define a new identity or silently select a newer reader.
    spaced = read_compilation_evidence_bundle_json(
        json.dumps(json.loads(text), indent=2)
    )
    assert type(spaced) is expected_type
    assert spaced.to_json() == text
    if version == 1:
        assert restored.physical_plan.coupling is None
        assert restored.physical_plan.initial_logical_to_physical == (0,)
    elif version == 2:
        assert isinstance(restored, CompilationEvidenceBundleV2)
        assert restored.physical_plan.reversed_cx_count == 1
        assert restored.physical_plan.coupling is not None
        assert restored.physical_plan.coupling.directed_edges == ((1, 0),)
    else:
        assert isinstance(restored, CompilationEvidenceBundleV3)
        assert restored.physical_plan.logical_wire_count == 2
        assert restored.physical_plan.physical_slot_count == 3
        assert restored.physical_plan.logical_result_physical_slots == (0, 2)
        assert restored.physical_plan.pre_restore_logical_to_physical == (1, 2)
        assert restored.physical_plan.final_logical_to_physical == (0, 2)
        assert len(restored.physical_plan.mapping_transitions) == 2


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize(
    "section,field",
    [
        ("source", "source_circuit_hash"),
        ("target", "snapshot_id"),
        ("output", "payload_sha256"),
    ],
)
def test_frozen_bundle_tampering_is_rejected_without_resealing(
    version: int, section: str, field: str
) -> None:
    payload = json.loads(
        (_FIXTURES / f"compilation_evidence_v{version}_identity.json").read_text(
            encoding="utf-8"
        )
    )
    payload[section][field] = "0" * 64
    failure = {
        "source": "compilation evidence lineage is inconsistent",
        "target": "plan_identity does not match physical plan evidence",
        "output": "bundle_identity does not match compilation evidence",
    }[section]
    with pytest.raises(ValueError, match=failure):
        read_compilation_evidence_bundle_json(json.dumps(payload))
