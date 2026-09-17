"""Fail-closed conversion tests for research Twin evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _module():
    path = Path(__file__).parents[1] / "tools/convert_q_atlas_twin_evidence.py"
    spec = importlib.util.spec_from_file_location("convert_q_atlas_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(payload, field):
    unsigned = dict(payload)
    unsigned.pop(field, None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit():
    rows = [
        {
            "physical_program_identity": "1" * 64,
            "prediction_error_tv": 0.03,
            "finite_shot_two_sample_tv_bound_95": 0.08,
            "gate_passed": True,
        },
        {
            "physical_program_identity": "2" * 64,
            "prediction_error_tv": 0.04,
            "finite_shot_two_sample_tv_bound_95": 0.09,
            "gate_passed": True,
        },
    ]
    audit = {
        "schema": "flagquantum.q_atlas_connected_subgraph_future_audit.v1",
        "state": "validated_fixed_program_envelope",
        "qpu_identity": "quafu:Shenglian",
        "binding": {
            "ordered_physical_chain": [82, 75, 68, 62],
            "source_shadow_candidate_identity": "3" * 64,
            "topology_identity": "8" * 64,
        },
        "summary": {"all_predeclared_gates_passed": True},
        "program_audits": rows,
        "provenance": {"source_draft_identity": "7" * 64},
    }
    audit["audit_identity"] = _digest(audit, "audit_identity")
    return audit


def _binding(audit):
    binding = {
        "schema": "flagquantum.q_atlas_twin_evidence_binding.v1",
        "source_validation_identity": "7" * 64,
        "snapshot_identity": "4" * 64,
        "qpu_identity": "quafu:Shenglian",
        "topology_identity": "8" * 64,
        "ordered_physical_chain": [82, 75, 68, 62],
        "source_shadow_candidate_identity": "3" * 64,
        "supported_operations": ["h", "ry", "cz", "measure"],
        "maximum_instruction_count": 16,
        "circuit_identities": {"1" * 64: "5" * 64, "2" * 64: "6" * 64},
        "created_before_target_outcomes": True,
        "target_outcome_count_available_when_frozen": 0,
    }
    provisional = dict(binding)
    provisional["binding_identity"] = _digest(provisional, "binding_identity")
    return provisional


def test_conversion_preserves_prospective_identity_boundary():
    module = _module()
    audit = _audit()
    binding = _binding(audit)

    envelope = module.convert(audit, binding)

    assert envelope.evidence_identity == audit["audit_identity"]
    assert envelope.verified_tv_error_bound == pytest.approx(0.13)
    assert envelope.confidence_level == pytest.approx(0.95)
    assert envelope.verified_circuit_identities == ("5" * 64, "6" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("created_before_target_outcomes", False, "not frozen prospectively"),
        ("target_outcome_count_available_when_frozen", 1, "not frozen prospectively"),
        ("ordered_physical_chain", [62, 68, 75, 82], "ordered_physical_chain"),
        ("source_validation_identity", "9" * 64, "does not descend"),
    ],
)
def test_conversion_rejects_retrospective_or_changed_bindings(field, value, message):
    module = _module()
    audit = _audit()
    binding = _binding(audit)
    binding[field] = value
    binding["binding_identity"] = _digest(binding, "binding_identity")

    with pytest.raises(ValueError, match=message):
        module.convert(audit, binding)


def test_conversion_rejects_tampered_audit():
    module = _module()
    audit = _audit()
    binding = _binding(audit)
    audit["program_audits"][0]["prediction_error_tv"] = 0.01

    with pytest.raises(ValueError, match="audit identity changed"):
        module.convert(audit, binding)
