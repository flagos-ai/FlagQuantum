from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


def _module(filename: str):
    path = Path(__file__).resolve().parents[2] / "paper/sc27" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    payload = _module("audit_payload.py")
    # Reuse the canonical contract fixture without importing pytest-dependent code.
    test_path = Path(__file__).with_name("test_sc27_payload_audit.py")
    spec = importlib.util.spec_from_file_location("payload_fixture", test_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    value = copy.deepcopy(fixture.valid_payload())
    assert payload.audit(value)["paper_ready_independent_run"]
    value.update(
        {
            "artifact_class": "measured_profile_preflight",
            "benchmark_evidence_class": "profile_preflight",
            "non_release_evidence": True,
            "release_gate_allowed": False,
            "scalability_blockers": ["profile_preflight_not_release_evidence"],
            "workload": {
                "n_wires": 28,
                "layers": 8,
                "seed": 41,
                "optimizer": "adam",
                "learning_rate": 0.01,
            },
            "ablation": {"id": "full"},
        }
    )
    return value


def test_accepts_frozen_non_release_profile_preflight() -> None:
    result = _module("audit_profile_preflight.py").audit(_payload())
    assert result["profile_preflight_passed"] is True
    assert result["blockers"] == []


def test_rejects_preflight_mislabeled_as_release_evidence() -> None:
    payload = _payload()
    payload["release_gate_allowed"] = True
    result = _module("audit_profile_preflight.py").audit(payload)
    assert "evidence:preflight_release_gate_allowed" in result["blockers"]
