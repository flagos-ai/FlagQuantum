from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_submission_readiness.py"
SPEC = importlib.util.spec_from_file_location("sc27_submission_readiness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_empty_control_fails_every_terminal_deliverable_without_crashing() -> None:
    result = MODULE.audit({
        "schema": "flagquantum.sc27.submission_control.v1",
        "artifacts": {},
    })
    assert result["submission_ready"] is False
    assert "artifact:container_identities:path_map_missing" in result["blockers"]
    assert "artifact:profile_preflight_payload:path_missing" in result["blockers"]
    assert "artifact:timed_campaign_manifest:path_missing" in result["blockers"]
    assert "artifact:independent_reproduction_report:path_missing" in result["blockers"]
    assert "paper:pdf_missing" in result["blockers"]


def test_rejects_invalid_control_schema() -> None:
    result = MODULE.audit({"schema": "wrong", "artifacts": {}})
    assert "control:schema_invalid" in result["blockers"]


def test_rejects_incomplete_container_identity_set() -> None:
    result = MODULE.audit({
        "schema": "flagquantum.sc27.submission_control.v1",
        "artifacts": {"container_identities": {"core": "/missing/core.json"}},
    })
    assert "containers:image_set_incomplete" in result["blockers"]
    assert "artifact:container_identity:pennylane:path_missing" in result["blockers"]
