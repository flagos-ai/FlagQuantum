from __future__ import annotations

import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_container_identity.py"
SPEC = importlib.util.spec_from_file_location("sc27_container_identity", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def identity() -> dict:
    return {
        "schema": "flagquantum.sc27.container_identity.v1",
        "image": "core",
        "digest": "sha256:" + "a" * 64,
        "source_commit": "b" * 40,
        "source_dirty": False,
        "stage": "frozen",
        "base_provenance": "reconstructible_pinned_recipe",
        "locked_packages": ["torch==2.13.0"],
        "sbom": {"validated": True, "sha256": "sha256:" + "c" * 64},
        "probe": {"passed": True},
    }


def test_accepts_complete_frozen_identity() -> None:
    assert MODULE.audit(identity())["container_identity_ready"] is True


def test_probe_identity_cannot_be_promoted() -> None:
    value = identity()
    value.update({
        "source_dirty": True,
        "stage": "compatibility_probe",
        "base_provenance": "legacy_environment_snapshot",
    })
    frozen = MODULE.audit(value)
    assert frozen["container_identity_ready"] is False
    assert "promotion:stage_not_frozen" in frozen["blockers"]
    assert MODULE.audit(value, require_frozen=False)["container_identity_ready"] is True


def test_rejects_missing_sbom_or_probe() -> None:
    value = identity()
    value["sbom"]["validated"] = False
    value["probe"]["passed"] = False
    result = MODULE.audit(value)
    assert "sbom:not_validated" in result["blockers"]
    assert "probe:not_passed" in result["blockers"]
