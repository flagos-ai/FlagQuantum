from __future__ import annotations

import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "paper" / "sc27" / "build_container_identity.py"
SPEC = importlib.util.spec_from_file_location("sc27_container_builder", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    lock = tmp_path / "locks" / "core.txt"
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(f"packaging=={version('packaging')}\n", encoding="utf-8")
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({
        "bomFormat": "CycloneDX", "specVersion": "1.6",
        "components": [{"name": "flagquantum"}],
    }), encoding="utf-8")
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({
        "schema": "flagquantum.sc27.profile_preflight_audit.v1",
        "profile_preflight_passed": True,
    }), encoding="utf-8")
    return lock, sbom, probe


def build(tmp_path: Path, **updates: object) -> dict:
    lock, sbom, probe = inputs(tmp_path)
    values = {
        "image": "core", "digest": "sha256:" + "a" * 64,
        "source_commit": "b" * 40, "source_dirty": False,
        "stage": "frozen", "base_provenance": "reconstructible_pinned_recipe",
        "lock": lock, "sbom": sbom, "probe_audit": probe,
    }
    values.update(updates)
    return MODULE.build_identity(**values)


def test_builds_content_addressed_identity(tmp_path: Path) -> None:
    identity = build(tmp_path)
    assert identity["locked_packages"] == [f"packaging=={version('packaging')}"]
    assert identity["sbom"]["validated"] is True
    assert identity["sbom"]["sha256"].startswith("sha256:")
    assert identity["probe"]["passed"] is True


def test_rejects_failed_or_wrong_provider_probe(tmp_path: Path) -> None:
    lock, sbom, probe = inputs(tmp_path)
    probe.write_text(json.dumps({
        "schema": "flagquantum.sc27.profile_preflight_audit.v1",
        "profile_preflight_passed": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        MODULE.build_identity(
            image="core", digest="sha256:" + "a" * 64,
            source_commit="b" * 40, source_dirty=True,
            stage="compatibility_probe", base_provenance="legacy_environment_snapshot",
            lock=lock, sbom=sbom, probe_audit=probe,
        )
    with pytest.raises(ValueError, match="schema"):
        build(tmp_path, image="pennylane")


def test_frozen_identity_requires_clean_reconstructible_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dirty source"):
        build(tmp_path, source_dirty=True)
    with pytest.raises(ValueError, match="pinned base"):
        build(tmp_path, base_provenance="legacy_environment_snapshot")
