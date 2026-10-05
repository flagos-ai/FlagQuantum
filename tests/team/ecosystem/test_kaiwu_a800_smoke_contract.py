from __future__ import annotations

from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.a800_sampler_smoke import _write_private_json

pytestmark = pytest.mark.unit


def test_a800_smoke_source_cannot_claim_real_provider_acceptance() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "a800_sampler_smoke.py"
    ).read_text(encoding="utf-8")

    assert '"evidence_class": "development_fake_transport"' in source
    assert '"system_acceptance": False' in source
    assert '"qboson_hardware_used": False' in source
    assert '"real_provider_evidence": False' in source
    assert '"transport": "in_memory_fake"' in source


def test_a800_smoke_record_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"

    _write_private_json(path, {"evidence_class": "development_fake_transport"})

    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(path, {"evidence_class": "development_fake_transport"})


def test_a800_container_runner_keeps_execution_bounded() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "run_a800_development_probe.sh"
    ).read_text(encoding="utf-8")

    assert "--gpus device=0" in source
    assert "--network none" in source
    assert "--read-only" in source
    assert '--hostname "$expected_hostname"' in source
    assert ":/workspace/flagquantum:ro" in source
    assert ":/workspace/kaiwu-plugin:ro" in source
    assert ':/source-preflight.json:ro"' in source
    assert "--source-preflight /source-preflight.json" in source
    assert "--plugin-root /workspace/kaiwu-plugin" in source
    assert "VALIDATION_IMAGE_ID" in source
    assert "docker image inspect" in source
    assert '"$validation_image_id"' in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "-m examples.qdiffusion_kaiwu.qdiffusion_system_development_probe" in source


def test_qdiffusion_development_source_cannot_claim_acceptance() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_system_development_probe.py"
    ).read_text(encoding="utf-8")

    assert '"evidence_class": "development_fake_transport"' in source
    assert '"system_acceptance": False' in source
    assert '"qboson_hardware_used": False' in source
    assert '"real_provider_evidence": False' in source
    assert '"transport": "in_memory_fake"' in source
    assert '"validation_image_id": validation_image_id' in source
    assert '"source_preflight_sha256": source_preflight_sha256' in source
    assert '"transfer_manifest_sha256": transfer_manifest_sha256' in source


def test_qdiffusion_live_source_requires_cost_and_provider_identity() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_system_live.py"
    ).read_text(encoding="utf-8")

    assert "I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE" in source
    assert "provider_identity_complete" in source
    assert "retrieval_resubmitted is False" in source
    assert '"returned_samples"' in source
    assert '"fallback_occurred": False' in source
    assert "resolve_kaiwu_credentials" in source
    assert "_write_private_redacted_json" in source


def test_local_golden_path_is_pinned_and_credential_free() -> None:
    path = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "run_local_conformance.sh"
    )
    source = path.read_text(encoding="utf-8")

    assert path.stat().st_mode & 0o111
    assert "b648b531c034bd6ae9b7a34fed994c717967cc72" in source
    assert "f047bce7b1077449967bbe9e9fab5741542b48d4" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "unset QBOSON_USER_ID QBOSON_SDK_CODE QBOSON_PROJECT_NO" in source
    assert "PYTHONNOUSERSITE=1" in source
    assert (
        'PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/flagquantum-kaiwu-pycache-$$"' in source
    )
    assert '"$PYTHON_BIN" -B -m pytest' in source
    assert "test_kaiwu_community_conformance.py" in source
    assert "test_kaiwu_pytorch_plugin_conformance.py" in source
    assert "test_qdiffusion_environment_lock.py" in source
    assert "qboson_live_smoke.py" not in source
