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
    assert "flagquantum/flagtree:0.7.0-validation" in source
    assert "qdiffusion_system_development_probe.py" in source


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
    assert '"fallback_occurred": False' in source
    assert "resolve_kaiwu_credentials" in source
    assert "_write_private_redacted_json" in source
