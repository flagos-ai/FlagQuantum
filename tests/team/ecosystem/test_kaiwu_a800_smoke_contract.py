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
