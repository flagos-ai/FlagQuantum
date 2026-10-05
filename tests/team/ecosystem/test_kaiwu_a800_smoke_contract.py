from __future__ import annotations

from pathlib import Path

import pytest

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
