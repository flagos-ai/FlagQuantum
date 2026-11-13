"""The native sampler adds exactly two unchanged upstream template inputs."""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_dem_source_closure_preserves_upstream_identity() -> None:
    root = Path(__file__).parents[2] / "flagquantum/simulation/stabilizer/csrc"
    lock = json.loads((root / "stim-sampler-source-lock.json").read_text())
    assert len(lock["file_sha256"]) == 98
    assert len(lock["translation_units"]) == 37
    expected = {
        "dem_sampler.h": "31e8ce743db8e0b787b7e908c49d3f323ce7fea3647e95994e97a23363f2bb32",
        "dem_sampler.inl": "330ebeb41ee90cf78737dce00fc2bf9f8e755686cb447128ac3c9aa3db3f039a",
    }
    for name, digest in expected.items():
        relative = "src/stim/simulators/" + name
        assert relative in lock["headers_and_templates"]
        assert lock["file_sha256"][relative] == digest
    for relative, digest in lock["file_sha256"].items():
        assert (
            hashlib.sha256(
                (root / "vendor/stim-1.16.0" / relative).read_bytes()
            ).hexdigest()
            == digest
        )
