"""Contract checks for the frozen FlagOS F5 capacity manifest."""

from pathlib import Path

import pytest

from tools.validate_flagos_statevector_capacity import _canonical_sha256, _manifest

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/manifests/flagos_statevector_capacity_f5.json"


def test_f5_manifest_is_development_only_and_frozen_to_eight_card_completion():
    payload = _manifest(MANIFEST)
    assert payload["release_eligible"] is False
    assert payload["artifact_classification"] == "development_calibrated_threshold"
    assert payload["acceptance"]["completion_world_size"] == 8
    assert payload["workload"]["n_wires"] == 32
    assert payload["workload"]["dtype"] == "complex128"
    assert len(_canonical_sha256(payload["workload"])) == 64
