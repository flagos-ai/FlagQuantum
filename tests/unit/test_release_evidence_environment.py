from pathlib import Path

import pytest

from tools.check_release_evidence_environment import readiness_errors

pytestmark = pytest.mark.unit


def test_release_evidence_preflight_accepts_complete_environment():
    assert (
        readiness_errors(
            commit="a" * 40,
            signing_key_present=True,
            device_uuids=("GPU-0", "GPU-1"),
            world_size=2,
        )
        == ()
    )


def test_release_evidence_preflight_fails_closed_without_authority_or_hardware():
    errors = readiness_errors(
        commit="",
        signing_key_present=False,
        device_uuids=("GPU-0",),
        world_size=2,
        promoted_json=(Path("candidate.json"),),
    )
    assert len(errors) == 4
    assert any("Git commit" in error for error in errors)
    assert any("FQ_EVIDENCE_SIGNING_KEY" in error for error in errors)
    assert any("GPU UUIDs" in error for error in errors)
    assert any("already contains promoted JSON" in error for error in errors)
