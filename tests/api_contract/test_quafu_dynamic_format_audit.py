"""Contract checks for API Change 093's maintained-source audit."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/development/API_CHANGE_093_QUAFU_DYNAMIC_FORMAT_AUDIT.md"
PROPOSAL = (
    ROOT
    / "docs/development/API_CHANGE_PROPOSAL_093_QUAFU_DYNAMIC_FORMAT_ACTIVATION.md"
)


def test_audit_is_revision_pinned_and_fail_closed() -> None:
    text = AUDIT.read_text(encoding="utf-8")

    assert "QUAFU-Quantum/quafu-sdk" in text
    assert "b54c6e5dadde81e0bba8529a4e03523794fd97c7" in text
    assert "package version: `0.2.7`" in text
    assert "**Outcome:** Unsupported by the current maintained SDK contract" in text
    assert "**Stable execution state:** Fail closed" in text
    assert 'fq.run(circuit, target="quafu:Baihua", shots=1024)' in text
    assert "not a permanent statement\nabout Quafu hardware" in text


def test_proposal_records_stage_a_without_authorizing_live_work() -> None:
    text = PROPOSAL.read_text(encoding="utf-8")

    assert "**Status:** Stage A approved and complete" in text
    assert "API_CHANGE_093_QUAFU_DYNAMIC_FORMAT_AUDIT.md" in text
    assert "Stable execution therefore\nremains fail closed" in text
    assert "It does not authorize credentials" in text
