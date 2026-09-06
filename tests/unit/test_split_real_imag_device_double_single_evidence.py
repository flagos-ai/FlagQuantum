from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_split_real_imag_device_double_single_evidence import (
    evidence_errors,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/split_real_imag_device_double_single_a800_20260825.json"


def _payload() -> dict[str, object]:
    value: dict[str, object] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return value


def test_checked_in_split_p4_a800_evidence_is_valid() -> None:
    assert evidence_errors(_payload()) == ()


def test_split_p4_a800_evidence_rejects_claim_promotion() -> None:
    payload = _payload()
    claims = payload["claims"]
    assert isinstance(claims, dict)
    claims["convergence_certification"] = True
    assert "forbidden claim promotion" in " ".join(evidence_errors(payload))


def test_split_p4_a800_evidence_keeps_historical_profile_identity() -> None:
    payload = _payload()
    operator_profile = payload["operator_profile"]
    assert isinstance(operator_profile, dict)
    operator_profile["sha256"] = "0" * 64
    assert "historical operator profile identity drifted" in " ".join(
        evidence_errors(payload)
    )


def test_split_p4_a800_evidence_rejects_host_gate_encoding() -> None:
    payload = _payload()
    routes = payload["routes"]
    assert isinstance(routes, list)
    flagos = next(route for route in routes if route["name"] == "torch_fl_flagos")
    flagos["host_gate_encoding"] = True
    assert "flagos route is incomplete" in " ".join(evidence_errors(payload)).lower()


def test_split_p4_a800_evidence_does_not_claim_flagcx_validation() -> None:
    payload = _payload()
    claims = payload["claims"]
    assert isinstance(claims, dict)
    claims["flagcx_collectives_validated"] = True
    assert "forbidden claim promotion" in " ".join(evidence_errors(payload))
