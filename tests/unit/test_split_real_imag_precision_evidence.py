from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_split_real_imag_precision_evidence import evidence_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/split_real_imag_precision_a800_20260824.json"


def _payload() -> dict[str, object]:
    value: dict[str, object] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return value


def test_checked_in_split_p2_a800_evidence_is_valid() -> None:
    assert evidence_errors(_payload()) == ()


def test_split_p2_a800_evidence_rejects_claim_promotion() -> None:
    payload = _payload()
    claims = payload["claims"]
    assert isinstance(claims, dict)
    claims["domestic_accelerator_certification"] = True
    assert "forbidden promotion claim" in " ".join(evidence_errors(payload))


def test_split_p2_a800_evidence_rejects_incomplete_flagos_route() -> None:
    payload = _payload()
    routes = payload["routes"]
    assert isinstance(routes, list)
    flagos = next(route for route in routes if route["name"] == "torch_fl_flagos")
    flagos["flagquantum_host_fallback"] = True
    assert "flagos route is incomplete" in " ".join(evidence_errors(payload)).lower()
