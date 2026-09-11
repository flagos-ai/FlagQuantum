from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_split_real_imag_optimizer_evidence import evidence_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/split_real_imag_optimizer_a800_20260825.json"


def _payload() -> dict[str, object]:
    value: dict[str, object] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return value


def test_checked_in_split_p5_optimizer_a800_evidence_is_valid() -> None:
    assert evidence_errors(_payload()) == ()


def test_split_p5_optimizer_evidence_rejects_complex128_promotion() -> None:
    payload = _payload()
    claims = payload["claims"]
    assert isinstance(claims, dict)
    claims["complex128_equivalence"] = True
    assert "forbidden promotion" in " ".join(evidence_errors(payload))


def test_split_p5_optimizer_evidence_rejects_tensor_grad_route() -> None:
    payload = _payload()
    routes = payload["routes"]
    assert isinstance(routes, list)
    routes[0]["tensor_grad_used"] = True
    assert "native CUDA route" in " ".join(evidence_errors(payload))


def test_split_p5_optimizer_evidence_rejects_fp32_regression() -> None:
    payload = _payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases[0]["double_single_parameter_absolute_error"] = 1.0
    assert "evidence envelope" in " ".join(evidence_errors(payload))
