from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]
ROOT = Path(__file__).resolve().parents[1]


def test_domestic_single_card_p0_p5_route() -> None:
    attestation = os.environ.get("FLAGQUANTUM_DOMESTIC_ATTESTATION")
    if attestation is None:
        pytest.skip("set FLAGQUANTUM_DOMESTIC_ATTESTATION on an attested FlagOS host")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/validate_domestic_single_card.py",
            "--attestation",
            attestation,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["device"] == "flagos:0"
    assert set(payload["phases"]) == {
        "p0_forward",
        "p1_expectation_gradient",
        "p2_selective_double_single",
        "p3_full_double_single",
        "p4_device_double_single",
        "p5_double_single_sgd",
    }
    claims = payload["claims"]
    assert claims["domestic_accelerator_certification_candidate"] is True
    assert claims["hardware_certification"] is False
    assert claims["flagcx_collectives_validated"] is False
    assert claims["scalability_claim_allowed"] is False
    assert claims["production_claim_allowed"] is False
