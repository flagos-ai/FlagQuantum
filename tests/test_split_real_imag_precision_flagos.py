from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.distributed_accel]
ROOT = Path(__file__).resolve().parents[1]


def test_split_real_imag_precision_runs_on_torch_fl_flagos_device() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_SPLIT_PRECISION_FLAGOS") != "1":
        pytest.skip(
            "set FLAGQUANTUM_TEST_SPLIT_PRECISION_FLAGOS=1 in a Torch-FL environment"
        )
    completed = subprocess.run(
        [sys.executable, "tools/validate_split_real_imag_precision_flagos.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "passed"
    assert payload["device"] == "flagos:0"
    assert payload["provider"] == "torch_fl"
    assert payload["reduction_dtype"] == "double_single_fp32"
    assert payload["logical_device_residency"] is True
    assert payload["flagquantum_host_fallback"] is False
    assert payload["full_state_double_single"] is False
    assert payload["convergence_certification"] is False
    assert payload["hardware_certification"] is False
    assert payload["scalability_claim_allowed"] is False
