from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.distributed_accel,
]
ROOT = Path(__file__).resolve().parents[1]


def test_split_real_imag_runs_on_torch_fl_flagos_device() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_SPLIT_FLAGOS") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_SPLIT_FLAGOS=1 in a Torch-FL environment")

    completed = subprocess.run(
        [sys.executable, "tools/validate_split_real_imag_flagos.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "passed"
    assert payload["device"] == "flagos:0"
    assert payload["storage_dtype"] == "float32"
    assert payload["logical_device_residency"] is True
    assert payload["flagquantum_host_fallback"] is False
    assert payload["provider_internal_route_audited"] is False
    assert payload["hardware_certification"] is False
