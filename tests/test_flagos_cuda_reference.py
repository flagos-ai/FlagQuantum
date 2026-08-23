from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def test_flagquantum_runs_on_torch_fl_cuda_backed_flagos_device() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_FLAGOS_CUDA") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_FLAGOS_CUDA=1 in a Torch-FL CUDA environment")

    completed = subprocess.run(
        [sys.executable, "tools/validate_flagos_cuda_reference.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["status"] == "passed"
    assert payload["validation_scope"] == "flagos_cuda_reference"
    assert payload["device"] == "flagos:0"
    assert payload["operator_profile"] == "statevector_local_p0"
    assert payload["operator_evidence_count"] == 21
    assert payload["validated_dtypes"] == ["complex64", "complex128"]
    assert payload["validated_depths"] == [8, 32, 128]
    assert len(payload["validation_matrix"]) == 6
    assert all(
        item["numerical_validation"]["passed"] for item in payload["validation_matrix"]
    )
    assert payload["hardware_certification"] is False
