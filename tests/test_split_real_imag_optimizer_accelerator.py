from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.distributed_accel]
ROOT = Path(__file__).resolve().parents[1]


def test_split_real_imag_p5_optimizer_accelerator_route() -> None:
    device = os.environ.get("FLAGQUANTUM_TEST_P5_OPTIMIZER_DEVICE")
    if device is None:
        pytest.skip("set FLAGQUANTUM_TEST_P5_OPTIMIZER_DEVICE to cuda:0 or flagos:0")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/validate_split_real_imag_optimizer_accelerator.py",
            "--device",
            device,
            "--checkpoints",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "passed"
    assert payload["device"] == device
    assert payload["logical_device_residency"] is True
    assert payload["parameter_word_count"] == 2
    assert payload["tensor_grad_used"] is False
    assert payload["accelerator_float64_tensor_materialized"] is False
    assert payload["flagcx_collectives_validated"] is False
    assert payload["convergence_certification"] is False
    assert payload["hardware_certification"] is False
