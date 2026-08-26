from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_accel, pytest.mark.gpu]


@pytest.mark.skipif(
    os.getenv("FLAGQUANTUM_TEST_FLAGOS_SCALE") != "1",
    reason="requires an explicit eight-device Torch-FL FlagOS environment",
)
def test_flagos_statevector_2_4_8_scale_ladder(tmp_path: Path):
    output = tmp_path / "flagos-statevector-scale.json"
    completed = subprocess.run(
        (
            sys.executable,
            "tools/validate_flagos_statevector_scale.py",
            "--output",
            str(output),
            "--timeout-seconds",
            "300",
        ),
        check=False,
        timeout=2700,
    )

    assert completed.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "flagquantum_flagos_statevector_scale_profile_v1"
    assert payload["world_sizes"] == [2, 4, 8]
    assert payload["scale_ladder_accepted"] is True
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["flagcx_route_verified"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
