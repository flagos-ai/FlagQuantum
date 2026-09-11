"""Scheduled hardware integration for the FlagOS F6 transport ladder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.distributed,
    pytest.mark.distributed_accel,
    pytest.mark.gpu,
]

ROOT = Path(__file__).resolve().parents[1]


def test_flagos_transport_observability_ladder(tmp_path: Path) -> None:
    if os.environ.get("FLAGQUANTUM_TEST_FLAGOS_TRANSPORT") != "1":
        pytest.skip(
            "set FLAGQUANTUM_TEST_FLAGOS_TRANSPORT=1 in an eight-device "
            "Torch-FL environment"
        )

    output = tmp_path / "flagos-transport-observability.json"
    environment = dict(os.environ)
    environment.setdefault("FLAGQUANTUM_SOURCE_REVISION", "pytest-worktree")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/observe_flagos_transport.py",
            "--output",
            str(output),
            "--timeout-seconds",
            "300",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "passed"
    assert payload["transport_observation_accepted"] is True
    assert payload["world_sizes"] == [2, 4, 8]
    assert payload["outer_backend"] == "flagos"
    assert payload["inner_communication_route"] == "unattributed"
    assert payload["flagcx_route_verified"] is False
    assert payload["no_host_staging_certified"] is False
    assert payload["communication_claim_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert all(
        observation["profiler_available"]
        and {"CPU", "CUDA"} <= set(observation["profiler_activities"])
        and observation["profiler_events"]
        and observation["input_device_type"] == "flagos"
        and observation["output_device_type"] == "flagos"
        for run in payload["runs"]
        for rank in run["ranks"]
        for observation in rank["observations"]
    )
