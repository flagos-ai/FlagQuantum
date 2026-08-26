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


def test_flagos_two_rank_process_group_and_sharded_statevector(tmp_path: Path) -> None:
    if os.environ.get("FLAGQUANTUM_TEST_FLAGOS_DISTRIBUTED") != "1":
        pytest.skip(
            "set FLAGQUANTUM_TEST_FLAGOS_DISTRIBUTED=1 in a two-device Torch-FL environment"
        )

    output = tmp_path / "flagos-distributed-conformance.json"
    environment = dict(os.environ)
    environment.setdefault("FLAGQUANTUM_SOURCE_REVISION", "pytest-worktree")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "tools/validate_flagos_distributed_conformance.py",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert payload["schema"] == "flagquantum_flagos_distributed_conformance_v1"
    assert payload["status"] == "passed"
    assert payload["outer_backend"] == "flagos"
    assert payload["world_size"] == 2
    assert payload["node_count"] == 1
    assert len(payload["collective_checks"]) == 10
    assert len(payload["statevector_checks"]) == 2
    assert all(item["device_type"] == "flagos" for item in payload["collective_checks"])
    assert all(
        item["distribution_semantics"] == "sharded_across_ranks"
        and not item["full_state_materialization"]
        for item in payload["statevector_checks"]
    )
    assert payload["mechanical_conformance_accepted"] is True
    assert payload["statevector_workload_conformance_accepted"] is True
    assert payload["flagcx_route_verified"] is False
    assert payload["communication_claim_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
