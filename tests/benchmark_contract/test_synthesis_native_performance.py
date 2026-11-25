"""Native performance receipts retain full-state correctness and phase boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.native_performance import run
from benchmarks.synthesis.native_workloads import NativeWorkload
from benchmarks.synthesis.performance import terminate_group

pytestmark = pytest.mark.benchmark_contract


def test_completed_worker_can_finish_teardown_before_group_cleanup(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "closed"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib,sys,time;time.sleep(0.3);pathlib.Path(sys.argv[1]).write_text('closed')",
            str(marker),
        ],
        start_new_session=True,
        text=True,
    )
    try:
        assert terminate_group(process, grace_seconds=2.0)
        assert process.returncode == 0
        assert marker.read_text() == "closed"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("name", ["arithmetic", "qaoa", "pauli", "block_encoding"])
def test_fixed_workflows_match_complete_independent_states(name: str) -> None:
    workload = NativeWorkload(name)
    program = workload.build()
    actual = fq.run(program).statevector().flatten()
    expected = workload.expected(program)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.abs().square().sum()) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("selected", [[], ["unknown"], ["qaoa", "qaoa"]])
def test_invalid_selection_creates_no_receipt(
    tmp_path: Path, selected: list[str]
) -> None:
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="distinct fixed"):
        run(path, selected)
    assert not path.exists()


def test_worker_receipt_separates_compile_execute_and_reference(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    run(path, ["arithmetic"], samples=1, memory_trials=1, warmups=1)
    receipt = json.loads(path.read_text())
    assert receipt["complete"] and not receipt["source_changes"]
    rows = receipt["rows"]["arithmetic"]["samples"]
    assert [row["mode"] for row in rows] == ["timing", "memory", "diagnostic"]
    assert len({row["circuit_hash"] for row in rows}) == 1
    for row in rows:
        assert row["status"] == "verified"
        assert row["process_group_reaped"]
        assert row["e0_ns"] > 0 and row["execution_ns"] > 0
        phases = [item["phase"] for item in row["phase_transitions"]]
        assert phases.index("compile_end") < phases.index("execution")
        assert phases.index("execution_end") < phases.index("verification")
        assert row["verification"]["maximum_error"] < 1e-12
        assert not row["verification"]["phase_alignment"]
    timing, memory, diagnostic = rows
    assert "memory" not in timing and "diagnostic" not in timing
    assert memory["memory"]["window_sampled_peak_rss_bytes"] > 0
    assert memory["execution_memory"]["window_sampled_peak_rss_bytes"] > 0
    assert all(point["phase"] != "verification" for point in memory["memory"]["points"])
    spans = diagnostic["diagnostic"]
    assert spans["exclusive_sum_ns"] + spans["unattributed_ns"] == diagnostic["e0_ns"]
