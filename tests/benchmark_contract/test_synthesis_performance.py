"""Actual fresh-worker workflows and rejection of misleading PF-00 receipts."""

from __future__ import annotations

import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from benchmarks.synthesis.expressions import corpus
from benchmarks.synthesis.performance import (
    CallTimeoutError,
    deadline,
    run,
    summary,
    terminate_group,
    trial,
)
from benchmarks.synthesis.performance_results import read, validate, validate_sample
from benchmarks.synthesis.performance_workloads import Prepared, Workload, workloads

pytestmark = pytest.mark.benchmark_contract


@pytest.fixture(scope="module")
def receipt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    path = tmp_path_factory.mktemp("pf00") / "trial.json"
    run(
        path,
        ["energy-2x2:qfunc:inline:logical:success"],
        samples=1,
        warmups=2,
        seconds=30,
        memory_trials=1,
        production_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=15
        ).strip(),
    )
    return read(path, formal=False)


def test_full_corpus_and_native_equivalents_are_retained() -> None:
    inventory = workloads()
    assert {
        w.case
        for w in inventory
        if w.frontend == "scalar"
        and w.variant == "inline"
        and w.target == "logical"
        and not w.refused
    } >= {c.identifier for c in corpus()}
    assert {w.frontend for w in inventory if w.case == "energy-2x2"} == {
        "scalar",
        "qfunc",
        "qmod",
    }
    assert {w.variant for w in inventory} == {"inline", "helper", "exchanged"}
    assert {w.target for w in inventory} == {"logical", "basis", "line"}
    assert len({w.identifier for w in inventory}) == len(inventory)
    with pytest.raises(ValueError):
        Workload("arbitrary_source")


@pytest.mark.parametrize("frontend", ["scalar", "qfunc", "qmod"])
@pytest.mark.parametrize("variant", ["inline", "helper", "exchanged"])
def test_real_user_workflow_has_independent_xor_phase_and_cleanup_reference(
    frontend: Literal["scalar", "qfunc", "qmod"],
    variant: Literal["inline", "helper", "exchanged"],
) -> None:
    workload = Workload("energy-2x2", frontend, variant)
    prepared = Prepared(workload)
    try:
        program = prepared.compile()
        verified = prepared.verify(program)
        assert verified["basis_checks"] == 256
        assert verified["nonzero_targets"] is True
        assert (
            isinstance(verified["e1"], dict) and verified["e1"]["status"] == "verified"
        )
    finally:
        prepared.close()


def test_actual_timing_memory_and_diagnostic_workers_are_separate(
    receipt: dict[str, object],
) -> None:
    rows = receipt["rows"]
    assert isinstance(rows, list)
    row = rows[0]
    raw = row["raw"]
    assert {s["mode"] for s in raw} == {"timing", "memory", "diagnostic"}
    assert len({s["worker_pid"] for s in raw}) == 3
    assert all(s["status"] == "verified" and s["process_group_reaped"] for s in raw)
    assert all(len(s["warmup_ns"]) == 2 for s in raw)
    diagnostic = next(s for s in raw if s["mode"] == "diagnostic")
    spans = diagnostic["diagnostic"]["spans"]
    assert {s["stage"] for s in spans} == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert {v for s in spans for k, v in s["context"] if k == "scheduling"} >= {
        "retain",
        "recompute",
    }
    assert {v for s in spans for k, v in s["context"] if k == "implementation"} >= {
        "conservative",
        "arithmetic",
    }
    assert any("plan_to_dict" in s["function"] for s in spans)
    assert any("Instruction.__init__" in s["function"] for s in spans)
    memory = next(s for s in raw if s["mode"] == "memory")
    assert "diagnostic" not in memory
    assert memory["memory"]["window_sampled_peak_rss_bytes"] > 0
    assert memory["compile_lifetime_high_water"]["bytes"] > 0
    assert (
        memory["memory"]["window_sampled_peak_rss_bytes"]
        != memory["verification"]["resources"]["planned_complex128_state_bytes"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "double_count",
        "bad_parent",
        "rss_units",
        "rss_scope",
        "missing_trial",
        "primary_profile",
        "statistics",
        "historical_hash",
        "manifest",
        "gc",
        "dirty_formal",
    ],
)
def test_reader_rejects_changed_or_overstated_measurement(
    receipt: dict[str, object], mutation: str
) -> None:
    data = copy.deepcopy(receipt)
    rows = data["rows"]
    assert isinstance(rows, list)
    row = rows[0]
    raw = row["raw"]
    diagnostic = next(s for s in raw if s["mode"] == "diagnostic")
    memory = next(s for s in raw if s["mode"] == "memory")
    if mutation == "double_count":
        diagnostic["diagnostic"]["exclusive_sum_ns"] += 1
    elif mutation == "bad_parent":
        diagnostic["diagnostic"]["spans"][0]["parent"] = 0
    elif mutation == "rss_units":
        memory["memory"]["points"][0]["raw_unit"] = "bytes"
    elif mutation == "rss_scope":
        memory["memory"]["window_sampled_peak_rss_bytes"] += 1
    elif mutation == "missing_trial":
        raw.pop()
    elif mutation == "primary_profile":
        data["configuration"]["profile_enabled_in_primary_timing"] = True
    elif mutation == "statistics":
        row["timing"]["median_ns"] = 0
    elif mutation == "historical_hash":
        data["original_resource_hashes"]["ao00.json"] = "0" * 64
    elif mutation == "manifest":
        data["source_hashes"] = {}
    elif mutation == "gc":
        raw[0]["gc_enabled"] = False
    elif mutation == "dirty_formal":
        data["source_dirty"] = True
        data["configuration"]["formal_samples"] = 5
    with pytest.raises(ValueError):
        validate(data, formal=mutation == "dirty_formal")


def test_failed_samples_are_retained_and_excluded_from_medians() -> None:
    samples = [
        {
            "mode": "timing",
            "status": "timeout",
            "worker_pid": 1,
            "process_group_reaped": True,
        },
        {"mode": "timing", "status": "verified", "e0_ns": 20},
    ]
    result = summary(samples, "timing")
    assert result == {
        "statuses": {"timeout": 1, "verified": 1},
        "sample_count": 1,
        "median_ns": 20,
        "min_ns": 20,
        "max_ns": 20,
    }
    with pytest.raises(ValueError):
        validate_sample(
            {"mode": "memory", "status": "timeout", "process_group_reaped": False}
        )


def test_kill_and_reap_isolated_worker_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        start_new_session=True,
        text=True,
    )
    try:
        assert terminate_group(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_gzip_roundtrip_keeps_raw_receipts(
    receipt: dict[str, object], tmp_path: Path
) -> None:
    target = tmp_path / "trial.json.gz"
    with gzip.open(target, "wt") as stream:
        json.dump(receipt, stream)
    assert read(target, formal=False) == receipt


def test_per_call_deadline_and_failed_worker_do_not_leave_a_success_receipt() -> None:
    import time

    with pytest.raises(CallTimeoutError), deadline(0.01):
        time.sleep(1)
    sample = trial(Workload("add-1x1"), "timing", seconds=0.000001)
    assert sample["status"] == "timeout"
    assert sample["process_group_reaped"] is True
    assert "verification" not in sample
    assert summary([sample], "timing")["sample_count"] == 0


@pytest.mark.parametrize("frontend", ["scalar", "qfunc", "qmod"])
def test_budget_refusal_is_measured_and_never_executed(
    frontend: Literal["scalar", "qfunc", "qmod"],
) -> None:
    sample = trial(Workload("energy-2x2", frontend, refused=True), "timing")
    assert sample["status"] == "resource_refused"
    assert sample["diagnostic_codes"]
    assert sample["verification"] == {"expected_refusal": True}
    assert "resources" not in sample["verification"]
    assert sample["e0_ns"] > 0
