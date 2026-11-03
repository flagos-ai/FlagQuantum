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


@pytest.mark.parametrize(
    "identifier,qmod_line",
    [
        ("unknown", False),
        ("energy-2x2:qfunc:inline:logical:refused", False),
        ("shared_square-8x8:qfunc:inline:logical:success", True),
    ],
)
def test_rss_review_refuses_invalid_selection_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifier: str, qmod_line: bool
) -> None:
    from benchmarks.synthesis import performance_rss_review

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid RSS selection must not launch a worker")

    monkeypatch.setattr(performance_rss_review, "trial", forbidden)
    output = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="fixed successful workload"):
        performance_rss_review.review(
            tmp_path,
            "0" * 40,
            "0" * 40,
            output,
            qmod_line=qmod_line,
            case=identifier,
        )
    assert not output.exists()


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


@pytest.mark.parametrize(
    "damage,message",
    [
        ("none", None),
        ("protocol", "protocol"),
        ("observer", "implementation"),
        ("timeout", "incomplete primary"),
        ("circuit", "selected circuit"),
        ("execution_scope", "verification scope"),
    ],
)
def test_comparison_requires_same_complete_workflow_and_measurement(
    receipt: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    message: str | None,
) -> None:
    from benchmarks.synthesis import performance_compare
    from benchmarks.synthesis.performance_bundle import objects
    from benchmarks.synthesis.results import _object

    # The reader's independent validation is covered above; exercise comparison
    # against a real measured workflow and changes that invalidate comparability.
    after = copy.deepcopy(receipt)
    row = objects(after["rows"])[0]
    timing = [s for s in objects(row["raw"]) if s["mode"] == "timing"]
    if damage == "protocol":
        _object(after["configuration"])["warmups_per_fresh_worker"] = 3
    elif damage == "observer":
        _object(after["source_hashes"])["benchmarks/synthesis/performance.py"] = (
            "0" * 64
        )
    elif damage == "timeout":
        timing[0]["status"] = "timeout"
    elif damage == "circuit":
        resources = _object(_object(timing[0]["verification"])["resources"])
        resources["circuit_hash"] = "0" * 64
    elif damage == "execution_scope":
        for sample in timing:
            _object(_object(sample["verification"])["e1"])[
                "status"
            ] = "not_run_state_budget"

    def supplied(path: Path) -> dict[str, object]:
        return receipt if path.name == "before" else after

    monkeypatch.setattr(performance_compare, "read", supplied)
    if message:
        with pytest.raises(ValueError, match=message):
            performance_compare.compare(Path("before"), Path("after"))
    else:
        result = performance_compare.compare(Path("before"), Path("after"))
        for value in _object(result["rows"]).values():
            for metric in _object(value).values():
                if metric is not None:
                    change = _object(_object(metric)["reduction"])
                    assert change["absolute"] == 0
                    assert change["meets_review_line"] is False


@pytest.mark.parametrize("frontend", ["scalar", "qfunc", "qmod"])
@pytest.mark.parametrize("target", ["logical", "basis", "line"])
@pytest.mark.parametrize("variant", ["inline", "helper", "exchanged"])
def test_real_user_workflow_has_independent_xor_phase_and_cleanup_reference(
    frontend: Literal["scalar", "qfunc", "qmod"],
    variant: Literal["inline", "helper", "exchanged"],
    target: Literal["logical", "basis", "line"],
) -> None:
    workload = Workload("energy-2x2", frontend, variant, target)
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
    phases = [p["phase"] for p in diagnostic["phase_transitions"]]
    assert (
        phases.index("compile_end")
        < phases.index("diagnostic_report")
        < phases.index("verification")
    )
    assert diagnostic["diagnostic_report_ns"] > 0
    assert all(
        p["phase"] not in {"diagnostic_report", "verification"}
        for p in diagnostic["memory"]["points"]
    )
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


def test_probe_does_not_mask_a_deadline_or_observer_exception() -> None:
    import sys

    from benchmarks.synthesis.performance_probe import Probe

    class OriginalError(RuntimeError):
        pass

    probe = Probe()
    with pytest.raises(OriginalError, match="original"), probe:
        # A failing callback can be disabled by CPython before unwind events.
        probe.stack.append(object())
        raise OriginalError("original")
    assert sys.getprofile() is None
    assert not probe.stack


@pytest.mark.parametrize(
    "field", ["source_revision", "production_tree", "production_baseline_revision"]
)
def test_pinned_source_or_production_cannot_be_relabelled(
    receipt: dict[str, object], field: str
) -> None:
    from benchmarks.synthesis.performance_results import verify_sources

    altered = copy.deepcopy(receipt)
    altered["source_dirty"] = False
    altered[field] = "0" * 40
    with pytest.raises(ValueError):
        verify_sources(altered, formal=True)


def test_frozen_bundle_keeps_failures_and_exact_cohort_selection() -> None:
    from collections import Counter

    from benchmarks.synthesis.performance_bundle import sha256

    base = Path(__file__).resolve().parents[2] / "benchmarks/synthesis/baselines"
    manifest = json.loads((base / "pf00-manifest.json").read_text())
    overview = json.loads((base / "pf00-summary.json").read_text())
    assert manifest["coverage"]["unique_configurations"] == 125
    assert manifest["coverage"]["attempted_configurations"] == 128
    assert manifest["coverage"]["complete_primary_configurations"] == 124
    expected = {
        f"energy-2x2:{name}:inline:line:success" for name in ("scalar", "qfunc", "qmod")
    }
    assert {
        key
        for key, value in manifest["selection"].items()
        if value == "line_correction"
    } == expected
    for cohort in manifest["cohorts"].values():
        assert sha256(base / cohort["file"]) == cohort["stored_sha256"]
    assert manifest["cohorts"]["initial"]["raw_statuses"]["validation_failed"] == 21
    assert manifest["cohorts"]["line_correction"]["raw_statuses"] == {"verified": 21}
    assert len(overview["rows"]) == 125
    primary = Counter()
    for row in overview["rows"]:
        primary.update(row["timing"]["statuses"])
        assert row["cohort"] == manifest["selection"][row["id"]]
    assert dict(primary) == manifest["coverage"]["selected_primary_statuses"]
    assert dict(primary) == {"verified": 605, "resource_refused": 15, "timeout": 5}


def test_archive_parts_reassemble_exact_bytes_and_reject_tampering(
    tmp_path: Path,
) -> None:
    from benchmarks.synthesis.performance_bundle import store_archive
    from benchmarks.synthesis.performance_results import archive_bytes

    original = bytes(range(256)) * 8000
    path = tmp_path / "record.gz"
    path.write_bytes(original)
    index = store_archive(path)
    assert not path.exists()
    assert archive_bytes(index) == original
    parts = json.loads(index.read_text())["parts"]
    assert all((tmp_path / part["file"]).stat().st_size <= 1_800_000 for part in parts)
    first = tmp_path / parts[0]["file"]
    raw = first.read_bytes()
    first.write_bytes(b"X" + raw[1:])
    with pytest.raises(ValueError, match="checksum"):
        archive_bytes(index)
