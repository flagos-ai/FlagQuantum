from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"
REPORT = RESULTS / "LINUX_X86_CPU_ROADMAP_CLOSEOUT_20261005.md"
RAW = RESULTS / "linux_x86_cpu_roadmap_closeout_main_20261005.json"
STAGE_REPORTS = (
    "LINUX_X86_CPU_SCORECARD_20261003.md",
    "SOCKET_LOCAL_MULTITASK_THROUGHPUT_LINUX_X86_20261003.md",
    "NUMA_MEMORY_TRAFFIC_LINUX_X86_20261004.md",
    "NATIVE_CPU_PRODUCT_MIXED_CLIFFORD_LINUX_X86_20261004.md",
    "LINUX_X86_CPU_REGRESSION_GATE_20261004.md",
)


def _load() -> dict[str, Any]:
    return json.loads(RAW.read_text(encoding="utf-8"))


def test_linux_x86_closeout_links_every_stage_and_raw_result() -> None:
    document = REPORT.read_text(encoding="utf-8")

    for name in (*STAGE_REPORTS, RAW.name):
        assert (RESULTS / name).is_file()
        assert f"]({name})" in document


def test_linux_x86_closeout_headlines_match_post_merge_measurement() -> None:
    payload = _load()
    document = REPORT.read_text(encoding="utf-8")
    native = payload["engines"]["flagquantum_native"]

    assert payload["benchmark_evidence_class"] == "comparison_non_release"
    assert payload["claim_evidence_type"] == "development_smoke"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["non_release_evidence"] is True
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["scalability_blockers"]
    assert payload["source_revision"] == "52314f6932d040bf3dbe78383b374fa1a144694a"
    assert payload["environment"]["machine"] == "x86_64"
    assert payload["environment"]["threads"] == 32
    assert payload["workload"]["n_wires"] == 22
    assert payload["workload"]["dtype"] == "complex128"
    assert len(native["samples_seconds"]) == 7

    for engine in payload["engines"].values():
        assert len(engine["samples_seconds"]) == 7
        assert engine["relative_median_absolute_deviation"] <= 0.20
        assert engine["max_abs_error"] <= 1e-10
        milliseconds = engine["median_seconds"] * 1000
        assert f"{milliseconds:.3f} ms" in document

    for name in ("qiskit_aer", "pennylane_lightning_qubit"):
        ratio = payload["engines"][name]["engine_over_flagquantum_median"]
        assert ratio > 1.0
        assert f"**{ratio:.3f}x**" in document

    gates = payload["gates"]
    assert gates["current_over_internal_baseline"] <= gates["maximum_internal_slowdown"]
    assert gates["qiskit_aer"] == "pass"
    assert gates["pennylane_lightning_qubit"] == "pass"
    assert gates["correctness"] == "pass"
    assert gates["stability"] == "pass"


def test_linux_x86_closeout_preserves_boundaries_and_restart_contract() -> None:
    document = REPORT.read_text(encoding="utf-8")

    assert "not a claim that FlagQuantum\nwins every circuit" in document
    assert "not distributed, noisy, approximate" in document
    assert "contains workloads where Aer or\n  Lightning had lower" in document
    assert (
        "Socket-local throughput and single-circuit latency are different" in document
    )
    assert "Reopen this tranche" in document
