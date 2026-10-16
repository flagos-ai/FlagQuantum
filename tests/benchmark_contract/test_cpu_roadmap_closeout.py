from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "comparison"
CLOSEOUT = RESULTS / "CPU_ROADMAP_CLOSEOUT_CPU_ARM64_20261003.md"

ARTIFACTS = (
    "CPU_PHASE1_SCORECARD_CPU_ARM64_20260930.md",
    "cpu_phase1_forward_cpu_arm64_20260930.json",
    "cpu_phase1_forward_gate_cpu_arm64_20260930.json",
    "cpu_phase1_adjoint_cpu_arm64_20260930.json",
    "cpu_phase1_adjoint_gate_cpu_arm64_20260930.json",
    "NATIVE_CPU_ADJOINT_THREAD_SCALING_CPU_ARM64_20260928.md",
    "NATIVE_CPU_ADJOINT_MEMORY_TIERS_CPU_ARM64_20260930.md",
    "native_cpu_adjoint_memory_tiers_cpu_arm64_20260930.json",
    "BATCHED_STATEVECTOR_CPU_PHASE_CLOSEOUT_CPU_ARM64_20261003_SCORECARD.md",
    "batched_statevector_cpu_phase_closeout_cpu_arm64_20261003.json",
    "BATCHED_STATEVECTOR_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md",
    "batched_statevector_adaptive_memory_cpu_arm64_20261003.json",
    "BATCHED_STATEVECTOR_QFT_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md",
    "batched_statevector_qft_adaptive_memory_cpu_arm64_20261003.json",
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_cpu_closeout_links_every_authoritative_artifact() -> None:
    document = CLOSEOUT.read_text(encoding="utf-8")

    for name in ARTIFACTS:
        assert (RESULTS / name).is_file()
        assert f"]({name})" in document


def test_cpu_closeout_headlines_match_measured_artifacts() -> None:
    document = CLOSEOUT.read_text(encoding="utf-8")
    forward = _load("cpu_phase1_forward_cpu_arm64_20260930.json")
    forward_gate = _load("cpu_phase1_forward_gate_cpu_arm64_20260930.json")
    adjoint_gate = _load("cpu_phase1_adjoint_gate_cpu_arm64_20260930.json")
    batched = _load("batched_statevector_cpu_phase_closeout_cpu_arm64_20261003.json")

    forward_ratios = [
        float(ratio)
        for case in forward["cases"]
        for ratio in case["comparison"]["engine_over_flagquantum_median"].values()
    ]
    batch_ratios = [
        float(
            case["comparison"]["engine_over_flagquantum_batch_median"][
                "pennylane_lightning_native_batch"
            ]
        )
        for case in batched["cases"]
    ]

    assert len(forward["cases"]) == 20
    assert len(forward_ratios) == 60
    assert min(forward_ratios) == pytest.approx(1.3789666588112375)
    assert max(forward_ratios) == pytest.approx(71.11685914947898)
    assert "**1.38x–71.12x**" in document

    assert forward_gate["verdict"] == "pass"
    assert adjoint_gate["verdict"] == "pass"
    assert not forward_gate["failures"]
    assert not adjoint_gate["failures"]

    assert len(batched["cases"]) == 5
    assert min(batch_ratios) == pytest.approx(1.3227519572315896)
    assert max(batch_ratios) == pytest.approx(2.5710914059396517)
    assert "**1.323x–2.571x**" in document


def test_cpu_closeout_keeps_claim_boundaries_and_restart_contract() -> None:
    document = CLOSEOUT.read_text(encoding="utf-8")

    assert "not a universal framework" in document
    assert "ranking, release claim" in document
    assert "Neither proves arbitrary 22- or 26-qubit capacity" in document
    assert "Thread scaling is sublinear" in document
    assert "Future CPU work must identify its own workload" in document
