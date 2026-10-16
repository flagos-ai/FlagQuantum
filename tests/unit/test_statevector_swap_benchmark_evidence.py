"""Contracts for reproducible SV-014 development evidence."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal.evidence.statevector_swap_probe import (
    BENCHMARK,
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    aggregate_runs,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _ROOT / "benchmarks" / "results" / "local" / "statevector_swap_sequence_a800.json"
)
_EVIDENCE_REVISION = "8751f367767c57b795c7e63ce72d199b66250921"


def _shape_record(
    batch: int,
    amplitudes: int,
    swaps: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {
        "batch": batch,
        "amplitudes_per_batch": amplitudes,
        "swaps": [list(pair) for pair in swaps],
    }


def _timing(seconds: float) -> dict[str, Any]:
    samples = [seconds] * 30
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": seconds,
    }


def _run(host: str, compiler_lane: str, *, revision: str = "a" * 40) -> dict[str, Any]:
    distribution, integration_path = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[compiler_lane]
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "source_revision": revision,
        "host_label": host,
        "compiler_lane": compiler_lane,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": distribution,
            "integration_path": integration_path,
            "identity_status": "resolved",
        },
        "measurement": {
            "warmup": 10,
            "repeats": 30,
            "group_size": 10,
            "ordering": "alternating direct-first and reference-first groups",
            "synchronization": "once after warmup and once per timed group",
            "statistic": "median synchronized wall seconds per invocation",
        },
        "cases": [
            {
                "shape": _shape_record(*shape),
                "direct_kernel_wrapper": _timing(1.0),
                "pytorch_swap_sequence_reference": _timing(1.5),
                "speedup_over_pytorch": 1.5,
                "maximum_absolute_error": 0.0,
                "relative_l2_error": 0.0,
            }
            for shape in SHAPE_MATRIX
        ],
    }


def _write_matrix(tmp_path: Path) -> list[Path]:
    paths = []
    for host in HOSTS:
        for compiler_lane in COMPILER_LANES:
            path = tmp_path / f"{host}-{compiler_lane}.json"
            path.write_text(
                json.dumps(_run(host, compiler_lane)),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def test_sv014_aggregate_requires_profitable_complete_matrix(tmp_path: Path) -> None:
    payload = aggregate_runs(_write_matrix(tmp_path))

    assert payload["benchmark"] == BENCHMARK
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    aggregate = payload["aggregate"]
    assert aggregate["case_count"] == 20
    assert aggregate["minimum_speedup_over_pytorch"] == 1.5
    assert aggregate["all_cases_win"] is True
    assert aggregate["decision"] == "eligible_for_dispatch_evaluation"


def test_sv014_aggregate_rejects_incomplete_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly four raw runs"):
        aggregate_runs(_write_matrix(tmp_path)[:-1])


def test_checked_in_sv014_evidence_is_exact_and_profitable() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["benchmark"] == BENCHMARK
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["source_revision"] == _EVIDENCE_REVISION
    assert payload["shape_matrix"] == [_shape_record(*shape) for shape in SHAPE_MATRIX]
    assert {(run["host_label"], run["compiler_lane"]) for run in payload["runs"]} == {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }
    aggregate = payload["aggregate"]
    assert aggregate["case_count"] == 20
    assert aggregate["minimum_speedup_over_pytorch"] > 1.29
    assert aggregate["maximum_speedup_over_pytorch"] > 2.51
    assert aggregate["maximum_absolute_error"] == 0.0
    assert aggregate["maximum_relative_l2_error"] == 0.0
    assert aggregate["all_cases_win"] is True
    assert aggregate["decision"] == "eligible_for_dispatch_evaluation"


def test_checked_in_sv014_samples_reproduce_claims() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    for run in payload["runs"]:
        assert run["source_revision"] == payload["source_revision"]
        assert run["compiler"]["identity_status"] == "resolved"
        assert run["environment"]["gpu"] == "NVIDIA A800-SXM4-80GB"
        assert run["measurement"] == payload["measurement"]
        repeats = int(run["measurement"]["repeats"])
        for case in run["cases"]:
            for result_name in RESULT_NAMES:
                result = case[result_name]
                samples = result["samples_seconds_per_invocation"]
                assert len(samples) == repeats
                assert math.isclose(
                    result["median_seconds_per_invocation"],
                    statistics.median(samples),
                )
            assert math.isclose(
                case["speedup_over_pytorch"],
                case["pytorch_swap_sequence_reference"]["median_seconds_per_invocation"]
                / case["direct_kernel_wrapper"]["median_seconds_per_invocation"],
            )
