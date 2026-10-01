"""Contracts for reproducible MPS-004 benchmark evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path

import pytest

from benchmarks.mps_environment_dispatch import (
    COMPILER_LANES,
    IMPLEMENTATION_ID,
    RESULT_NAMES,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "mps_environment_dispatch_a800.json"
)


def _run(
    host: str,
    lane: str,
    *,
    eager_speedup: float = 0.9,
    compiled_speedup: float = 0.8,
) -> dict[str, object]:
    samples = [1e-4, 2e-4, 3e-4]
    cases = []
    for batch, left_bond, right_bond, insert_z in SHAPE_MATRIX:
        result = {
            "samples_seconds_per_invocation": samples,
            "median_seconds_per_invocation": statistics.median(samples),
            "peak_memory_bytes": 1024,
            "peak_memory_delta_bytes": 256,
        }
        public_dispatch = copy.deepcopy(result)
        public_eager = copy.deepcopy(result)
        public_eager["samples_seconds_per_invocation"] = [
            sample * eager_speedup for sample in samples
        ]
        public_eager["median_seconds_per_invocation"] = statistics.median(
            public_eager["samples_seconds_per_invocation"]
        )
        public_compiled = copy.deepcopy(result)
        public_compiled["samples_seconds_per_invocation"] = [
            sample * compiled_speedup for sample in samples
        ]
        public_compiled["median_seconds_per_invocation"] = statistics.median(
            public_compiled["samples_seconds_per_invocation"]
        )
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "left_bond": left_bond,
                    "physical_dimension": 2,
                    "right_bond": right_bond,
                },
                "insert_z": insert_z,
                "dtype": "complex64",
                "layout": "contiguous_mps_environment_and_site_tensor",
                "maximum_absolute_error": 1e-5,
                "relative_l2_error": 1e-5,
                "direct_kernel_wrapper": copy.deepcopy(result),
                "direct_pytorch_einsum": copy.deepcopy(result),
                "public_catalog_dispatch": public_dispatch,
                "public_pytorch_eager": public_eager,
                "public_pytorch_compiled": public_compiled,
                "direct_kernel_speedup_over_pytorch": 1.0,
                "public_dispatch_speedup_over_eager": eager_speedup,
                "public_dispatch_speedup_over_compiled": compiled_speedup,
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": "0123456789abcdef",
        "runner": RUNNER,
        "command": f"python {RUNNER} run",
        "host_label": host,
        "reported_hostname": host,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": lane,
        "compiler": {
            "distribution": distribution,
            "version": "3.0.0",
            "integration_path": "direct" if lane == "stock_triton" else "flagtree",
            "identity_source": "python_package_metadata",
            "identity_status": "resolved",
        },
        "environment": {
            "gpu": "NVIDIA A800-SXM4-80GB",
            "gpu_total_memory_bytes": 85_000_000_000,
            "cuda_runtime": "12.8",
            "pytorch": "2.7.1",
            "python": "3.12.11",
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": 2,
            "repeats": len(samples),
            "group_size": 4,
            "seed": 261002,
        },
        "cases": cases,
    }


def _matrix(
    *,
    eager_speedup: float = 0.9,
    compiled_speedup: float = 0.8,
) -> list[dict[str, object]]:
    return [
        _run(
            host,
            lane,
            eager_speedup=eager_speedup,
            compiled_speedup=compiled_speedup,
        )
        for host in ("jp-a800-171", "jp-a800-172")
        for lane in COMPILER_LANES
    ]


def test_run_validator_accepts_complete_raw_measurements() -> None:
    validate_run(_run("jp-a800-171", "stock_triton"))


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_run_validator_recomputes_each_median(result_name: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][result_name]["median_seconds_per_invocation"] = 9.0

    with pytest.raises(ValueError, match="median is not reproducible"):
        validate_run(payload)


@pytest.mark.parametrize(
    "field",
    (
        "direct_kernel_speedup_over_pytorch",
        "public_dispatch_speedup_over_eager",
        "public_dispatch_speedup_over_compiled",
    ),
)
def test_run_validator_recomputes_each_speedup(field: str) -> None:
    payload = _run("jp-a800-171", "stock_triton")
    payload["cases"][0][field] = 9.0

    with pytest.raises(ValueError, match=f"{field} is not reproducible"):
        validate_run(payload)


def test_merge_requires_full_host_and_compiler_cross_product() -> None:
    with pytest.raises(ValueError, match="each required host"):
        merge_runs(_matrix()[:-1], required_hosts=("jp-a800-171", "jp-a800-172"))


def test_aggregate_retains_opt_in_when_either_public_baseline_wins() -> None:
    payload = merge_runs(
        _matrix(eager_speedup=1.1, compiled_speedup=0.9),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_dispatch_win_over_eager_on_all_cases"]
    assert not payload["public_dispatch_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "retain_opt_in"


def test_aggregate_allows_default_only_after_both_public_wins() -> None:
    payload = merge_runs(
        _matrix(eager_speedup=1.1, compiled_speedup=1.2),
        required_hosts=("jp-a800-171", "jp-a800-172"),
    )

    validate_evidence(payload)
    assert payload["public_dispatch_win_over_eager_on_all_cases"]
    assert payload["public_dispatch_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"


def test_checked_in_a800_evidence_is_canonical_and_default_eligible() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    validate_evidence(payload)
    assert payload["direct_kernel_win_on_all_cases"]
    assert payload["public_dispatch_win_over_eager_on_all_cases"]
    assert payload["public_dispatch_win_over_compiled_on_all_cases"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"
