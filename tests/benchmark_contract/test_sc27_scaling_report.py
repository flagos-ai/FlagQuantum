from __future__ import annotations

import importlib.util
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[2]
REPORT = load(ROOT / "paper" / "sc27" / "build_scaling_report.py", "sc27_report")
FIXTURES = load(
    ROOT / "tests" / "benchmark_contract" / "test_sc27_scaling_matrix_audit.py",
    "sc27_scaling_fixtures",
)


def test_builds_deterministic_hierarchical_bootstrap_report() -> None:
    payloads = FIXTURES.matrix("strong")
    for payload in payloads:
        world = payload["world_size"]
        payload["training_step"]["samples_seconds"] = [8.0 / world] * 20
    first = REPORT.build_report(
        payloads, mode="strong", resamples=100, bootstrap_seed=7
    )
    second = REPORT.build_report(
        payloads, mode="strong", resamples=100, bootstrap_seed=7
    )
    assert first == second
    assert first["statistically_complete"] is False
    assert [point["baseline_time_over_point_time"] for point in first["points"]] == [
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    ]
    assert all(
        point["baseline_normalized_efficiency"] == 1.0 for point in first["points"]
    )


def test_submission_report_requires_ten_thousand_resamples() -> None:
    payloads = FIXTURES.matrix("weak")
    report = REPORT.build_report(payloads, mode="weak", resamples=10_000)
    assert report["statistically_complete"] is True
