from __future__ import annotations

import json

from benchmarks.statevector_mlsys_bootstrap import (
    bootstrap_median,
    bootstrap_ratio,
    build_report,
)


def test_bootstrap_median_and_ratio_are_deterministic() -> None:
    median = bootstrap_median([1.0, 2.0, 3.0, 4.0, 5.0], resamples=1000, seed=7)
    ratio = bootstrap_ratio(
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
        resamples=1000,
        seed=7,
    )
    assert median["median_seconds"] == 3.0
    assert median["confidence_interval_seconds"][0] <= 3.0
    assert median["confidence_interval_seconds"][1] >= 3.0
    assert ratio["ratio"] == 2.0
    assert ratio["confidence_interval"][0] <= 2.0
    assert ratio["confidence_interval"][1] >= 2.0


def test_report_flags_nonstationary_external_samples(tmp_path) -> None:
    def write(name: str, benchmark: str, world: int, samples: list[float]):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "world_size": world,
                    "value_and_grad": {"samples_seconds": samples},
                }
            ),
            encoding="utf-8",
        )
        return path

    fq = write("fq.json", "flagquantum_adjoint_value_and_full_gradient", 2, [1, 1, 1])
    tqd = write(
        "tqd.json",
        "torchquantum_dist_invertible_training",
        2,
        [9, 6, 3],
    )
    report = build_report([fq, tqd], resamples=1000, seed=11)
    assert report["ratios"][0]["ratio"] == 6.0
    tqd_point = next(
        point for point in report["points"] if point["system"] == "TorchQuantum-Dist"
    )
    assert tqd_point["stationarity_warning"] is not None
