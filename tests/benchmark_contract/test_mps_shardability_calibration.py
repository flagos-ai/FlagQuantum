"""The matched-speed calibration is recomputed here, not taken on trust.

The release contract's speed threshold is a ratio against the world size, and the
world size is not what the workload is limited by: the timed step is dominated by
two environment scans that walk every site in a fixed global order, so most of its
per-site cost is a recurrence no partition divides. A threshold of
``speedup / world_size`` therefore asks for a fraction of the shardable part that
the arithmetic does not permit, and the calibration is what measures how much is
actually available.

These tests recompute that measurement from the committed probe records with
arithmetic written here rather than importing the builder's own answer, so a
change to the fitted form fails here instead of quietly moving the threshold. The
records themselves are checked to be auxiliary reports, because a diagnostic that
the audit reads as a claim is a claim nobody reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import audit_results
from benchmarks.audit_results import audit_paths
from benchmarks.build_mps_shardability_calibration import (
    CALIBRATED_METRIC,
    CROSS_CHECK_METRICS,
    SCHEMA,
    _two_term_fit,
    _verify_committed,
    shardable_ceiling_speedup,
)

pytestmark = pytest.mark.benchmark_contract

RECORDS = Path("benchmarks/results/local/mps_matched_speed_shardability")
CALIBRATION = RECORDS / "summary.json"
MANIFEST = Path("benchmarks/manifests/mps_release_v1.json")


def _ladder() -> list[dict[str, object]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return list(manifest["speed_workload"]["configuration_ladder"])


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _wall_seconds(world: int, rung: str) -> float:
    payload = json.loads(
        (RECORDS / f"{rung}_w{world}.json").read_text(encoding="utf-8")
    )
    iterations = payload["ranks"][0]["iterations"]
    return _median([float(entry["wall_seconds"]) for entry in iterations])


def _measured_slopes(worlds: list[int]) -> dict[int, float]:
    """Fit each world's own per-site slope, abscissa first then the other world."""

    ladder = _ladder()
    slopes: dict[int, float] = {}
    for world in worlds:
        xs = [float(entry["n_sites"]) for entry in ladder]
        ys = [_wall_seconds(world, str(entry["name"])) for entry in ladder]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        numerator = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
        )
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slopes[world] = numerator / denominator
    return slopes


def test_the_calibration_is_an_auxiliary_report_with_no_claim_on_it():
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["artifact_class"] == "auxiliary_report"
    assert payload["non_release_evidence"] is True
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert audit_paths([CALIBRATION])["invalid_count"] == 0
    assert audit_paths([CALIBRATION])["claimable_count"] == 0


def test_the_committed_records_are_auxiliary_reports_the_audit_does_not_claim():
    """The record directory is a diagnostic input, not a claim, and says so.

    ``audit_paths`` classifies each payload it is handed, so an auxiliary report
    is reported as auxiliary rather than as an invalid claim. The discovery step
    the command line uses is what skips them entirely, and that is asserted too:
    a diagnostic that reaches the release gate is a claim nobody reviewed.
    """

    # The fit lives beside the records it reads, so the record inventory excludes
    # it by name rather than by the directory holding nothing else. A stray
    # payload dropped here still breaks the count below.
    records = sorted(path for path in RECORDS.glob("*.json") if path != CALIBRATION)

    assert CALIBRATION.is_file()
    assert len(records) == len(_ladder()) * 3
    assert audit_results._json_files(RECORDS) == []
    audit = audit_paths(records)
    assert audit["file_count"] == len(records)
    assert audit["invalid_count"] == 0
    assert audit["claimable_count"] == 0
    for record in audit["records"]:
        assert record["status"] == "auxiliary"
        assert record["release_gate_allowed"] is False
    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["artifact_class"] == "auxiliary_report"
        assert payload["scalability_claim_allowed"] is False


def test_the_serial_fraction_is_reproducible_from_the_records_alone():
    """The number the threshold divides is a fit, so the fit is re-derived here."""

    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    worlds = [1, 8, 16]
    slopes = _measured_slopes(worlds)
    reciprocal = [1.0 / world for world in worlds]
    measured = [slopes[world] for world in worlds]
    count = len(worlds)
    sum_x = sum(reciprocal)
    sum_xx = sum(value * value for value in reciprocal)
    sum_y = sum(measured)
    sum_xy = sum(x * y for x, y in zip(reciprocal, measured, strict=True))
    determinant = count * sum_xx - sum_x * sum_x
    shardable = (count * sum_xy - sum_x * sum_y) / determinant
    serial = (sum_xx * sum_y - sum_x * sum_xy) / determinant

    assert payload["serial_fraction"] == pytest.approx(serial / (serial + shardable))
    assert payload["shardable_fraction"] == pytest.approx(
        shardable / (serial + shardable)
    )
    assert payload["serial_fraction"] == pytest.approx(0.8146523972125517)
    # Two terms cannot describe the measurements unless each world's own slope is
    # close to the line fitted through all of them.
    assert payload["max_abs_relative_residual"] < 0.005
    assert [int(world) for world in payload["worlds"]] == worlds
    assert payload["calibrated_metric"] == CALIBRATED_METRIC
    assert payload["cross_check_metrics"] == list(CROSS_CHECK_METRICS)


def test_the_fitted_form_is_the_two_term_one_and_not_the_old_ratio():
    """The superseded three-point algebra put the fraction near 0.766, not 0.815.

    A regression to that form would not fail any other test in this file, because
    it still produces a number in ``[0, 1)`` and still fits three points loosely.
    The solved value is pinned here, and the two-point solves are pinned beside it
    so a reader can see the spread the least-squares answer sits inside.
    """

    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    spread = payload["two_point_serial_fraction_spread"]

    assert payload["serial_fraction"] == pytest.approx(0.8146523972125517)
    assert spread < 0.005
    solves = payload["two_point_solves"]
    assert solves["8"]["serial_fraction"] == pytest.approx(0.813063, abs=1e-5)
    assert solves["16"]["serial_fraction"] == pytest.approx(0.815886, abs=1e-5)
    # The superseded algebra reported roughly 0.766; the corrected one cannot.
    assert payload["serial_fraction"] > 0.8


def test_the_fit_refuses_inputs_that_separate_nothing():
    with pytest.raises(ValueError, match="rather than one device"):
        _two_term_fit([2, 4], {2: 1.0, 4: 0.5})
    with pytest.raises(ValueError, match="two world sizes"):
        _two_term_fit([1], {1: 1.0})
    with pytest.raises(ValueError, match="non-positive per-site slope"):
        _two_term_fit([1, 2], {1: 0.0, 2: 0.5})
    with pytest.raises(ValueError, match="no shardable workload"):
        # A slope that falls faster than the arithmetic permits would need a
        # negative serial term, which no partition of a recurrence produces.
        _two_term_fit([1, 4], {1: 1.0, 4: 0.1})


def test_the_ceiling_is_the_speedup_the_workload_arithmetic_permits():
    assert shardable_ceiling_speedup(0.0, 8) == pytest.approx(8.0)
    assert shardable_ceiling_speedup(0.5, 1) == pytest.approx(1.0)
    assert shardable_ceiling_speedup(0.8146523972125517, 1) == pytest.approx(1.0)
    assert shardable_ceiling_speedup(0.8146523972125517, 8) == pytest.approx(
        1.1935725912182111
    )
    assert shardable_ceiling_speedup(0.8146523972125517, 16) == pytest.approx(
        1.2103070390553239
    )
    # A serial fraction of one is every cost and no partition, which is not a
    # speedup any world size approaches, so it is refused rather than returned
    # as a ceiling of one.
    with pytest.raises(ValueError, match="not in"):
        shardable_ceiling_speedup(1.0, 8)
    with pytest.raises(ValueError, match="names no device"):
        shardable_ceiling_speedup(0.5, 0)


def test_the_release_world_is_the_one_the_frozen_contract_releases():
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    released = [int(world) for world in manifest["topologies"]["release_world_sizes"]]

    assert payload["release_world_sizes"] == released
    assert payload["release_world"] == max(released)
    assert payload["widest_world"] == max(released)
    assert (
        payload["ladder_fingerprint"]
        == manifest["speed_workload"]["ladder_fingerprint"]
    )
    # The point of the fit: the threshold the contract froze is compared against
    # the arithmetic ceiling at the world the contract releases.
    assert payload["release_world_fraction_of_ceiling"] == pytest.approx(
        payload["release_world_speedup"] / payload["release_world_ceiling_speedup"]
    )
    assert payload["release_world_fraction_of_ceiling"] > 0.9


def test_the_calibration_names_a_committed_copy_of_every_record_it_read():
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    root = Path(".")

    assert len(payload["sources"]) == len(_ladder()) * 3
    for entry in payload["sources"]:
        committed = Path(entry["committed_path"])
        assert committed.is_file()
        assert audit_paths([committed])["claimable_count"] == 0
    # Every source is readable from this repository, which is what makes the fit
    # checkable by a reader rather than merely reported.
    _verify_committed(payload, root)


def test_the_builder_refuses_a_source_whose_bytes_moved(tmp_path):
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    entry = payload["sources"][0]
    target = tmp_path / entry["committed_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="does not match the recorded digest"):
        _verify_committed(payload, tmp_path)

    # And an absent copy is refused for the same reason rather than skipped.
    target.unlink()
    with pytest.raises(SystemExit, match="absent"):
        _verify_committed(payload, tmp_path)
