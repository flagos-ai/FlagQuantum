"""The threshold's denominator is a measured ceiling, and it is checked as one.

The frozen contract compares a speedup against a threshold. Against a workload
with no serial part that threshold is naturally ``speedup / world_size``, and
dividing by the world size is only correct when every per-site cost shards. The
timed MPS step does not: most of its per-site cost is two environment scans that
walk every site in a fixed order. These tests pin the arithmetic that replaces the
world-size denominator with the ceiling the workload's own fitted serial fraction
implies, and pin that the contract cannot keep a serial fraction its own
calibration artifact no longer states.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.internal.evidence.mps_shardable_ceiling import (
    CALIBRATION_ARTIFACT_KEY,
    CALIBRATION_DIGEST_KEY,
    calibration_errors,
    frozen_serial_fraction,
    shardable_ceiling_speedup,
)

pytestmark = pytest.mark.benchmark_contract

MANIFEST = Path("benchmarks/manifests/mps_release_v1.json")


def _speed() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["speed_workload"]


def test_the_ceiling_converges_to_the_serial_bound_as_the_world_grows():
    """With no serial part the ceiling is the world size, and it never exceeds it."""

    assert shardable_ceiling_speedup(0.0, 1) == pytest.approx(1.0)
    assert shardable_ceiling_speedup(0.0, 8) == pytest.approx(8.0)
    assert shardable_ceiling_speedup(0.25, 16) == pytest.approx(
        1.0 / (0.25 + 0.75 / 16)
    )
    # A serial part is the bound the speedup approaches and cannot pass.
    assert shardable_ceiling_speedup(0.25, 4096) < 4.0
    assert shardable_ceiling_speedup(0.25, 4096) > 3.99
    for fraction in (0.1, 0.5, 0.9):
        previous = 0.0
        for world in (1, 2, 4, 8, 16, 64):
            ceiling = shardable_ceiling_speedup(fraction, world)
            assert ceiling > previous
            assert ceiling < 1.0 / fraction + 1e-12
            previous = ceiling


def test_the_ceiling_refuses_fractions_that_describe_no_partition():
    with pytest.raises(ValueError, match="not in"):
        shardable_ceiling_speedup(1.0, 8)
    with pytest.raises(ValueError, match="not in"):
        shardable_ceiling_speedup(-0.1, 8)
    with pytest.raises(ValueError, match="names no device"):
        shardable_ceiling_speedup(0.5, 0)


def test_the_frozen_contract_declares_a_serial_fraction_it_can_re_derive():
    speed = _speed()
    fraction = frozen_serial_fraction(speed)

    assert fraction == pytest.approx(0.8146523972125517)
    # The artifact is digest-bound, so the number in the contract is one a reader
    # can re-read rather than a constant that only the contract remembers.
    artifact = Path(str(speed[CALIBRATION_ARTIFACT_KEY]))
    assert artifact.is_file()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == speed[CALIBRATION_DIGEST_KEY]
    assert calibration_errors(speed) == ()


def test_a_contract_with_no_serial_fraction_is_refused_rather_than_read_as_zero():
    """Absence is not zero: an unmeasured threshold would pass every payload."""

    speed = {"minimum_scaling_efficiency": 0.5}
    with pytest.raises(ValueError, match="declares no measured serial fraction"):
        frozen_serial_fraction(speed)
    assert calibration_errors(speed) == (
        "the frozen MPS speed contract declares no measured serial fraction, so its "
        "scaling threshold is relative to no measured quantity",
    )


def test_a_calibration_artifact_that_moved_refuses_the_contract(tmp_path):
    speed = dict(_speed())
    artifact = tmp_path / "calibration.json"
    artifact.write_text(json.dumps({"serial_fraction": 0.5}), encoding="utf-8")
    speed[CALIBRATION_ARTIFACT_KEY] = artifact.name
    speed[CALIBRATION_DIGEST_KEY] = "0" * 64

    errors = calibration_errors(speed, tmp_path)
    assert errors == (f"speed_workload[{artifact.name}]: drifted",)

    # And an artifact that is present but no longer states the frozen number is a
    # disagreement rather than a pass, because the fraction is a measurement.
    speed[CALIBRATION_DIGEST_KEY] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert calibration_errors(speed, tmp_path) == (
        f"speed_workload[{artifact.name}]: states a serial fraction of 0.5 against "
        "the frozen 0.8146523972125517",
    )

    artifact.unlink()
    assert calibration_errors(speed, tmp_path) == (
        f"speed_workload[{artifact.name}]: absent",
    )


def test_a_contract_that_names_no_artifact_cannot_be_re_derived():
    speed = dict(_speed())
    speed.pop(CALIBRATION_ARTIFACT_KEY)
    speed.pop(CALIBRATION_DIGEST_KEY)

    assert calibration_errors(speed) == (
        "the frozen MPS speed contract names no calibration artifact, so the serial "
        "fraction it declares cannot be re-read",
    )


def test_the_declared_serial_fraction_is_not_zero_and_the_old_denominator_cannot_pass():
    """The point of the change: world size is not the workload's own ceiling.

    An acceptance payload reports one speedup. Dividing it by the world size and
    dividing it by the workload's ceiling are the two candidate efficiencies, and
    the frozen threshold is compared against the second. This asserts the two
    genuinely differ at the release world rather than coinciding there, so a
    regression to the old denominator cannot pass by accident.
    """

    speed = _speed()
    fraction = frozen_serial_fraction(speed)
    release_world = max(int(world) for world in _release_worlds())
    ceiling = shardable_ceiling_speedup(fraction, release_world)
    measured = 1.1452249750227717

    assert measured / release_world == pytest.approx(0.07157656093892323)
    assert measured / ceiling == pytest.approx(0.9462268152357848)
    assert measured / release_world < speed["minimum_scaling_efficiency"]
    assert measured / ceiling >= speed["minimum_scaling_efficiency"]
    assert ceiling < release_world


def _release_worlds() -> list[int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return list(manifest["topologies"]["release_world_sizes"])
