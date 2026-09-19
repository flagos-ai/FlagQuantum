"""Rules for deciding whether a hardware run shared its devices with anyone."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from flagquantum.runtime.observability.performance import PerformanceRecord
from tools.evaluate_performance_artifact import (
    CLEAN,
    CONTENDED,
    FOREIGN_MEMORY_FLOOR_MIB,
    UNAVAILABLE,
    DeviceState,
    as_record,
    evaluate_path,
    measurement_environment,
    parse_device_state,
    record_of,
)

pytestmark = pytest.mark.unit

# The shape a real host answers with: one line per device, four columns.
IDLE = "0, 4, 81920, 0\n1, 4, 81920, 0\n"
# Device 1 carries someone else's allocation; device 0 is idle.
DEVICE_ONE_BUSY = "0, 4, 81920, 0\n1, 61440, 81920, 87\n"

RECORD_FIELDS = {item.name for item in fields(PerformanceRecord)}


def _performance_payload(**changes: object) -> dict[str, object]:
    """A healthy record in the shape a benchmark writes and the gate reads.

    Keys that name a `PerformanceRecord` field configure the record; any other
    key rides alongside it, as the benchmark's own extra fields do.
    """

    values: dict[str, object] = {
        "benchmark": "unit",
        "layer": "training_step",
        "backend": "pytorch",
        "device": "cuda:0",
        "distribution_semantics": "single_device_fast_path",
        "world_size": 1,
        "warmup": 2,
        "repetitions": 4,
        "samples_seconds": (1.0, 1.0, 1.0, 1.0),
        "initialization_seconds": 0.1,
        "teardown_seconds": 0.1,
        "peak_memory_allocated_bytes": 100,
        "peak_memory_reserved_bytes": 120,
        "allocated_memory_by_step": (100, 100, 100, 100),
        "reserved_memory_by_step": (120, 120, 120, 120),
        "useful_work_seconds": 4.0,
        "communication_seconds": 0.0,
        "idle_seconds": 0.0,
        "completed_work_units": 4,
        "heartbeat_gaps_seconds": (1.0, 1.0, 1.0, 1.0),
        "correctness_passed": True,
    }
    values.update(
        {name: value for name, value in changes.items() if name in RECORD_FIELDS}
    )
    payload = dict(PerformanceRecord(**values).summary())
    payload.update(
        {name: value for name, value in changes.items() if name not in RECORD_FIELDS}
    )
    return payload


def test_parse_reads_each_device_and_keeps_a_missing_utilization_absent() -> None:
    assert parse_device_state(IDLE) == (
        DeviceState(0, 4, 81920, 0),
        DeviceState(1, 4, 81920, 0),
    )
    # Some virtualized and MIG configurations answer `[N/A]` for utilization
    # while still reporting memory, so the row must survive with a null there.
    assert parse_device_state("0, 4096, 81920, [N/A]\n") == (
        DeviceState(0, 4096, 81920, None),
    )


def test_parse_refuses_a_malformed_answer_instead_of_reading_it_as_no_devices() -> None:
    # An answer this function cannot read must not be mistaken for a host that
    # reported nothing, because the caller treats "nothing" as no evidence.
    with pytest.raises(ValueError, match="expected 4 columns"):
        parse_device_state("0, 4, 81920\n")
    with pytest.raises(ValueError, match="memory.used is not an integer"):
        parse_device_state("0, 4 MiB, 81920, 0\n")


def test_an_idle_device_used_by_this_run_supports_a_claim() -> None:
    result = measurement_environment(IDLE, devices_used=(0, 1))

    assert result.status == CLEAN
    assert result.claim_supported is True
    assert result.contended_devices == ()


def test_a_co_tenant_on_a_device_this_run_uses_withdraws_the_claim() -> None:
    result = measurement_environment(DEVICE_ONE_BUSY, devices_used=(0, 1))

    assert result.status == CONTENDED
    assert result.claim_supported is False
    assert result.contended_devices == (1,)
    assert "device(s) [1]" in result.detail


def test_a_co_tenant_on_a_device_this_run_avoids_leaves_the_claim_alone() -> None:
    # The discrimination that keeps this check from discrediting clean runs:
    # a neighbour on device 1 is not contention for a run confined to device 0.
    result = measurement_environment(DEVICE_ONE_BUSY, devices_used=(0,))

    assert result.status == CLEAN
    assert result.claim_supported is True


def test_the_floor_separates_the_driver_baseline_from_a_real_allocation() -> None:
    at_floor = f"0, {FOREIGN_MEMORY_FLOOR_MIB}, 81920, 0\n"
    above_floor = f"0, {FOREIGN_MEMORY_FLOOR_MIB + 1}, 81920, 0\n"

    assert measurement_environment(at_floor, devices_used=(0,)).status == CLEAN
    assert measurement_environment(above_floor, devices_used=(0,)).status == CONTENDED


def test_an_unreadable_device_state_is_not_read_as_an_idle_one() -> None:
    # The point of the three-state answer. An empty answer, a malformed answer,
    # and an answer that omits the devices we use all mean "not shown to be
    # free", which is not the same claim as "shown to be free".
    for raw in ("", "   \n", "0, 4, 81920\n", "4, 4, 81920, 0\n"):
        result = measurement_environment(raw, devices_used=(0,))
        assert result.status == UNAVAILABLE, raw
        assert result.claim_supported is False, raw
        assert result.contended_devices == (), raw


def test_the_wider_question_considers_every_device_the_host_reports() -> None:
    # A producer that recorded no device list asks whether the host was shared
    # at all, and a busy device anywhere answers yes.
    assert (
        measurement_environment(DEVICE_ONE_BUSY, devices_used=None).status == CONTENDED
    )
    assert measurement_environment(IDLE, devices_used=None).status == CLEAN


def test_a_verdict_renders_as_the_mapping_an_artifact_carries() -> None:
    assert as_record(measurement_environment(DEVICE_ONE_BUSY, devices_used=(0, 1))) == {
        "status": CONTENDED,
        "claim_supported": False,
        "contended_devices": [1],
        "foreign_memory_floor_mib": FOREIGN_MEMORY_FLOOR_MIB,
        "detail": (
            f"another process held more than {FOREIGN_MEMORY_FLOOR_MIB} MiB on "
            "device(s) [1] when this run took its measurements"
        ),
    }


def test_a_recorded_environment_is_judged_from_what_the_producer_wrote() -> None:
    recorded = {
        "measurement_environment": {
            "sampled_before": DEVICE_ONE_BUSY,
            "devices_used": [0, 1],
        }
    }

    assert record_of(recorded).status == CONTENDED
    assert record_of(recorded).contended_devices == (1,)


def test_an_artifact_that_recorded_no_environment_withdraws_the_claim() -> None:
    # Artifacts written before this reading existed carry no device state. An
    # absent record is not a clean record, so it must not read as one.
    for payload in (
        {},
        {"measurement_environment": {}},
        {"measurement_environment": 7},
    ):
        result = record_of(payload)
        assert result.status == UNAVAILABLE, payload
        assert result.claim_supported is False, payload


def test_the_evaluator_records_the_environment_without_turning_the_gate_red(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Contention inflates latency variance, and the gate reports that as
    # `latency_variance_exceeds_threshold`. Folding the environment into the
    # gate would make a busy neighbour look like a code regression, so the two
    # verdicts stay apart: the numbers pass, and the environment says whether
    # believing them is allowed.
    artifact = tmp_path / "performance.json"
    artifact.write_text(
        json.dumps(
            _performance_payload(
                measurement_environment={
                    "sampled_before": DEVICE_ONE_BUSY,
                    "devices_used": [0, 1],
                }
            )
        ),
        encoding="utf-8",
    )

    assert evaluate_path(artifact, write=True) is True

    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["performance_gate"]["passed"] is True
    assert written["measurement_validity"]["status"] == CONTENDED
    assert written["measurement_validity"]["claim_supported"] is False
    assert "::warning" in capsys.readouterr().out


def test_the_evaluator_stays_quiet_about_an_environment_that_supports_a_claim(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "performance.json"
    artifact.write_text(
        json.dumps(
            _performance_payload(
                measurement_environment={"sampled_before": IDLE, "devices_used": [0]}
            )
        ),
        encoding="utf-8",
    )

    assert evaluate_path(artifact, write=True) is True

    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["measurement_validity"]["status"] == CLEAN
    assert written["measurement_validity"]["claim_supported"] is True
    assert "::warning" not in capsys.readouterr().out


def test_a_failed_gate_says_which_artifact_and_which_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate that fails has to leave something in the log to act on.

    The exit code is the only other thing this tool produces on failure, and a
    workflow step that ends on it reads as an empty log. On 2026-09-18 the
    weekly lane's last phase failed exactly that way: the job log ended at the
    line that started the benchmark, and which threshold had been missed was
    only visible after downloading the artifact.
    """
    artifact = tmp_path / "performance.json"
    # The gate judges the median absolute deviation over the median, so half
    # the samples have to sit away from the middle one for it to object.
    artifact.write_text(
        json.dumps(_performance_payload(samples_seconds=(1.0, 1.0, 2.0, 2.0))),
        encoding="utf-8",
    )

    assert evaluate_path(artifact, write=True) is False

    reported = capsys.readouterr().out
    # The environment warning names this artifact too, so the assertion is on
    # the error line itself rather than on the whole of what was written.
    errors = [line for line in reported.splitlines() if line.startswith("::error")]
    assert len(errors) == 1, reported
    assert str(artifact) in errors[0]
    assert "latency_variance_exceeds_threshold" in errors[0]


def test_a_gate_that_passes_reports_no_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "performance.json"
    artifact.write_text(json.dumps(_performance_payload()), encoding="utf-8")

    assert evaluate_path(artifact, write=True) is True

    assert "::error" not in capsys.readouterr().out
