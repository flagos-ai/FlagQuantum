#!/usr/bin/env python3
"""Re-evaluate committed performance artifacts against hardware thresholds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from flagquantum.runtime.observability.performance import (
    PerformanceRecord,
    PerformanceThresholds,
    calibrate_cost_model,
    evaluate_performance,
)

# The columns a producer is expected to have asked `nvidia-smi` for, in order.
# Three files carry a copy of this literal because every tool in this repository
# is a self-contained script: a script under `tools/` cannot name a sibling
# module, so the reading lives here and `tests/unit/test_hardware_lane_policy.py`
# holds all three copies to each other. A query and a parser that drift apart
# yield an empty sample rather than an error, and an empty sample is
# indistinguishable from an idle host.
DEVICE_QUERY = "index,memory.used,memory.total,utilization.gpu"

# An idle device still reports a few MiB held by the driver, while a co-tenant
# framework allocates orders of magnitude more. One GiB separates the two
# without having to model any particular framework's baseline.
FOREIGN_MEMORY_FLOOR_MIB = 1024

CLEAN = "clean"
CONTENDED = "contended"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeviceState:
    """One device's occupancy at the instant it was sampled."""

    index: int
    memory_used_mib: int
    memory_total_mib: int
    # Some virtualized and MIG configurations answer `[N/A]` for utilization
    # while still reporting memory, so this alone may be absent.
    utilization_percent: int | None


@dataclass(frozen=True)
class MeasurementEnvironment:
    """Whether the environment a run took its numbers in supports a claim.

    ``unavailable`` is deliberately not ``clean``. A run that never read its
    device state has not shown the devices were free, so it does not support a
    claim: anything reading as "no evidence of contention" must not be allowed
    to pass for "evidence of no contention".
    """

    status: str
    claim_supported: bool
    contended_devices: tuple[int, ...]
    floor_mib: int
    detail: str


def parse_device_state(raw: str) -> tuple[DeviceState, ...]:
    """Interpret the CSV body of an ``--query-gpu`` answer.

    Parameters
    ----------
    raw:
        The stdout of an ``nvidia-smi --query-gpu=<DEVICE_QUERY>
        --format=csv,noheader,nounits`` call.

    Raises
    ------
    ValueError
        If a line carries a different number of columns, or a field that must
        be an integer is not one. A malformed answer is refused rather than
        read as an empty device list, so the caller can fail closed.
    """

    devices: list[DeviceState] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) != 4:
            raise ValueError(
                f"line {number}: expected 4 columns, found {len(fields)}: {stripped!r}"
            )
        utilization = fields[3]
        devices.append(
            DeviceState(
                index=_integer(fields[0], number=number, column="index"),
                memory_used_mib=_integer(
                    fields[1], number=number, column="memory.used"
                ),
                memory_total_mib=_integer(
                    fields[2], number=number, column="memory.total"
                ),
                utilization_percent=(
                    None
                    if utilization.upper().startswith("[N/A")
                    else _integer(utilization, number=number, column="utilization.gpu")
                ),
            )
        )
    return tuple(devices)


def measurement_environment(
    raw: str,
    *,
    devices_used: Sequence[int] | None,
    floor_mib: int = FOREIGN_MEMORY_FLOOR_MIB,
) -> MeasurementEnvironment:
    """Report whether a run took its numbers on a contended environment.

    Only the devices the run itself uses decide the answer. A neighbour
    occupying a device the run steps around is not contention, and treating it
    as such would discredit runs that were in fact clean.

    Parameters
    ----------
    raw:
        A recorded device-state answer, or any answer in the same format.
    devices_used:
        The device indices the run used, or ``None`` when the producer did not
        record them, in which case every device the host reports is considered.
    floor_mib:
        Memory a device may hold before it counts as carrying a co-tenant.
    """

    if not raw.strip():
        return MeasurementEnvironment(
            status=UNAVAILABLE,
            claim_supported=False,
            contended_devices=(),
            floor_mib=floor_mib,
            detail=(
                "no device state was recorded, so the devices are not known "
                "to have been free"
            ),
        )
    try:
        devices = parse_device_state(raw)
    except ValueError as exc:
        return MeasurementEnvironment(
            status=UNAVAILABLE,
            claim_supported=False,
            contended_devices=(),
            floor_mib=floor_mib,
            detail=f"device state could not be interpreted: {exc}",
        )
    if not devices:
        return MeasurementEnvironment(
            status=UNAVAILABLE,
            claim_supported=False,
            contended_devices=(),
            floor_mib=floor_mib,
            detail="the device-state answer named no devices",
        )

    if devices_used is None:
        considered = devices
    else:
        wanted = set(devices_used)
        considered = tuple(device for device in devices if device.index in wanted)
    if not considered:
        return MeasurementEnvironment(
            status=UNAVAILABLE,
            claim_supported=False,
            contended_devices=(),
            floor_mib=floor_mib,
            detail=(
                "the device-state answer named none of the devices this run "
                f"used: {sorted(set(devices_used or ()))}"
            ),
        )

    contended = tuple(
        device.index for device in considered if device.memory_used_mib > floor_mib
    )
    if contended:
        return MeasurementEnvironment(
            status=CONTENDED,
            claim_supported=False,
            contended_devices=contended,
            floor_mib=floor_mib,
            detail=(
                f"another process held more than {floor_mib} MiB on device(s) "
                f"{list(contended)} when this run took its measurements"
            ),
        )
    return MeasurementEnvironment(
        status=CLEAN,
        claim_supported=True,
        contended_devices=(),
        floor_mib=floor_mib,
        detail=(
            f"every device this run used held at most {floor_mib} MiB, which is "
            "consistent with an idle device"
        ),
    )


def record_of(payload: Mapping[str, object]) -> MeasurementEnvironment:
    """Judge the environment a producer recorded in an artifact payload."""

    recorded = payload.get("measurement_environment")
    if not isinstance(recorded, Mapping):
        return measurement_environment("", devices_used=None)
    raw = recorded.get("sampled_before")
    devices = recorded.get("devices_used")
    return measurement_environment(
        raw if isinstance(raw, str) else "",
        devices_used=(
            tuple(int(device) for device in devices)
            if isinstance(devices, list)
            else None
        ),
    )


def as_record(result: MeasurementEnvironment) -> dict[str, object]:
    """Render a verdict as the JSON-ready mapping artifacts carry."""

    return {
        "status": result.status,
        "claim_supported": result.claim_supported,
        "contended_devices": list(result.contended_devices),
        "foreign_memory_floor_mib": result.floor_mib,
        "detail": result.detail,
    }


def _integer(field: str, *, number: int, column: str) -> int:
    try:
        return int(field)
    except ValueError as exc:
        raise ValueError(
            f"line {number}: {column} is not an integer: {field!r}"
        ) from exc


def _record(payload: dict[str, object]) -> PerformanceRecord:
    names = {item.name for item in fields(PerformanceRecord)}
    return PerformanceRecord(**{name: payload[name] for name in names})


def evaluate_path(
    path: Path, *, write: bool, baseline_path: Path | None = None
) -> bool:
    payload = json.loads(path.read_text())
    record = _record(payload)
    baseline = (
        _record(json.loads(baseline_path.read_text()))
        if baseline_path is not None
        else None
    )
    thresholds = PerformanceThresholds(
        max_coefficient_of_variation=0.25 if record.world_size >= 8 else 0.10
    )
    payload.update(record.summary())
    payload["cost_model_calibration"] = calibrate_cost_model(record).__dict__
    payload["regression_thresholds"] = thresholds.__dict__
    payload["performance_gate"] = evaluate_performance(
        record, baseline=baseline, thresholds=thresholds
    ).__dict__
    payload["comparison_baseline"] = (
        str(baseline_path) if baseline_path is not None else None
    )
    # The environment verdict is recorded rather than folded into the gate.
    # A contended run fails `latency_variance_exceeds_threshold` exactly as a
    # regressed one does, and letting contention turn the gate red would make a
    # busy neighbour look like a code regression. The gate keeps judging the
    # numbers; this says whether the numbers were taken somewhere that allows
    # believing them, which is the question the gate cannot answer on its own.
    environment = record_of(payload)
    payload["measurement_validity"] = as_record(environment)
    if not environment.claim_supported:
        print(f"::warning title=measurement environment::{path}: {environment.detail}")
    if write:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return bool(payload["performance_gate"]["passed"])


def resolve_optional_baseline(path: Path | None) -> Path | None:
    """Return an optional baseline only when its artifact is available."""
    if path is None or path.exists():
        return path
    print(
        f"performance baseline not present: {path}; "
        "evaluating intrinsic thresholds only"
    )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument("--baseline", type=Path)
    baseline_group.add_argument("--baseline-if-present", type=Path)
    args = parser.parse_args()
    requested_baseline = args.baseline or args.baseline_if_present
    if requested_baseline is not None and len(args.paths) != 1:
        parser.error("a baseline requires exactly one current artifact")
    baseline = (
        args.baseline
        if args.baseline is not None
        else resolve_optional_baseline(args.baseline_if_present)
    )
    passed = all(
        evaluate_path(path, write=args.write, baseline_path=baseline)
        for path in args.paths
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
