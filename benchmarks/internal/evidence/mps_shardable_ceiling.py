"""The speed a matched-speed MPS workload's own arithmetic permits.

The frozen matched-speed contract publishes a ratio between one device and a
sharded leg and compares it against a threshold. Dividing that ratio by the world
size assumes the workload has no serial part, and this one does: the timed step is
dominated by two environment scans that walk every site in a fixed global order,
so most of its per-site cost is a recurrence no partition divides. Against a
partitionable workload ``speedup / world_size`` is the natural efficiency, and
against this one it asks for a fraction that no number of devices can produce, so
the threshold is measured against the ceiling the workload's own serial fraction
implies instead.

The fraction is not asserted here. It is fitted by
``benchmarks/build_mps_shardability_calibration.py`` from a sweep that times every
rung of the frozen ladder at several world sizes, and the manifest freezes both
the fitted number and the digest of the artifact it was read from. This module
owns the arithmetic that turns that one number plus a world size into a ceiling,
and the checks that the frozen number is the one the artifact states, so the
producer, the calibration builder and the release gate cannot disagree about what
the threshold means.

Nothing here reads a measurement of the sharded leg. A ceiling computed from the
leg it is compared against would make the threshold depend on the answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The manifest keys that freeze the calibration. They are named here rather than
# spelled out at each use so a reader can see the whole freeze in one place.
SERIAL_FRACTION_KEY = "measured_serial_fraction"
CALIBRATION_ARTIFACT_KEY = "shardable_calibration_artifact"
CALIBRATION_DIGEST_KEY = "shardable_calibration_artifact_sha256"


def shardable_ceiling_speedup(serial_fraction: float, world_size: int) -> float:
    """Return the speedup a workload with this serial fraction can reach.

    ``serial_fraction`` is the share of the per-site cost that no partition
    divides, so ``world_size`` devices pay ``serial_fraction + (1 -
    serial_fraction) / world_size`` times what one device pays and the speedup is
    the reciprocal. A fraction outside ``[0, 1)`` describes no partitionable cost
    at all, and the world size has to name at least one device, so both are
    refused rather than clamped.
    """

    serial = float(serial_fraction)
    if not 0.0 <= serial < 1.0:
        raise ValueError(
            f"a serial fraction of {serial!r} is not in [0, 1), so it describes no "
            "speedup a partition can approach"
        )
    world = int(world_size)
    if world < 1:
        raise ValueError(f"a world size of {world!r} names no device")
    return 1.0 / (serial + (1.0 - serial) / world)


def frozen_serial_fraction(speed: Mapping[str, Any]) -> float:
    """Return the serial fraction the frozen speed contract declares.

    A contract that names no fraction is not the same as one that names zero: it
    is a contract whose threshold nobody has measured, so it is refused instead of
    being read as a fully partitionable workload.
    """

    if SERIAL_FRACTION_KEY not in speed:
        raise ValueError(
            "the frozen MPS speed contract declares no measured serial fraction, so "
            "its scaling threshold is relative to no measured quantity"
        )
    fraction = float(speed[SERIAL_FRACTION_KEY])
    # The ceiling formula refuses the endpoints for its own reasons; calling it
    # here surfaces the same refusal at the point the contract is read.
    shardable_ceiling_speedup(fraction, 1)
    return fraction


def calibration_errors(
    speed: Mapping[str, Any], root: Path | None = None
) -> tuple[str, ...]:
    """Return how the frozen calibration and the artifact it names disagree.

    The fraction is a measurement rather than a constant written into the
    contract, so the artifact it came from is digest-bound the same way the
    capacity workload's definitions are. A missing artifact, a drifted one, or one
    whose own ``serial_fraction`` differs from the frozen value is returned as its
    own name, and the caller turns a non-empty result into a refusal: a contract
    that could silently keep a number the evidence no longer supports is a
    threshold nobody can re-derive.
    """

    errors: list[str] = []
    try:
        frozen = frozen_serial_fraction(speed)
    except (TypeError, ValueError) as error:
        return (str(error),)

    declared = speed.get(CALIBRATION_ARTIFACT_KEY)
    digest = speed.get(CALIBRATION_DIGEST_KEY)
    if not declared or not digest:
        errors.append(
            "the frozen MPS speed contract names no calibration artifact, so the "
            "serial fraction it declares cannot be re-read"
        )
        return tuple(errors)

    path = Path(str(declared))
    if root is not None:
        path = root / path
    if not path.is_file():
        errors.append(f"speed_workload[{declared}]: absent")
        return tuple(errors)
    if hashlib.sha256(path.read_bytes()).hexdigest() != str(digest):
        errors.append(f"speed_workload[{declared}]: drifted")
        return tuple(errors)

    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("serial_fraction")
    if stated is None:
        errors.append(f"speed_workload[{declared}]: states no serial fraction")
    elif float(stated) != frozen:
        errors.append(
            f"speed_workload[{declared}]: states a serial fraction of {stated!r} "
            f"against the frozen {frozen!r}"
        )
    return tuple(errors)
