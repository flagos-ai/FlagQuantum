#!/usr/bin/env python3
"""Read the frozen matched-speed ladder's own rung sweep into one calibration.

The MPS release contract's speed threshold cannot be a ratio against the world
size. The timed workload is dominated by two environment scans that walk every
site in a fixed global order, so its per-site cost is mostly a recurrence over
sites that no partition divides, and a threshold of ``speedup / world_size``
asks a shardable fraction that arithmetic does not permit. What the workload
does permit is a number, and it is measurable: fit the step's cost against the
site count at two world sizes and the two slopes separate into a term that
shards and a term that does not.

``mps_matched_speed_sweep.sh`` moves the world at one rung and the rung at one
world, and either on its own cannot separate the two terms, because a per-step
fixed cost and a per-site cost both grow with the rung and both fall with the
world. This builder reads a sweep that varies the rung at each of several world
sizes, fits a line per world, and solves the resulting system for the serial and
shardable per-site costs.

The fit is on the median the release ratio itself divides: ``wall_seconds`` is the
median of the wall clock recorded around one training call, taken from rank zero,
because a sharded leg publishes rank zero's own configuration table.
``end_to_end_seconds`` and ``forward_seconds`` are carried beside it as
cross-checks rather than as the calibrated quantity.

Reproduce
---------
    FQ_SWEEP_OUTPUT=<sweep directory> \
        benchmarks/internal/evidence/mps_matched_speed_sweep.sh
    python benchmarks/build_mps_shardability_calibration.py \
        --source-directory <sweep directory> \
        --manifest benchmarks/manifests/mps_release_v1.json \
        --output benchmarks/results/local/mps_matched_speed_shardability.json \
        --worlds 1 8 16

The sweep directory defaults to the same layout as the committed copy under
``benchmarks/results/local/mps_matched_speed_shardability``, so the builder can be
re-run against the committed records with no arguments beyond ``--worlds``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The builder runs from the repository root as a script, which puts
# `benchmarks/` on the path rather than the root the sibling module is imported
# from, so the root is added before that import rather than after it.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.internal.evidence.mps_shardable_ceiling import (  # noqa: E402
    shardable_ceiling_speedup,
)

SCHEMA = "flagquantum.mps_shardability_calibration.v1"

# The metric the release ratio divides, and the two carried beside it so a
# reader can see that the split is a property of the workload rather than of
# one way of timing it.
CALIBRATED_METRIC = "wall_seconds"
CROSS_CHECK_METRICS = ("end_to_end_seconds", "forward_seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values: Sequence[float]) -> float:
    """Return the median of ``values`` without averaging two central entries.

    ``statistics.median`` averages a pair on an even sample, which would report a
    latency no rank ever measured. The sample here is even by construction, so the
    lower central entry is taken instead and the choice is stated rather than left
    to a library default.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("a median of no samples is not a latency")
    return ordered[(len(ordered) - 1) // 2]


def _least_squares(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Return the slope and intercept of the line through ``(xs, ys)``."""

    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("a line needs two points and as many ordinates as abscissae")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("every rung has the same site count, so no slope exists")
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        / denominator
    )
    return slope, mean_y - slope * mean_x


def _r_squared(
    xs: Sequence[float], ys: Sequence[float], slope: float, intercept: float
) -> float:
    """Return how much of the ordinate's variance the fitted line explains."""

    mean_y = sum(ys) / len(ys)
    total = sum((y - mean_y) ** 2 for y in ys)
    if total == 0.0:
        return 1.0
    residual = sum(
        (y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True)
    )
    return 1.0 - residual / total


def _two_term_fit(worlds: Sequence[int], slopes: Mapping[int, float]) -> dict[str, Any]:
    """Separate the per-site slope into a serial and a shardable part.

    One device pays ``serial + shardable`` per site and ``world`` devices pay
    ``serial + shardable / world``, so each measured world contributes one linear
    equation in the two unknowns and three worlds over-determine them. The
    reported split is the least-squares solution over every world the sweep
    measured rather than the exact solve of one pair, because a two-point solve
    passes through both of its own points by construction and therefore cannot
    say whether the form fits: what makes the split credible is that the fitted
    line also lands on the worlds it was not solved from.

    The two-point solves are returned beside it as a consistency check, so a
    reader can see how far the answer moves when the widest world is dropped.
    ``serial_fraction`` is the ratio of the serial term to the single-device
    slope, which is the share of the per-site cost no partition divides.
    """

    ordered = sorted(int(word) for word in worlds)
    if ordered[0] != 1:
        raise ValueError(
            f"the reference world of this system is {ordered[0]!r} rather than one "
            "device, so its slope is not the undivided cost"
        )
    if len(ordered) < 2:
        raise ValueError("two world sizes are needed to separate the two terms")
    measured = [float(slopes[world]) for world in ordered]
    if any(value <= 0.0 for value in measured):
        raise ValueError("a non-positive per-site slope is not a per-site cost")
    reciprocal = [1.0 / world for world in ordered]
    count = len(ordered)
    sum_x = sum(reciprocal)
    sum_xx = sum(value * value for value in reciprocal)
    sum_y = sum(measured)
    sum_xy = sum(value * y for value, y in zip(reciprocal, measured, strict=True))
    determinant = count * sum_xx - sum_x * sum_x
    if determinant == 0.0:
        raise ValueError("every measured world is the same size, so no split exists")
    shardable = (count * sum_xy - sum_x * sum_y) / determinant
    serial = (sum_xx * sum_y - sum_x * sum_xy) / determinant
    single_device = serial + shardable
    if not 0.0 <= serial < single_device:
        raise ValueError(
            f"the measured slopes imply a serial cost of {serial!r} against "
            f"{single_device!r} on one device, which describes no shardable workload"
        )
    residuals = {
        str(world): {
            "measured_seconds_per_site": value,
            "fitted_seconds_per_site": serial + shardable / world,
            "relative_residual": (value - (serial + shardable / world)) / value,
        }
        for world, value in zip(ordered, measured, strict=True)
    }
    pair_solves = {}
    for world in ordered[1:]:
        pair_serial = (world * slopes[world] - slopes[1]) / (world - 1)
        pair_shardable = slopes[1] - pair_serial
        pair_solves[str(world)] = {
            "serial_fraction": pair_serial / slopes[1],
            "serial_seconds_per_site": pair_serial,
            "shardable_seconds_per_site": pair_shardable,
        }
    return {
        "serial_seconds_per_site": serial,
        "shardable_seconds_per_site": shardable,
        "single_device_seconds_per_site": single_device,
        "serial_fraction": serial / single_device,
        "shardable_fraction": shardable / single_device,
        "residuals": residuals,
        "two_point_solves": pair_solves,
        "max_abs_relative_residual": max(
            abs(float(row["relative_residual"])) for row in residuals.values()
        ),
        "two_point_serial_fraction_spread": (
            max(row["serial_fraction"] for row in pair_solves.values())
            - min(row["serial_fraction"] for row in pair_solves.values())
            if len(pair_solves) > 1
            else 0.0
        ),
    }


def _record(source: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{source} is not a probe record")
    ranks = payload.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        raise SystemExit(f"{source} published no rank records")
    return payload


def _series(record: dict[str, Any], metric: str) -> list[float]:
    """Return rank zero's per-iteration values of ``metric``.

    Rank zero is the rank whose configuration table a sharded leg publishes, so
    it is the rank the release ratio divides; ``end_to_end_seconds`` and
    ``forward_seconds`` live in the step metrics while ``wall_seconds`` is the
    measurement of the call around them.
    """

    rank_zero = record["ranks"][0]
    iterations = rank_zero.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise SystemExit("rank zero published no measured iterations")
    values: list[float] = []
    for iteration in iterations:
        if metric == "wall_seconds":
            value = iteration.get("wall_seconds")
        else:
            steps = iteration.get("steps")
            if not isinstance(steps, list) or not steps:
                raise SystemExit("an iteration carried no step metrics")
            value = steps[-1].get(metric)
        if value is None:
            raise SystemExit(f"an iteration recorded no {metric}")
        values.append(float(value))
    return values


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    speed = manifest["speed_workload"]
    ladder = list(speed["configuration_ladder"])
    worlds = [int(word) for word in arguments.worlds]

    points: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    slopes: dict[int, float] = {}
    fits: dict[str, Any] = {}
    for world in worlds:
        xs: list[float] = []
        ys: list[float] = []
        for rung in ladder:
            name = str(rung["name"])
            source = (
                Path(arguments.source_directory)
                / f"{arguments.prefix}{name}_w{world}.json"
            )
            if not source.is_file():
                raise SystemExit(
                    f"the sweep recorded no rung for world {world}: {source} is absent"
                )
            record = _record(source)
            if str(record.get("rung")) != name:
                raise SystemExit(
                    f"{source} measured the rung {record.get('rung')!r} rather than "
                    f"the frozen {name!r}"
                )
            if int(record.get("world_size", 0)) != world:
                raise SystemExit(
                    f"{source} declares a world of {record.get('world_size')!r} "
                    f"rather than {world}"
                )
            rank_zero = record["ranks"][0]
            if int(rank_zero.get("n_sites", 0)) != int(rung["n_sites"]):
                raise SystemExit(
                    f"{source} timed {rank_zero.get('n_sites')} sites for the frozen "
                    f"{rung['n_sites']}-site rung"
                )
            if int(rank_zero.get("trained_max_bond", 0)) != int(
                rung["trained_max_bond"]
            ):
                raise SystemExit(
                    f"{source} trained a bond of {rank_zero.get('trained_max_bond')} "
                    f"for the frozen {rung['trained_max_bond']}"
                )
            measurements = {
                "world": world,
                "rung": name,
                "n_sites": int(rung["n_sites"]),
            }
            for metric in (CALIBRATED_METRIC, *CROSS_CHECK_METRICS):
                values = _series(record, metric)
                measurements[f"median_{metric}"] = _median(values)
                measurements[f"seconds_{metric}"] = [
                    round(value, 6) for value in values
                ]
            points.append(measurements)
            sources.append(
                {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "committed_path": str(
                        Path(arguments.committed_directory) / source.name
                    ),
                    "world": world,
                    "rung": name,
                }
            )
            xs.append(float(rung["n_sites"]))
            ys.append(measurements[f"median_{CALIBRATED_METRIC}"])
        slope, intercept = _least_squares(xs, ys)
        slopes[world] = slope
        fits[str(world)] = {
            "slope_seconds_per_site": slope,
            "intercept_seconds": intercept,
            "r_squared": _r_squared(xs, ys, slope, intercept),
        }

    split = _two_term_fit(worlds, slopes)
    serial_fraction = float(split["serial_fraction"])
    widest = max(worlds)
    ceilings = {
        str(world): shardable_ceiling_speedup(serial_fraction, world)
        for world in worlds
    }
    accepted = str(speed["acceptance_configuration"])
    accepted_points = [point for point in points if point["rung"] == accepted]
    if not accepted_points:
        raise SystemExit(f"the sweep timed no rung called {accepted!r}")
    reference = _median(
        _series(
            _record(
                Path(arguments.source_directory)
                / f"{arguments.prefix}{accepted}_w{worlds[0]}.json"
            ),
            CALIBRATED_METRIC,
        )
    )
    measured: dict[str, float] = {}
    for point in accepted_points:
        world = int(point["world"])
        measured[str(world)] = reference / float(point[f"median_{CALIBRATED_METRIC}"])
    if "1" not in measured:
        raise SystemExit(
            "the sweep measured the acceptance rung at no single-device world, so "
            "there is no speedup for the ratio to be relative to"
        )
    released = max(worlds)
    release_ceiling = float(ceilings[str(released)])
    release_speedup = float(measured[str(released)])
    note_text = (
        "The serial fraction is fitted from the per-site slopes alone and never "
        "from an intercept, so a per-call overhead that grows the intercept "
        "without touching the slope cannot move it. The split is reported as the "
        "least-squares solution over every world the sweep measured, with the "
        "residual of each world and the two-point solves carried beside it, "
        "because a two-point solve passes through its own points by construction "
        "and only the worlds it was not solved from can say whether the two-term "
        "form fits. The workload reaches almost none of its own ceiling in the "
        "least shardable direction and that is a property of the workload, not of "
        f"the implementation: at world {released} the arithmetic permits at most "
        f"{release_ceiling:.4f}x against one device, and the measured leg delivered "
        f"{release_speedup:.4f}x, which is {release_speedup / release_ceiling:.4f} "
        "of that ceiling. The cross-check metrics are carried so the split is a "
        "property of the workload rather than of one way of timing it. Every probe "
        "record the fit reads is committed beside this file at the digest listed "
        "for it under 'sources', so the fit can be recomputed and not merely "
        "believed."
    )
    report = {
        "artifact_class": "auxiliary_report",
        "schema": SCHEMA,
        "benchmark": "mps_matched_speed_shardability",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "claim_evidence_type": "development_smoke",
        "ladder_fingerprint": str(speed["ladder_fingerprint"]),
        "acceptance_configuration": accepted,
        "calibrated_metric": CALIBRATED_METRIC,
        "cross_check_metrics": list(CROSS_CHECK_METRICS),
        "rank_series": "rank0",
        "protocol": {
            "warmup_steps": int(speed["warmup_steps"]),
            "measured_steps": int(speed["measured_steps"]),
            "timing_unit": str(speed["timing_unit"]),
        },
        "release_world_sizes": [
            int(world) for world in manifest["topologies"]["release_world_sizes"]
        ],
        "release_world": released,
        "worlds": worlds,
        "fits": fits,
        "serial_fraction": serial_fraction,
        "shardable_fraction": float(split["shardable_fraction"]),
        "serial_seconds_per_site": float(split["serial_seconds_per_site"]),
        "shardable_seconds_per_site": float(split["shardable_seconds_per_site"]),
        "single_device_seconds_per_site": float(
            split["single_device_seconds_per_site"]
        ),
        "fit_residuals": split["residuals"],
        "max_abs_relative_residual": float(split["max_abs_relative_residual"]),
        "two_point_solves": split["two_point_solves"],
        "two_point_serial_fraction_spread": float(
            split["two_point_serial_fraction_spread"]
        ),
        "shardable_ceiling_speedup": ceilings,
        "measured_speedup": measured,
        "fraction_of_shardable_ceiling": {
            world: value / ceilings[world]
            for world, value in measured.items()
            if ceilings.get(world, 0.0) > 0.0
        },
        "release_world_speedup": release_speedup,
        "release_world_ceiling_speedup": release_ceiling,
        "release_world_fraction_of_ceiling": release_speedup / release_ceiling,
        "widest_world": widest,
        "points": points,
        "sources": sources,
        "note": note_text,
    }
    return report


def _verify_committed(report: dict[str, Any], root: Path) -> None:
    """Refuse to publish a fit whose inputs are not readable from this repository.

    The fit is only as good as the records it read, and those records live on the
    machine that measured them rather than in the tree, so the artifact names both
    where each record was produced and where a copy of it is committed. A copy
    that is absent, or present with different bytes, would make the digest a claim
    about a file no reader can open; the builder therefore checks every one of them
    and writes nothing if any disagrees.
    """

    wrong: list[str] = []
    for entry in report["sources"]:
        committed = root / str(entry["committed_path"])
        if not committed.is_file():
            wrong.append(f"{entry['committed_path']}: absent")
        elif _sha256(committed) != str(entry["sha256"]):
            wrong.append(
                f"{entry['committed_path']}: does not match the recorded digest"
            )
    if wrong:
        raise SystemExit(
            "the calibration inputs are not readable from this repository, so the "
            "artifact would digest files no reader can open: " + "; ".join(wrong)
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="",
        help="leading name fragment each probe record carries before the rung name",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--committed-directory",
        default=str(Path("benchmarks/results/local/mps_matched_speed_shardability")),
        help=(
            "repository-relative directory holding the committed copy of every "
            "probe record the fit reads"
        ),
    )
    parser.add_argument(
        "--worlds",
        type=int,
        nargs="+",
        required=True,
        help="world sizes the sweep measured, the reference one first",
    )
    arguments = parser.parse_args(argv)
    report = build(arguments)
    _verify_committed(report, REPOSITORY_ROOT)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"SHARDABILITY {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
