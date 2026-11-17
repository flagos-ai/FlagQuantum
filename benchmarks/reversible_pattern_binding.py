# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from locked Kickmix circuit_pattern.perf.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_pattern.perf.cc

"""Check and time the five locked physical-column pattern workloads."""

import argparse
import json
from hashlib import sha256
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.artifacts.reversible_pattern import (
    LaneColumn,
    bind_reversible_pattern,
)
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation as Operation,
)

_LANES = range(256)
_THREE = ((0, 1), (2, 3), (2047, -1))
_FOUR = (*_THREE, (1999, -2))
_WORKLOADS = (
    (((2, 3),), "9a331da2875b1135c7ace3d6b0a3b6706a8837711ba4a924ab86dd0b238c1557"),
    (
        ((2, 3), (2047, -1)),
        "92351f3b845f832c02e81106becce0105b13de37d1d93915e62745c79de9168a",
    ),
    (_THREE, "513978261a83c3d244803d015bf01f7732a6bb7e1f3dac8ac9f3765c91a52e6d"),
    (_FOUR, "a659f217fead0d1dc5bc4049ff4267e90f9445146038f80cefd2dc78b16cb5d9"),
    (
        (*_FOUR, (6, 1), (100, 0), (200, 2)),
        "21d3e7f7d0211ad9e625068b2676cbca625c2aec49fe7ae4ff2fafa01d355a53",
    ),
)


def run(iterations: int) -> list[dict[str, int | float | str]]:
    """Verify source operand checksums and report native binding times."""
    data = tuple(index * index + 3 for index in range(2048))
    rows: list[dict[str, int | float | str]] = []
    for layout, expected_digest in _WORKLOADS:
        columns: tuple[LaneColumn, ...] = tuple(
            (
                tuple(data[start + lane * stride] for lane in _LANES)
                if stride
                else data[start]
            )
            for start, stride in layout
        )
        template = tuple(
            Operation("neg", (), condition_bit=index) for index in range(len(layout))
        )
        actual = bind_reversible_pattern(template, _LANES, bits=columns)
        expected_count = len(_LANES) * len(layout)
        digest = sha256(
            ",".join(str(op.condition_bit) for op in actual).encode()
        ).hexdigest()
        if len(actual) != expected_count or digest != expected_digest:
            raise ValueError(
                f"Operand stream differs from the locked {len(layout)}-column workload"
            )
        times = []
        for _ in range(iterations):
            started = perf_counter_ns()
            bound = bind_reversible_pattern(template, _LANES, bits=columns)
            times.append(perf_counter_ns() - started)
            del bound
        rows.append(
            {
                "columns": len(layout),
                "lanes": len(_LANES),
                "operands": expected_count,
                "operand_sha256": digest,
                "native_binding_median_ns": median(times),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    print(json.dumps(run(args.iterations), indent=2))


if __name__ == "__main__":
    main()
